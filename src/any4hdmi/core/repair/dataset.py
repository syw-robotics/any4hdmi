from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Lock, Thread

import numpy as np

from any4hdmi.core.format import MOTIONS_SUBDIR, find_dataset_root, load_manifest, load_motion, save_motion
from any4hdmi.core.model import load_model
from any4hdmi.core.repair import RepairCancelled, RepairSettings, inspect_motion, repair_motion, validate_motion
from any4hdmi.utils.mjcf import qpos_names_from_model


def motion_path(path):
    path = Path(path).expanduser().absolute()
    # HF motion symlinks point into blobs; preserve the file's dataset location.
    return path.parent.resolve() / path.name


def select_motions(source: Path) -> list[Path]:
    source = motion_path(source)
    manifest = load_manifest(find_dataset_root(source))
    motions = (manifest.root / manifest.payload.get('motions_subdir', MOTIONS_SUBDIR)).resolve()
    if source.is_file() and source.suffix == '.npz' and source.is_relative_to(motions):
        return [source]
    if source == manifest.root:
        source = motions
    if source.is_dir() and source.is_relative_to(motions):
        paths = sorted(source.rglob('*.npz'))
        if paths:
            return paths
    raise ValueError('Select a dataset, nonempty motion subdirectory, or motion .npz')


class RepairSession:
    """One sequential background job, with only the active clip held in memory."""

    def __init__(self, paths, settings=RepairSettings()):
        self.paths = tuple(dict.fromkeys(motion_path(path) for path in paths))
        if not self.paths:
            raise ValueError('Select at least one motion')
        self.manifest = load_manifest(find_dataset_root(self.paths[0]))
        self.motions_root = (self.manifest.root / self.manifest.payload.get('motions_subdir', MOTIONS_SUBDIR)).resolve()
        if any(not path.is_relative_to(self.motions_root) for path in self.paths):
            raise ValueError('Selected motions must belong to one dataset')
        self.settings = settings
        self._scratch = TemporaryDirectory(prefix='any4hdmi-repair-')
        self.directory = Path(self._scratch.name)
        self._results = {}
        self._sources = {}
        self._candidates = {}
        self._lock = Lock()
        self._cancel = Event()
        self._thread = None
        self._current = None
        self._running = False
        self._check_only = False

    def start(self, check_only=False):
        with self._lock:
            if self._thread is not None:
                raise RuntimeError('This job has already started; create a new session to retry')
            self._check_only = check_only
            self._running = True
            self._thread = Thread(target=self._run, name='motion-repair', daemon=True)
            self._thread.start()

    def _process(self, model, path, index):
        qpos = load_motion(path)
        validate_motion(model, qpos, self.manifest.timestep)
        source = self.directory / f'{index}.source.npy'
        np.save(source, qpos)
        with self._lock:
            self._sources[path] = source
        if self._check_only:
            return inspect_motion(model, qpos, self.manifest.timestep, self.settings, self._cancel.is_set)
        candidate, report = repair_motion(model, qpos, self.manifest.timestep, self.settings, self._cancel.is_set)
        output = self.directory / f'{index}.candidate.npy'
        np.save(output, candidate)
        with self._lock:
            self._candidates[path] = output
        return report

    def _run(self):
        try:
            model = load_model(self.manifest.mjcf_path)
            if self.manifest.payload['qpos_names'] != qpos_names_from_model(model):
                raise ValueError('Manifest qpos_names do not match the model')
            if self.manifest.qpos_dim != model.nq:
                raise ValueError('Manifest qpos_dim does not match the model')
            for index, path in enumerate(self.paths):
                if self._cancel.is_set():
                    break
                with self._lock:
                    self._current = path
                try:
                    report = self._process(model, path, index)
                except RepairCancelled:
                    break
                except Exception as exc:
                    # A corrupt clip or failed solve must not discard the rest of a batch.
                    report = dict(outcome='failed', reasons=[str(exc)], operation='check' if self._check_only else 'repair')
                with self._lock:
                    self._results[path] = report
        except Exception as exc:
            with self._lock:
                self._results = {path: dict(outcome='failed', reasons=[str(exc)]) for path in self.paths}
        finally:
            with self._lock:
                self._running = False
                self._current = None

    def snapshot(self):
        with self._lock:
            return dict(running=self._running, current=self._current, total=len(self.paths),
                        cancelled=self._cancel.is_set(), check_only=self._check_only, results=dict(self._results))

    def source(self, path):
        with self._lock:
            file = self._sources.get(motion_path(path))
        return np.load(file) if file else None

    def candidate(self, path):
        with self._lock:
            file = self._candidates.get(motion_path(path))
        return np.load(file) if file else None

    def cancel(self):
        self._cancel.set()

    def wait(self):
        if self._thread is not None:
            self._thread.join()

    def close(self):
        self.cancel()
        self.wait()
        self._scratch.cleanup()

    def export(self, paths, output_root, *, passed_only=False):
        paths = tuple(dict.fromkeys(motion_path(path) for path in paths))
        state = self.snapshot()
        if state['check_only']:
            raise ValueError('Check-only results cannot be exported; run Repair first')
        if not paths or any(path not in state['results'] for path in paths):
            raise ValueError('Select completed results; unfinished motions cannot be exported')
        output_root = Path(output_root).expanduser().resolve()
        if output_root.exists():
            raise FileExistsError(f'Output directory already exists: {output_root}')
        if output_root.is_relative_to(self.manifest.root):
            raise ValueError('Output directory must be outside the source dataset')
        report, files = self._export_selection(paths, state['results'], passed_only)
        if not files:
            raise ValueError('No motions qualify for export')
        output_root.mkdir(parents=True, exist_ok=False)
        frames = 0
        for relative, file in files:
            qpos = np.load(file)
            save_motion(output_root / MOTIONS_SUBDIR / relative, qpos)
            frames += len(qpos)
        payload = dict(self.manifest.payload)
        if isinstance(self.manifest.mjcf, Path):
            payload['mjcf'] = str(self.manifest.mjcf)
        payload.update(dataset_name=f'{self.manifest.dataset_name}_repaired', motions_subdir=MOTIONS_SUBDIR,
                       num_motions=len(files), total_hours=frames * self.manifest.timestep / 3600,
                       source=dict(dataset=str(self.manifest.root), repair=asdict(self.settings),
                                   original_source=payload.get('source', {})))
        report_path = output_root / 'repair_report.json'
        report_path.write_text(json.dumps(dict(settings=asdict(self.settings), motions=report,
                                              policy='passed_only' if passed_only else 'preserve_selection'), indent=2) + '\n')
        # The manifest is the publication marker; interrupted exports stay invalid.
        (output_root / 'manifest.json').write_text(json.dumps(payload, indent=2) + '\n')
        return report_path

    def _export_selection(self, paths, results, passed_only):
        report, files = {}, []
        for path in paths:
            item = dict(results[path])
            outcome = item['outcome']
            relative = path.relative_to(self.motions_root).as_posix()
            if passed_only and outcome not in ('passed', 'unchanged'):
                item['exported'] = 'excluded'
                report[relative] = item
                continue
            use_candidate = outcome == 'passed'
            file = (self._candidates if use_candidate else self._sources).get(path)
            if file is None:
                raise ValueError(f'No valid source for {relative}; exclude it or use passed-only export')
            item['exported'] = 'candidate' if use_candidate else 'original'
            item['exported_metrics'] = item.get('after' if use_candidate else 'before')
            report[relative] = item
            files.append((relative, file))
        return report, files
