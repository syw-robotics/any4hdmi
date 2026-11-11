"""Persistent, cancellable whole-dataset screening and audited whitelist export."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
from threading import Event, RLock, Thread
from uuid import uuid4

import mujoco
import numpy as np

from any4hdmi.core.format import load_manifest
from any4hdmi.core.selection import FEATURE_VERSION, SelectionPolicy, classify_features, extract_features


def _digest(value):
    return hashlib.sha256(value).hexdigest()


def _json_digest(value):
    return _digest(json.dumps(value, sort_keys=True).encode())


def _model_digest(model):
    buffer = np.empty(mujoco.mj_sizeModel(model), dtype=np.uint8)
    mujoco.mj_saveModel(model, None, buffer)
    return _digest(buffer.tobytes())


def _write_json(path, payload):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


class SelectionProject:
    def __init__(self, input_root, project_root, *, torso_body=None):
        self.input_root = Path(input_root).resolve()
        self.project_root = Path(project_root).resolve()
        self.manifest = load_manifest(self.input_root)
        self.motions_root = self.input_root / self.manifest.payload.get('motions_subdir', 'motions')
        self.paths = {p.relative_to(self.motions_root).as_posix(): p for p in sorted(self.motions_root.rglob('*.npz'))}
        if not self.paths:
            raise ValueError('Dataset contains no motions')
        self.names = list(self.paths)
        self.model = mujoco.MjModel.from_xml_path(str(self.manifest.mjcf_path))
        self.torso_body = torso_body
        self.identity = dict(input_root=str(self.input_root), manifest_sha256=_digest(self.manifest.path.read_bytes()),
                             model_sha256=_model_digest(self.model), torso_body=torso_body, feature_version=FEATURE_VERSION)
        self.policy = SelectionPolicy()
        self.results, self.overrides = {}, {}
        self._lock = RLock()
        self._cancel = Event()
        self._thread = None
        self._running = False
        self._exporting = False
        self._revision = 0
        self._completed_jobs = 0
        self._verified = False
        self.done = 0
        self.total = len(self.names)
        self.message = 'Ready. Batch filter all motions to verify this dataset.'
        self.project_root.mkdir(parents=True, exist_ok=True)
        (self.project_root / 'features').mkdir(exist_ok=True)
        self.state_path = self.project_root / 'session.json'
        if self.state_path.exists():
            saved = json.loads(self.state_path.read_text())
            old_identity = saved['identity']
            if old_identity['input_root'] != str(self.input_root) or old_identity.get('torso_body') != torso_body:
                raise ValueError('Project belongs to a different source or torso mapping; choose a new project directory')
            self.policy = SelectionPolicy(**saved['policy'])
            if old_identity == self.identity:
                self.results = {k: v for k, v in saved['results'].items() if k in self.paths}
                self.overrides = {k: v for k, v in saved['overrides'].items() if k in self.paths}
        self._save()

    def _save(self):
        _write_json(self.state_path, dict(identity=self.identity, policy=asdict(self.policy),
                                         results=self.results, overrides=self.overrides))

    def set_policy(self, policy):
        with self._lock:
            self.policy = policy
            self._revision += 1
            self._verified = False
            self.message = 'Rules changed. Preview or batch filter before exporting.'
            self._save()

    def row(self, name):
        with self._lock:
            result = deepcopy(self.results.get(name, {}))
            current = result.get('policy') == asdict(self.policy)
            override = self.overrides.get(name)
            if override and override['source_sha256'] != result.get('source_sha256'):
                override = None
            status = result.get('auto_status', 'pending') if current else 'pending'
            if override and current:
                status = override['status']
            return dict(result, motion=name, status=status, override=deepcopy(override), stale=not current)

    def snapshot(self):
        with self._lock:
            counts = Counter()
            durations = Counter()
            policy = asdict(self.policy)
            for name in self.names:
                row = self.results.get(name, {})
                status = row.get('auto_status', 'pending') if row.get('policy') == policy else 'pending'
                override = self.overrides.get(name)
                if status != 'pending' and override and override['source_sha256'] == row.get('source_sha256'):
                    status = override['status']
                counts[status] += 1
                durations[status] += row.get('duration_s', 0)
            ready = self._verified and not self._running and not self._exporting and counts['pending'] == 0
            return dict(running=self._running, completed_jobs=self._completed_jobs, done=self.done, total=self.total, message=self.message,
                        counts={s: counts[s] for s in ('keep', 'reject', 'review', 'pending')},
                        durations={s: durations[s] for s in ('keep', 'reject', 'review', 'pending')},
                        can_export=ready and counts['keep'] > 0)

    def set_override(self, name, status, note=''):
        with self._lock:
            row = self.row(name)
            if status is None:
                self.overrides.pop(name, None)
            else:
                if status not in ('keep', 'reject', 'review') or row['stale']:
                    raise ValueError('Evaluate the motion under current rules before overriding')
                if status == 'keep' and row.get('error'):
                    raise ValueError('Invalid motion cannot be manually kept')
                self.overrides[name] = dict(status=status, note=note, source_sha256=row['source_sha256'])
            self._revision += 1
            self._save()

    def _evaluate(self, name, policy):
        source_hash = None
        try:
            raw = self.paths[name].read_bytes()
            source_hash = _digest(raw)
            cache_key = _json_digest(dict(identity=self.identity, source_sha256=source_hash))
            cache_path = self.project_root / 'features' / f'{cache_key}.npz'
            features = None
            if cache_path.exists():
                try:
                    with np.load(cache_path, allow_pickle=False) as saved:
                        features = {key: saved[key] for key in saved.files}
                except (ValueError, OSError, EOFError):
                    features = None
            if features is None:
                with np.load(io.BytesIO(raw), allow_pickle=False) as motion:
                    qpos = np.asarray(motion['qpos'])
                features = extract_features(self.model, qpos, timestep=self.manifest.timestep, torso_body=self.torso_body)
                temporary = cache_path.with_suffix('.tmp')
                with temporary.open('wb') as out:
                    np.savez_compressed(out, **features)
                temporary.replace(cache_path)
            result = classify_features(features, policy)
        except Exception as error:
            result = dict(auto_status='review', reasons=['invalid_motion'], events=[], error=f'{type(error).__name__}: {error}',
                          num_frames=0, fps=1 / self.manifest.timestep, duration_s=0)
        result.update(source_sha256=source_hash, policy=asdict(policy))
        with self._lock:
            old_override = self.overrides.get(name)
            if old_override and old_override['source_sha256'] != source_hash:
                self.overrides.pop(name)
            self.results[name] = result
        return result

    def start_batch(self):
        self._start(self.names, batch=True)

    def preview(self, name):
        self._start([name], batch=False)

    def _start(self, names, *, batch):
        with self._lock:
            if self._running or self._exporting:
                raise RuntimeError('A filtering or export job is already running')
            if any(name not in self.paths for name in names):
                raise ValueError('Motion is not part of this dataset')
            self._running, self._verified = True, False
            self._cancel.clear()
            self.done, self.total = 0, len(names)
            policy = self.policy
            self.message = 'Batch filtering all motions' if batch else 'Previewing current motion'
            self._thread = Thread(target=self._run, args=(list(names), policy, batch), daemon=True)
            self._thread.start()

    def _run(self, names, policy, batch):
        try:
            for name in names:
                if self._cancel.is_set():
                    break
                self._evaluate(name, policy)
                with self._lock:
                    self.done += 1
                    self.message = f'Filtered {self.done}/{self.total}: {name}'
                    # Checkpoint periodically rather than rewriting an ever-growing
                    # JSON report for every clip in a large dataset.
                    if self.done % 100 == 0:
                        self._save()
            with self._lock:
                self._verified = batch and self.done == len(names) and policy == self.policy and not self._cancel.is_set()
                self.message = ('Cancelled; run batch filtering to export.' if self._cancel.is_set() else
                                'Complete.' if self._verified else 'Preview complete or rules changed; batch filtering required for export.')
        except Exception as error:
            with self._lock:
                self.message = f'Filtering failed: {error}'
                self._verified = False
        finally:
            with self._lock:
                self._running = False
                self._completed_jobs += 1
                self._save()

    def cancel(self):
        self._cancel.set()

    def wait(self):
        thread = self._thread
        if thread is not None:
            thread.join()

    def close(self):
        self.cancel()
        self.wait()

    def export(self):
        with self._lock:
            summary = self.snapshot()
            if not summary['can_export']:
                raise ValueError('Complete a current batch with a non-empty keep whitelist before exporting')
            revision = self._revision
            rows = [self.row(name) for name in self.names]
            policy = asdict(self.policy)
            self._exporting = True
        try:
            # Source verification can be slow. Never hold the UI/session lock here.
            current_names = sorted(p.relative_to(self.motions_root).as_posix() for p in self.motions_root.rglob('*.npz'))
            if current_names != self.names or _digest(self.manifest.path.read_bytes()) != self.identity['manifest_sha256']:
                raise ValueError('Dataset inventory or manifest changed; reopen the project')
            current_model = mujoco.MjModel.from_xml_path(str(self.manifest.mjcf_path))
            if _model_digest(current_model) != self.identity['model_sha256']:
                raise ValueError('Robot model changed; reopen the project and run batch filtering')
            for row in rows:
                try:
                    source_hash = _digest(self.paths[row['motion']].read_bytes())
                except OSError:
                    source_hash = None
                if source_hash != row['source_sha256']:
                    raise ValueError(f"Source changed: {row['motion']}; run batch filtering again")
            kept = [r['motion'] for r in rows if r['status'] == 'keep']
            export_id = uuid4().hex
            exports = self.project_root / 'exports'
            exports.mkdir(exist_ok=True)
            temporary = exports / f'.{export_id}.tmp'
            temporary.mkdir()
            payload = dict(schema_version=1, identity=self.identity, policy=policy, summary=summary, motions=rows)
            _write_json(temporary / 'selection_manifest.json', payload)
            (temporary / 'filenames.txt').write_text('\n'.join(kept) + '\n')
            destination = exports / export_id
            with self._lock:
                if self._revision != revision:
                    raise ValueError('Rules or manual decisions changed during export; export again after batch filtering')
                temporary.rename(destination)
                self.message = f'Exported {len(kept)} motions to {destination}'
            return destination
        except Exception:
            with self._lock:
                self._verified = False
            raise
        finally:
            with self._lock:
                self._exporting = False
