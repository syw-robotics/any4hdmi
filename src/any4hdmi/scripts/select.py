"""Standalone Viser screening workspace for any4hdmi motion datasets."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from queue import Empty, SimpleQueue
from threading import Thread
import time

import mujoco
import numpy as np

from any4hdmi.core.format import load_motion
from any4hdmi.core.selection import SelectionPolicy
from any4hdmi.core.selection.project import SelectionProject
from any4hdmi.dataset.loading import resolve_input_paths


class SelectionViewer:
    PAGE_SIZE = 100

    def __init__(self, server, project, scene):
        self.project, self.scene = project, scene
        self.ground = server.scene.add_grid("/selection-ground", position=(0., 0., project.policy.ground_z))
        # Playback never shares mutable MjData with background extraction.
        self.data = mujoco.MjData(project.model)
        self.queue = SimpleQueue()
        self.qpos, self.current = None, None
        self.cursor, self.bounds, self.playing = 0, (0, 0), False
        self.next_time, self.last_poll = 0., 0.
        self.page = 0
        self.previous_running = False
        self.completed_jobs = 0
        self.export_thread = None
        self.export_message = None
        self.events = {}
        gui = server.gui
        gui.configure_theme(control_width='large')
        gui.add_markdown('## Motion selection\nJumps and crawling are allowed. Rules screen inversions and ground rolls. '
                         '**Provisional thresholds: review borderline examples before training.**')
        with gui.add_folder('Batch filtering', order=0):
            self.batch = gui.add_button('Batch filter all motions', color='green', hint='Always processes the entire source dataset, regardless of list filters.')
            self.cancel = gui.add_button('Cancel batch', disabled=True)
            self.progress = gui.add_progress_bar(0.)
            self.summary = gui.add_markdown('')
            self.message = gui.add_text('Job status', initial_value=project.message, disabled=True)
        with gui.add_folder('Rules', order=1):
            p = project.policy
            self.rules = {
                'inversion': gui.add_dropdown('Inversion action', ('reject', 'review', 'allow'), initial_value=p.inversion),
                'ground_roll': gui.add_dropdown('Ground roll action', ('reject', 'review', 'allow'), initial_value=p.ground_roll),
                'inversion_deg': gui.add_slider('Inversion angle (deg)', min=91., max=180., step=1., initial_value=p.inversion_deg),
                'min_duration_s': gui.add_number('Minimum inversion duration (s)', initial_value=p.min_duration_s, min=.01, max=2., step=.01),
                'roll_deg': gui.add_number('Roll rotation (deg)', initial_value=p.roll_deg, min=30., max=720., step=5.),
                'roll_window_s': gui.add_number('Roll window (s)', initial_value=p.roll_window_s, min=.1, max=10., step=.1),
                'ground_height_m': gui.add_number('Near-ground root height (m)', initial_value=p.ground_height_m, min=.01, max=2., step=.01),
                'ground_z': gui.add_number('Ground Z (m)', initial_value=p.ground_z, step=.01),
            }
            self.preview = gui.add_button('Preview current motion')
            gui.add_markdown('Changing rules marks results pending. Batch filtering reuses cached features. '
                             'Near-ground height measures the free-root origin, not the sole.')
        with gui.add_folder('Browse motions', order=2):
            self.status_filter = gui.add_dropdown('Decision filter', ('all', 'keep', 'reject', 'review', 'pending'), initial_value='all')
            self.reason_filter = gui.add_dropdown('Reason filter', ('all', 'inversion', 'ground_roll', 'invalid_motion'), initial_value='all')
            self.search = gui.add_text('Search path', initial_value='')
            self.page_info = gui.add_text('Page', initial_value='', disabled=True)
            previous_page = gui.add_button('Previous page')
            next_page = gui.add_button('Next page')
            self.motion = gui.add_dropdown('Motion', tuple(project.names[:self.PAGE_SIZE]), initial_value=project.names[0])
            previous = gui.add_button('Previous motion')
            following = gui.add_button('Next motion')
        with gui.add_folder('Playback and evidence', order=3):
            self.play = gui.add_button('Play / Pause')
            self.frame = gui.add_slider('Frame', min=0, max=1, step=1, initial_value=0)
            self.speed = gui.add_slider('Speed', min=.1, max=3., step=.1, initial_value=1.)
            self.event = gui.add_dropdown('Event', ('Full clip',), initial_value='Full clip')
            self.details = gui.add_markdown('')
        with gui.add_folder('Manual review and export', order=4):
            self.note = gui.add_text('Review note', initial_value='')
            self.keep = gui.add_button('Manually keep')
            self.reject = gui.add_button('Manually reject')
            self.clear = gui.add_button('Clear manual override')
            self.export_button = gui.add_button('Export manifest + whitelist', disabled=True, color='green')
            self.export_path = gui.add_text('Export directory', initial_value='', disabled=True)
            gui.add_markdown('Only final **keep** motions are whitelisted. Each export gets its own directory. '
                             'Rules and manual decisions are saved automatically.')

        self._click(self.batch, project.start_batch)
        self._click(self.cancel, project.cancel)
        self._click(self.preview, lambda: project.preview(self.current))
        self._click(self.play, self.toggle_play)
        self._click(previous, lambda: self.step_motion(-1))
        self._click(following, lambda: self.step_motion(1))
        self._click(previous_page, lambda: self.change_page(-1))
        self._click(next_page, lambda: self.change_page(1))
        self._click(self.keep, lambda: self.override('keep'))
        self._click(self.reject, lambda: self.override('reject'))
        self._click(self.clear, lambda: self.override(None))
        self._click(self.export_button, self.start_export)
        for control in self.rules.values():
            self._update(control, self.change_policy)
        for control in (self.status_filter, self.reason_filter, self.search):
            self._update(control, self.filter_changed)
        self._update(self.motion, lambda: self.select_motion(self.motion.value))
        self._update(self.frame, self.seek)
        self._update(self.event, self.select_event)
        self.refresh_list()
        self.poll()

    def _click(self, control, callback):
        control.on_click(lambda _: self.queue.put(callback))

    def _update(self, control, callback):
        # Viser emits client=None for server-side value changes.
        control.on_update(lambda event: self.queue.put(callback) if event.client is not None else None)

    def change_policy(self):
        try:
            policy = SelectionPolicy(**{key: control.value for key, control in self.rules.items()})
        except ValueError:
            for key, value in asdict(self.project.policy).items():
                self.rules[key].value = value
            raise
        self.project.set_policy(policy)
        self.ground.position = np.asarray(self.scene._scene_offset) + (0., 0., policy.ground_z)
        self.refresh_list()
        self.refresh_details()

    def matching_names(self):
        query = self.search.value.lower()
        result = []
        for name in self.project.names:
            if query not in name.lower():
                continue
            row = self.project.row(name)
            if self.status_filter.value != 'all' and row['status'] != self.status_filter.value:
                continue
            if self.reason_filter.value != 'all' and self.reason_filter.value not in row.get('reasons', []):
                continue
            result.append(name)
        return result

    def filter_changed(self):
        self.page = 0
        self.refresh_list()

    def change_page(self, offset):
        self.page = max(0, self.page + offset)
        self.refresh_list()

    def refresh_list(self):
        matches = self.matching_names()
        self.page = min(self.page, max(0, (len(matches) - 1) // self.PAGE_SIZE))
        options = matches[self.page * self.PAGE_SIZE:(self.page + 1) * self.PAGE_SIZE]
        self.page_info.value = f'{len(matches)} matches; page {self.page + 1}/{max(1, (len(matches) + self.PAGE_SIZE - 1) // self.PAGE_SIZE)}'
        self.motion.options = tuple(options) or ('No matching motions',)
        self.motion.disabled = not options
        if self.current not in options:
            if options:
                self.motion.value = options[0]
                self.select_motion(options[0])
            else:
                self.current, self.qpos, self.playing = None, None, False
                self.details.content = 'No matching motions. Playback paused; the scene may show the previous pose.'
                self.event.options = ('Full clip',)
                self.event.value = 'Full clip'
        elif self.motion.value != self.current:
            self.motion.value = self.current

    def step_motion(self, offset):
        options = list(self.motion.options)
        if self.current in options:
            name = options[(options.index(self.current) + offset) % len(options)]
            self.motion.value = name
            self.select_motion(name)

    def select_motion(self, name):
        if name not in self.project.paths:
            return
        self.current, self.playing, self.qpos = name, False, None
        try:
            qpos = load_motion(self.project.paths[name])
            if not len(qpos) or qpos.shape[1] != self.project.model.nq:
                raise ValueError('Empty motion or qpos width mismatch')
            if not np.isfinite(qpos).all():
                raise ValueError('Motion contains nonfinite values')
            self.qpos = qpos
            self.bounds = (0, len(qpos) - 1)
            self.frame.max = max(1, len(qpos) - 1)
            self.show_frame(0)
        except Exception as error:
            self.message.value = f'Cannot play {name}: {error}'
        row = self.project.row(name)
        self.note.value = (row.get('override') or {}).get('note', '')
        self.refresh_details()

    def refresh_details(self):
        if self.current is None:
            return
        row = self.project.row(self.current)
        self.details.content = (f"**{self.current}**\n\nAutomatic: **{row.get('auto_status', 'pending')}** · "
                                f"Final: **{row['status']}**{' · manual override' if row['override'] else ''}\n\n"
                                f"{'Results pending under current rules.' if row['stale'] else ''}\n\n"
                                '```json\n' + json.dumps({k: row[k] for k in ('metrics', 'reasons', 'error') if k in row}, indent=2) + '\n```')
        events = [] if row['stale'] else row.get('events', [])
        self.events = {f"{i + 1}: {e['type']} [{e['start_frame']}, {e['end_frame_exclusive']}) {e['severity']}": e
                       for i, e in enumerate(events)}
        options = ('Full clip', *self.events)
        if tuple(self.event.options) != options:
            self.event.options = options
            self.event.value = 'Full clip'
            if self.qpos is not None:
                self.bounds = (0, len(self.qpos) - 1)

    def select_event(self):
        if self.qpos is None:
            return
        event = self.events.get(self.event.value)
        padding = round(.4 / self.project.manifest.timestep)
        self.bounds = ((max(0, event['start_frame'] - padding),
                        min(len(self.qpos) - 1, event['end_frame_exclusive'] - 1 + padding))
                       if event else (0, len(self.qpos) - 1))
        self.show_frame(self.bounds[0])
        self.playing, self.next_time = True, time.monotonic()

    def show_frame(self, frame):
        if self.qpos is None:
            return
        self.cursor = max(0, min(int(frame), len(self.qpos) - 1))
        self.data.qpos[:] = self.qpos[self.cursor]
        mujoco.mj_kinematics(self.project.model, self.data)
        self.scene.update_from_mjdata(self.data)
        self.ground.position = np.asarray(self.scene._scene_offset) + (0., 0., self.project.policy.ground_z)
        self.frame.value = self.cursor

    def seek(self):
        self.playing = False
        self.show_frame(self.frame.value)

    def toggle_play(self):
        self.playing = not self.playing and self.qpos is not None
        self.next_time = time.monotonic()

    def override(self, status):
        if self.current is not None:
            self.project.set_override(self.current, status, self.note.value)
            self.refresh_list()
            self.refresh_details()

    def start_export(self):
        if self.export_thread is not None and self.export_thread.is_alive():
            return
        self.message.value = 'Verifying source content and exporting…'
        def export():
            try:
                self.export_message = (str(self.project.export()), None)
            except Exception as error:
                self.export_message = (None, str(error))
        self.export_thread = Thread(target=export, daemon=True)
        self.export_thread.start()

    def wait_export(self):
        if self.export_thread is not None:
            self.export_thread.join()

    def poll(self):
        state = self.project.snapshot()
        exporting = self.export_thread is not None and self.export_thread.is_alive()
        busy = state['running'] or exporting
        self.batch.disabled = busy
        self.cancel.disabled = not state['running']
        self.preview.disabled = busy or self.current is None
        self.export_button.disabled = busy or not state['can_export']
        self.progress.value = state['done'] / max(1, state['total'])
        self.summary.content = ' · '.join(f"**{status}: {count}** ({state['durations'][status] / 60:.1f} min)" for status, count in state['counts'].items())
        completed = state['completed_jobs'] != self.completed_jobs
        if state['running'] or self.previous_running or completed:
            self.message.value = state['message']
        if completed:
            self.completed_jobs = state['completed_jobs']
            self.refresh_list()
            self.refresh_details()
        self.previous_running = state['running']
        row = self.project.row(self.current) if self.current else None
        self.keep.disabled = busy or not row or row['stale'] or bool(row.get('error'))
        self.reject.disabled = busy or not row or row['stale']
        self.clear.disabled = busy or not row or not row['override']
        if self.export_message is not None:
            path, error = self.export_message
            self.export_message = None
            if path:
                self.export_path.value = path
                self.message.value = f'Exported to {path}'
            else:
                self.message.value = f'Export failed: {error}'

    def tick(self, *, force=False):
        handled = False
        while True:
            try:
                callback = self.queue.get_nowait()
            except Empty:
                break
            handled = True
            try:
                callback()
            except Exception as error:
                self.message.value = str(error)
        now = time.monotonic()
        if handled or force or now - self.last_poll > .5:
            # A tiny dataset may finish between callbacks and the first poll.
            if not self.project.snapshot()['running'] and (handled or force):
                self.refresh_details()
            self.poll()
            self.last_poll = now
        if self.playing and self.qpos is not None and now >= self.next_time:
            low, high = self.bounds
            frame = self.cursor + 1 if low <= self.cursor < high else low
            self.show_frame(frame)
            self.next_time = now + self.project.manifest.timestep / self.speed.value

    def close(self):
        self.project.close()
        self.wait_export()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-path', required=True, help='Local or hf:// dataset root')
    parser.add_argument('--project-path', required=True, help='Directory for rules, cache, review state and exports')
    parser.add_argument('--torso-body', default=None, help='Body with local +Z up; defaults to torso_link, then free-root body')
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8081)
    args = parser.parse_args()
    import viser
    from mjviser import ViserMujocoScene
    root = resolve_input_paths(Path.cwd(), args.input_path)[0]
    project = SelectionProject(root, args.project_path, torso_body=args.torso_body)
    server = viser.ViserServer(host=args.host, port=args.port, label='any4hdmi-select')
    scene = ViserMujocoScene(server, project.model, num_envs=1)
    scene.camera_tracking_enabled = True
    scene.create_scene_gui()
    viewer = SelectionViewer(server, project, scene)
    print(f'[any4hdmi-select] http://{args.host}:{server.get_port()}', flush=True)
    try:
        while True:
            viewer.tick()
            time.sleep(.01)
    except KeyboardInterrupt:
        pass
    finally:
        viewer.close()
        server.stop()


if __name__ == '__main__':
    main()
