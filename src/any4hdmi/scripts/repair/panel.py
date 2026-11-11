from __future__ import annotations

from pathlib import Path
from threading import Thread

from any4hdmi.core.repair import RepairSettings
from any4hdmi.core.repair.dataset import RepairSession, motion_path


OUTCOMES = {'Unchanged': 'unchanged', 'Passed': 'passed', 'Needs review': 'needs_review', 'Failed': 'failed'}


class RepairPanel:
    """Viser controls around the same disk-backed job used by the CLI."""

    def __init__(self, server, motions, current_label, show, *, synchronize=lambda fn: fn):
        self.motions = {name: motion_path(path) for name, path in motions.items()}
        self.names = {path: name for name, path in self.motions.items()}
        self.current_label, self.show = current_label, show
        self.selected = set()
        self.session = None
        self._export_thread = None
        self._export_message = None
        self._previewing = False
        self._result_key = None
        with server.gui.add_folder('Repair', order=6, expand_by_default=False):
            self.scope = server.gui.add_dropdown('Repair scope', ('Current motion', 'Selected motions', 'All loaded motions'), initial_value='Current motion')
            self.pick = server.gui.add_dropdown('Selection motion', tuple(self.motions), initial_value=next(iter(self.motions)))
            add = server.gui.add_button('Add to selection')
            remove = server.gui.add_button('Remove from selection')
            clear = server.gui.add_button('Clear selection')
            self.selection = server.gui.add_text('Selected motions', initial_value='0 selected', disabled=True)
            self.ground = server.gui.add_checkbox('Repair ground penetration', initial_value=True)
            self.collision = server.gui.add_checkbox('Repair self collisions', initial_value=True)
            self.ground_z = server.gui.add_number('Repair ground Z (m)', initial_value=0., step=.001)
            self.check_button = server.gui.add_button('Check selected')
            self.repair_button = server.gui.add_button('Repair selected')
            self.cancel_button = server.gui.add_button('Cancel repair', disabled=True)
            self.status = server.gui.add_text('Repair status', initial_value='Ready', disabled=True)
            self.filter = server.gui.add_dropdown('Result filter', ('All', *OUTCOMES), initial_value='All')
            self.result = server.gui.add_dropdown('Repair result', ('No completed results',), disabled=True)
            self.previous = server.gui.add_button('Previous result', disabled=True)
            self.next = server.gui.add_button('Next result', disabled=True)
            self.preview = server.gui.add_button('Preview repair', disabled=True)
            restore = server.gui.add_button('Restore source')
            self.details = server.gui.add_markdown('No result selected.')
            self.export_scope = server.gui.add_dropdown('Repair export scope', ('All completed results', 'Current result', 'Selected motions'), initial_value='All completed results')
            self.policy = server.gui.add_dropdown('Repair export policy', ('Preserve selection', 'Passed / unchanged only'), initial_value='Preserve selection')
            self.export_summary = server.gui.add_markdown('**Export:** 0 completed')
            self.output = server.gui.add_text('Repair output directory', initial_value='')
            self.export_button = server.gui.add_button('Export repair', disabled=True)
            server.gui.add_markdown('Repair checks full source clips, independent of playback FPS, stride and Trim. '
                                    'Preview does not change source files. Preserve selection exports originals for unresolved results.')
        self.selection_controls = (self.pick, add, remove, clear, self.selection)
        add.on_click(synchronize(lambda _: self.change_selection('add')))
        remove.on_click(synchronize(lambda _: self.change_selection('remove')))
        clear.on_click(synchronize(lambda _: self.change_selection('clear')))
        self.check_button.on_click(synchronize(lambda _: self.start(True)))
        self.repair_button.on_click(synchronize(lambda _: self.start(False)))
        self.cancel_button.on_click(synchronize(lambda _: self.session.cancel() if self.session else None))
        self.filter.on_update(synchronize(lambda _: self.poll()))
        self.result.on_update(synchronize(lambda _: self.update_details()))
        self.previous.on_click(synchronize(lambda _: self.shift_result(-1)))
        self.next.on_click(synchronize(lambda _: self.shift_result(1)))
        self.preview.on_click(synchronize(lambda _: self.preview_result()))
        restore.on_click(synchronize(lambda _: self.restore_source()))
        self.export_scope.on_update(synchronize(lambda _: self.poll()))
        self.policy.on_update(synchronize(lambda _: self.poll()))
        self.export_button.on_click(synchronize(lambda _: self.export()))
        self.scope.on_update(synchronize(lambda _: self.poll()))

    def change_selection(self, action):
        if action == 'clear':
            self.selected.clear()
        elif action == 'add':
            self.selected.add(self.pick.value)
        else:
            self.selected.discard(self.pick.value)
        names = sorted(self.selected)
        self.selection.value = f'{len(names)} selected: ' + ', '.join(names[:20]) + (' …' if len(names) > 20 else '')
        self.poll()

    def selected_paths(self):
        if self.scope.value == 'All loaded motions':
            return list(self.motions.values())
        if self.scope.value == 'Selected motions':
            return [self.motions[name] for name in self.motions if name in self.selected]
        return [self.motions[self.current_label()]]

    def start(self, check_only):
        if self.busy():
            return
        try:
            settings = RepairSettings(ground=bool(self.ground.value), self_collision=bool(self.collision.value), ground_z=float(self.ground_z.value))
            paths = self.selected_paths()
            if not paths:
                raise ValueError('Select at least one motion')
            if self._previewing:
                self.restore_source()
            if self.session:
                self.session.close()
            self.session = RepairSession(paths, settings)
            self._result_key = None
            self._export_message = None
            self.session.start(check_only=check_only)
        except (ValueError, OSError) as exc:
            self.status.value = str(exc)
            return
        self.poll()

    def busy(self):
        return bool((self.session and self.session.snapshot()['running']) or
                    (self._export_thread and self._export_thread.is_alive()))

    def poll(self):
        custom = self.scope.value == 'Selected motions' or self.export_scope.value == 'Selected motions'
        for control in self.selection_controls:
            control.visible = custom
        if not self.session:
            self.status.value = f'Ready: {len(self.selected_paths())} motions in scope'
            return
        state = self.session.snapshot()
        results = state['results']
        busy = self.busy()
        self.check_button.disabled = self.repair_button.disabled = busy
        self.cancel_button.disabled = not state['running']
        operation = 'Check' if state['check_only'] else 'Repair'
        stage = 'Cancelled' if state['cancelled'] else 'Running' if state['running'] else 'Complete'
        current = self.names.get(state['current'], '')
        message = f"{operation} {stage}: {len(results)}/{state['total']} {current}"
        if self._export_message:
            message = self._export_message
        if self.status.value != message:
            self.status.value = message
        outcome = OUTCOMES.get(self.filter.value)
        options = tuple(self.names[path] for path, report in results.items() if outcome is None or report['outcome'] == outcome)
        key = (options, self.filter.value)
        if key != self._result_key:
            old = self.result.value
            self.result.options = options or ('No completed results',)
            self.result.value = old if old in options else self.result.options[0]
            self._result_key = key
        self.result.disabled = self.previous.disabled = self.next.disabled = not options
        self.update_details()
        self.update_export(state)

    def update_details(self):
        path = self.motions.get(self.result.value)
        report = self.session.snapshot()['results'].get(path) if self.session else None
        self.preview.disabled = not report or report.get('operation') != 'repair' or report['outcome'] == 'failed'
        if not report:
            self.details.content = 'No result selected.'
            return
        parts = [report['outcome'], '; '.join(report.get('reasons', []))]
        before, after = report.get('before', {}), report.get('after', {})
        for kind in ('ground', 'self'):
            if before:
                parts.append(f"{kind}: {before[kind + '_frames']} → {after[kind + '_frames']} frames, "
                             f"{before[kind + '_depth_m'] * 1000:.1f} → {after[kind + '_depth_m'] * 1000:.1f} mm")
        if after:
            parts.append(f"joint limits: {before['joint_frames']} → {after['joint_frames']} frames")
            parts.append(f"support drift {report['support_drift_m'] * 1000:.1f} mm; root change {report['root_change_m'] * 1000:.1f} mm; "
                         f"joint change {report['joint_change_max']:.3f}; correction speed/acceleration "
                         f"{report['correction_speed_max']:.2f}/{report['correction_acceleration_max']:.2f}")
            settings = report['settings']
            parts.append(f"Job settings: ground={settings['ground']}, self={settings['self_collision']}, Z={settings['ground_z']:g}")
        text = '\n\n'.join(part for part in parts if part)
        if self.details.content != text:
            self.details.content = text

    def shift_result(self, delta):
        if self.result.disabled:
            return
        options = self.result.options
        self.result.value = options[(options.index(self.result.value) + delta) % len(options)]
        self.update_details()
        self.preview_result()

    def preview_result(self):
        if self.preview.disabled:
            return
        path = self.motions[self.result.value]
        candidate, original = self.session.candidate(path), self.session.source(path)
        if candidate is not None and original is not None:
            self.show(path, candidate, original)
            self._previewing = True

    def restore_source(self):
        self.show(self.motions[self.current_label()], None, None)
        self._previewing = False

    def export_paths(self, results):
        if self.export_scope.value == 'Current result':
            path = self.motions.get(self.result.value)
            return [path] if path in results else []
        if self.export_scope.value == 'Selected motions':
            return [self.motions[name] for name in self.motions if name in self.selected]
        return list(results)

    def update_export(self, state):
        results = state['results']
        paths = self.export_paths(results)
        unfinished = sum(path not in results for path in paths)
        passed = sum(results[path]['outcome'] in ('passed', 'unchanged') for path in paths if path in results)
        unresolved = len(paths) - unfinished - passed
        only_passed = self.policy.value == 'Passed / unchanged only'
        included = passed if only_passed else len(paths) - unfinished
        summary = (f'**Export:** {included} included · {0 if only_passed else unresolved} unresolved originals\n\n'
                   f'{unresolved if only_passed else 0} excluded · {unfinished} unfinished')
        if self.export_summary.content != summary:
            self.export_summary.content = summary
        self.export_button.disabled = state['check_only'] or self.busy() or not included or bool(unfinished)

    def export(self):
        if self.export_button.disabled:
            return
        if not self.output.value.strip():
            self.status.value = 'Enter a new output directory'
            return
        paths = self.export_paths(self.session.snapshot()['results'])
        output = Path(self.output.value)
        passed_only = self.policy.value == 'Passed / unchanged only'
        self._export_message = 'Exporting…'
        def write():
            try:
                report = self.session.export(paths, output, passed_only=passed_only)
                self._export_message = f'Saved: {report}'
            except (OSError, ValueError) as exc:
                self._export_message = f'Export failed: {exc}'
        self._export_thread = Thread(target=write, name='repair-export', daemon=True)
        self._export_thread.start()
        self.poll()

    def wait_export(self):
        if self._export_thread:
            self._export_thread.join()

    def close(self):
        self.wait_export()
        if self.session:
            self.session.close()
