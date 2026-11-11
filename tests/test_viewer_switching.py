from __future__ import annotations

import threading
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

import mujoco
import numpy as np

from any4hdmi.core.format import save_motion, write_manifest
from any4hdmi.scripts.viewer.app import _run_mjviser
from any4hdmi.scripts.viewer.comparison import ComparisonView
from any4hdmi.utils.mjcf import qpos_names_from_model


class ViewerSwitchingTest(unittest.TestCase):
    def test_shorter_clip_switch_is_atomic_with_playback(self) -> None:
        xml = '<mujoco><worldbody><body><freejoint name="floating_base_joint"/><geom size=".1"/></body></worldbody></mujoco>'
        model = mujoco.MjModel.from_xml_string(xml)
        with TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'model.xml').write_text(xml)
            for name in ('original', 'corrected'):
                for clip, frames in [('long', 5), ('short', 2)]:
                    save_motion(root / name / f'motions/{clip}.npz', np.tile(model.qpos0, (frames, 1)))
                write_manifest(root / name, dataset_name=name, mjcf=root / 'model.xml', timestep=.02,
                               qpos_names=qpos_names_from_model(model), num_motions=2, total_hours=0,
                               source={'dataset': str(root / 'original'), 'foot_grounding': {}})
            controls = {}
            server = mock.MagicMock()

            def control(label, *args, **kwargs):
                handle = mock.Mock(value=kwargs.get('initial_value'))
                handle.on_click.side_effect = lambda callback: setattr(handle, 'click', callback) or callback
                handle.on_update.side_effect = lambda callback: setattr(handle, 'update', callback) or callback
                controls[label] = handle
                return handle

            for method in ('add_button', 'add_text', 'add_checkbox', 'add_dropdown', 'add_slider', 'add_number', 'add_command'):
                getattr(server.gui, method).side_effect = control
            scene = SimpleNamespace(_mesh_groups=[], _fixed_geom_handles={}, _scene_offset=np.zeros(3),
                                    geom_groups_visible=[True, False, True, False, False, False], create_scene_gui=mock.Mock(), update_from_mjdata=mock.Mock())
            swapped, release = threading.Event(), threading.Event()
            load = ComparisonView.load

            def blocked_load(comparison, path, qpos):
                load(comparison, path, qpos)
                if path.name == 'short.npz':
                    swapped.set()
                    release.wait(2)

            stage = 0
            threads = []

            def interact(_):
                nonlocal stage
                if stage == 0:
                    controls['Frame'].value = 4
                    controls['Frame'].update(mock.Mock(client_id=0, target=controls['Frame']))
                elif stage == 1:
                    self.assertEqual(controls['Source frame (0-based)'].value, 4)
                    worker = threading.Thread(target=controls['Next'].click, args=(None,))
                    threads.append(worker)
                    worker.start()
                    self.assertTrue(swapped.wait(2))
                    # Let the next playback tick overlap the blocked callback.
                    timer = threading.Timer(.05, release.set)
                    threads.append(timer)
                    timer.start()
                else:
                    self.assertLess(controls['Source frame (0-based)'].value, 2)
                    raise StopIteration
                stage += 1

            try:
                with mock.patch('viser.ViserServer', return_value=server), \
                     mock.patch('any4hdmi.scripts.viewer.app._create_motion_scene', return_value=scene), \
                     mock.patch('any4hdmi.scripts.viewer.app.ContactHighlights'), \
                     mock.patch('any4hdmi.scripts.viewer.comparison.ContactHighlights'), \
                     mock.patch('any4hdmi.scripts.viewer.app.IssueNavigator'), \
                     mock.patch.object(ComparisonView, 'load', blocked_load), \
                     mock.patch('any4hdmi.scripts.viewer.app.time.sleep', side_effect=interact):
                    with self.assertRaises(StopIteration):
                        _run_mjviser(model, mujoco.MjData(model), [(name, root / 'corrected/motions' / name) for name in ('long.npz', 'short.npz')],
                                     'long.npz', start=0, end=-1, stride=1, frame_dt=1000, loop=False, port=8080)
            finally:
                release.set()
                for thread in threads:
                    thread.join(2)


if __name__ == '__main__':
    unittest.main()
