from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

import mujoco
import numpy as np

from any4hdmi.core.format import save_motion, write_manifest
from any4hdmi.scripts.viewer.app import _run_mjviser
from any4hdmi.utils.mjcf import qpos_names_from_model


class ViewerWorkflowTest(unittest.TestCase):
    def test_support_uses_source_time_and_separate_comparison_clip(self) -> None:
        xml = '''<mujoco><worldbody><geom name="floor" type="plane" size="5 5 .1"/>
          <body name="left_ankle_roll_link" pos="0 0 .028">
            <freejoint name="floating_base_joint"/><geom type="box" size=".1 .04 .02"/>
          </body></worldbody></mujoco>'''
        model = mujoco.MjModel.from_xml_string(xml)
        with TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'robot.xml').write_text(xml)
            current = np.tile(model.qpos0, (30, 1))
            current[15:, 0] += np.arange(15) * .01
            original = np.tile(model.qpos0, (30, 1))
            original[:, 0] += np.arange(30) * .01
            for name, poses in [('original', original), ('corrected', current)]:
                save_motion(root / name / 'motions/a.npz', poses)
                write_manifest(root / name, dataset_name=name, mjcf=root / 'robot.xml', timestep=.02,
                               qpos_names=qpos_names_from_model(model), num_motions=1, total_hours=0,
                               source={'dataset': str(root / 'original'), 'foot_grounding': {}})
            controls, meshes = {}, {}
            server = mock.MagicMock()

            def control(label, *args, **kwargs):
                handle = mock.Mock(value=kwargs.get('initial_value'))
                for action in ('on_click', 'on_update', 'on_trigger'):
                    callbacks = []
                    getattr(handle, action).side_effect = lambda callback, callbacks=callbacks: callbacks.append(callback) or callback
                    setattr(handle, action + '_callback', lambda event, callbacks=callbacks: [callback(event) for callback in callbacks])
                controls[label] = handle
                return handle

            def mesh(name, *args, **kwargs):
                meshes[name] = mock.Mock(**kwargs)
                return meshes[name]

            for method in ('add_button', 'add_text', 'add_checkbox', 'add_dropdown', 'add_slider', 'add_number', 'add_command'):
                getattr(server.gui, method).side_effect = control
            server.scene.add_mesh_simple.side_effect = mesh
            scene = SimpleNamespace(_mesh_groups=[], _fixed_geom_handles={}, _scene_offset=np.array([0., 0., -.7]),
                                    geom_groups_visible=[True, False, True, False, False, False],
                                    create_scene_gui=mock.Mock(), update_from_mjdata=mock.Mock())
            stage = 0

            def interact(_):
                nonlocal stage
                if stage == 0:
                    controls['Stable foot support'].value = True
                    controls['Stable foot support'].on_update_callback(None)
                    controls['Start / Pause'].on_click_callback(None)
                elif stage == 1:
                    self.assertTrue(meshes['/contact_links/1/0'].visible)
                    self.assertFalse(any(h.visible for name, h in meshes.items() if name.startswith('/comparison/')))
                    controls['Frame'].value = 9  # Source frame 27 with stride=3.
                    controls['Frame'].on_update_callback(mock.Mock(client_id=0, target=controls['Frame']))
                elif stage == 2:
                    self.assertEqual(controls['Source frame (0-based)'].value, 27)
                    self.assertFalse(meshes['/contact_links/1/0'].visible)
                    controls['Frame'].value = 1
                    controls['Frame'].on_update_callback(mock.Mock(client_id=0, target=controls['Frame']))
                else:
                    self.assertEqual(controls['Source frame (0-based)'].value, 3)
                    self.assertTrue(meshes['/contact_links/1/0'].visible)
                    raise StopIteration
                stage += 1

            with mock.patch('viser.ViserServer', return_value=server), \
                 mock.patch('any4hdmi.scripts.viewer.app._create_motion_scene', return_value=scene), \
                 mock.patch('any4hdmi.scripts.viewer.app.time.sleep', side_effect=interact):
                with self.assertRaises(StopIteration):
                    _run_mjviser(model, mujoco.MjData(model), [('a.npz', root / 'corrected/motions/a.npz')], 'a.npz',
                                 start=0, end=-1, stride=3, frame_dt=1000, loop=False, port=8080)

    def test_issue_jumps_seek_both_poses_exactly_despite_stride_and_switch_resets_scan(self) -> None:
        xml = '''<mujoco><worldbody><geom name="floor" type="plane" size="5 5 .1"/>
          <body name="robot" pos="0 0 1"><freejoint name="floating_base_joint"/><geom size=".1"/>
            <body name="left_ankle_roll_link" pos="0 0 -.9"><geom size=".1"/></body>
          </body></worldbody></mujoco>'''
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        with TemporaryDirectory() as temp:
            root = Path(temp)
            (root / 'robot.xml').write_text(xml)
            original = np.tile(model.qpos0, (6, 1)).astype(np.float32)
            original[:, 2] = [1.1, .98, 1.1, 1.1, 1.1, .98]
            corrected = original.copy()
            corrected[:, 2] += .03
            for name, poses in [('original', original), ('corrected', corrected)]:
                save_motion(root / name / 'motions/group/a.npz', poses)
                write_manifest(root / name, dataset_name=name, mjcf=root / 'robot.xml', timestep=.02,
                               qpos_names=qpos_names_from_model(model), num_motions=1,
                               source={'dataset': str(root / 'original'), 'foot_grounding': {}}, total_hours=0)
            second = save_motion(root / 'corrected/motions/b.npz', corrected)
            current = root / 'corrected/motions/group/a.npz'
            controls = {}
            server = mock.MagicMock()

            def control(label, *args, **kwargs):
                handle = mock.Mock(value=kwargs.get('initial_value'), options=tuple(args[0]) if args else (), disabled=kwargs.get('disabled', False))
                for action in ('on_click', 'on_update', 'on_trigger'):
                    getattr(handle, action).side_effect = lambda callback, action=action: setattr(handle, action + '_callback', callback) or callback
                controls[label] = handle
                return handle

            for method in ('add_button', 'add_text', 'add_checkbox', 'add_dropdown', 'add_slider', 'add_number', 'add_command'):
                getattr(server.gui, method).side_effect = control
            server.scene.add_batched_glb.side_effect = lambda *args, **kwargs: mock.Mock(**kwargs)
            scene = SimpleNamespace(_mesh_groups=[SimpleNamespace(body_ids=np.array([1, 2]), group_id=0, handle=mock.Mock(glb_data=b'glb'))],
                                    _fixed_geom_handles={}, geom_groups_visible=[True, False, True, False, False, False], _scene_offset=np.zeros(3),
                                    create_scene_gui=mock.Mock(), update_from_mjdata=mock.Mock())
            stage = 0

            def interact(_):
                nonlocal stage
                if stage == 0:
                    controls['Start / Pause'].on_click_callback(None)
                    controls['Scan current clip'].on_click_callback(None)
                elif stage == 2:  # Both scans have finished across separate loop ticks.
                    controls['Issue source'].value = 'Original'
                    controls['Issue source'].on_update_callback(None)
                    controls['Issue type'].value = 'Ground penetration'
                    controls['Issue type'].on_update_callback(None)
                    self.assertEqual(controls['Issue interval'].options, ('1..1', '5..5'))
                    controls['Next issue'].on_click_callback(None)
                elif stage == 3:
                    self.assertEqual(controls['Source frame (0-based)'].value, 5)
                    self.assertEqual(controls['Issue interval'].value, '5..5')
                    self.assertEqual(controls['Status'].value, 'Paused')
                    np.testing.assert_array_equal(data.qpos, corrected[5])
                    # Inspect the real comparison renderer's pose messages.
                    reference_handle = server.scene.add_batched_glb.side_effect_handles[0]
                    self.assertAlmostEqual(reference_handle.batched_positions[0, 2], .98, places=6)
                    controls['Previous issue'].on_click_callback(None)
                elif stage == 4:
                    self.assertEqual(controls['Source frame (0-based)'].value, 1)
                    np.testing.assert_array_equal(data.qpos, corrected[1])
                    controls['Next'].on_click_callback(None)
                    self.assertEqual(controls['Scan status'].value, 'Not scanned')
                    self.assertTrue(controls['Next issue'].disabled)
                    self.assertTrue(controls['Show original'].disabled)
                    # Switching before the next scan completes must discard old results.
                    controls['Scan current clip'].on_click_callback(None)
                    controls['Previous'].on_click_callback(None)
                    self.assertEqual(controls['Scan status'].value, 'Not scanned')
                    self.assertTrue(controls['Next issue'].disabled)
                    raise StopIteration
                stage += 1

            reference_handles = []

            def reference_mesh(*args, **kwargs):
                handle = mock.Mock(**kwargs)
                reference_handles.append(handle)
                return handle

            server.scene.add_batched_glb.side_effect = reference_mesh
            server.scene.add_batched_glb.side_effect_handles = reference_handles
            with mock.patch('viser.ViserServer', return_value=server), \
                 mock.patch('any4hdmi.scripts.viewer.app._create_motion_scene', return_value=scene), \
                 mock.patch('any4hdmi.scripts.viewer.app.time.sleep', side_effect=interact):
                with self.assertRaises(StopIteration):
                    _run_mjviser(model, data, [('group/a.npz', current), ('b.npz', second)], 'group/a.npz',
                                 start=0, end=-1, stride=2, frame_dt=1000, loop=False, port=8080)


if __name__ == '__main__':
    unittest.main()
