from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

import mujoco
import numpy as np

from any4hdmi.core.format import save_motion, write_manifest
from any4hdmi.scripts.viewer_comparison import ComparisonView, load_comparison_motion
from any4hdmi.utils.mjcf import qpos_names_from_model


class ViewerComparisonTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name="robot"><freejoint name="floating_base_joint"/><geom size=".1"/></body></worldbody></mujoco>')
        self.original = np.tile(self.model.qpos0, (4, 1)).astype(np.float32)
        self.original[:, 2] = [.1, .2, .3, .4]
        self.current = self.original.copy()
        self.current[:, 2] += .05
        for name, poses in [("original", self.original), ("corrected", self.current)]:
            dataset = self.root / name
            save_motion(dataset / 'clips/group/a.npz', poses)
            (dataset / 'robot.xml').write_text('<mujoco/>')
            write_manifest(dataset, dataset_name=name, mjcf=dataset / 'robot.xml', timestep=.02,
                           qpos_names=qpos_names_from_model(self.model), num_motions=1,
                           source={"dataset": str(self.root / 'original'), "foot_grounding": {}}, total_hours=0)
            payload = json.loads((dataset / 'manifest.json').read_text())
            payload['motions_subdir'] = 'clips'
            (dataset / 'manifest.json').write_text(json.dumps(payload))
        self.path = self.root / 'corrected/clips/group/a.npz'

    def test_pair_uses_provenance_and_dataset_relative_paths(self) -> None:
        for reference in (None, self.root / 'original', self.root / 'original/clips/group/a.npz'):
            poses, message = load_comparison_motion(self.path, self.current, reference)
            np.testing.assert_array_equal(poses, self.original)
            self.assertIn('a.npz', message)

    def test_directory_symlink_preserves_pairing(self) -> None:
        alias = self.root / 'alias'
        alias.symlink_to(self.root / 'corrected', target_is_directory=True)
        poses, message = load_comparison_motion(alias / 'clips/group/a.npz', self.current)
        self.assertIsNotNone(poses, message)
        np.testing.assert_array_equal(poses, self.original)

    def test_missing_mismatched_frames_timestep_and_schema_hide_reference(self) -> None:
        poses, message = load_comparison_motion(self.path, self.current, self.root / 'missing')
        self.assertIsNone(poses)
        self.assertTrue(message)
        poses, _ = load_comparison_motion(self.path, self.current[:-1])
        self.assertIsNone(poses)
        path = self.root / 'original/manifest.json'
        payload = json.loads(path.read_text())
        for key, value in [('timestep', .01), ('qpos_names', ['wrong'] * 7)]:
            invalid = dict(payload, **{key: value})
            path.write_text(json.dumps(invalid))
            poses, message = load_comparison_motion(self.path, self.current)
            self.assertIsNone(poses)
            self.assertIn(key, message)
        path.write_text(json.dumps(payload))

    def test_same_frame_common_offset_preserves_vertical_difference_and_hides_missing_pair(self) -> None:
        server = mock.MagicMock()
        for method in ('add_checkbox', 'add_text'):
            getattr(server.gui, method).side_effect = lambda *args, **kwargs: mock.Mock(value=kwargs.get('initial_value'))
        server.scene.add_batched_glb.side_effect = lambda *args, **kwargs: mock.Mock(**kwargs)
        data = mujoco.MjData(self.model)
        data.qpos[:] = self.current[2]
        mujoco.mj_forward(self.model, data)
        scene = SimpleNamespace(_mesh_groups=[SimpleNamespace(body_ids=np.array([1]), group_id=0, handle=mock.Mock(glb_data=b'glb'))],
                                _fixed_geom_handles={}, _scene_offset=np.array([0., 0, -.35]), geom_groups_visible=[True, False, True, False, False, False])
        contacts = mock.Mock()
        with mock.patch('any4hdmi.scripts.viewer_comparison.ContactHighlights'):
            comparison = ComparisonView(server, self.model, scene, contacts)
            comparison.load(self.path, self.current)
            comparison.update(2)
            self.assertAlmostEqual(comparison.data.qpos[2], .3, places=6)
            position = comparison._mesh_groups[0].handle.batched_positions[0]
            self.assertAlmostEqual(position[2], -.05, places=6)  # Original Z .3 + common offset -.35.
            self.assertGreater(abs(position[1]), 1)
            comparison.update(0)
            self.assertAlmostEqual(comparison.data.qpos[2], .1, places=6)
            comparison.enabled.value = False
            comparison.update(0)
            self.assertFalse(comparison._mesh_groups[0].handle.visible)
            comparison.enabled.value = True
            comparison.load(self.root / 'corrected/clips/group/missing.npz', self.current)
            comparison.update(0)
            self.assertIsNone(comparison.qpos)
            self.assertFalse(comparison._mesh_groups[0].handle.visible)


if __name__ == '__main__':
    unittest.main()
