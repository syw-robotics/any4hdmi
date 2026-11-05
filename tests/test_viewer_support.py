from __future__ import annotations

import unittest

import mujoco
import numpy as np

from any4hdmi.scripts.viewer_support import infer_support, stable_support_mask


FOOT_XML = '''<mujoco><worldbody>
  <geom name="floor" type="plane" size="5 5 .1"/>
  <body name="left_ankle_roll_link" pos="0 0 .02">
    <freejoint/><geom type="box" size=".1 .04 .02"/>
  </body>
</worldbody></mujoco>'''


class StableSupportTest(unittest.TestCase):
    def test_standing_above_old_five_mm_cutoff_but_not_hovering_or_penetrating(self):
        for height, expected in [(.008, True), (.08, False), (-.05, False)]:
            points = np.zeros((20, 1, 3))
            points[:, :, 2] = height
            self.assertEqual(stable_support_mask(points, .02).tolist(), [expected] * 20)

    def test_near_ground_sliding_and_fast_swing_are_not_support(self):
        points = np.zeros((20, 1, 3))
        points[:, 0, 0] = np.arange(20) * .01  # 0.5 m/s despite touching.
        self.assertFalse(stable_support_mask(points, .02).any())
        points[:, 0, 0] = np.abs(np.arange(20) - 10) * .01
        self.assertFalse(stable_support_mask(points, .02).any())

    def test_source_timestep_changes_speed_not_sample_displacement(self):
        points = np.zeros((20, 1, 3))
        points[:, 0, 0] = np.arange(20) * .01
        self.assertFalse(stable_support_mask(points, .02).any())
        self.assertTrue(stable_support_mask(points, .2).all())

    def test_brief_contact_and_single_frame_cannot_establish_support(self):
        self.assertFalse(stable_support_mask(np.zeros((1, 1, 3)), .02).any())
        points = np.zeros((20, 1, 3))
        points[:, :, 2] = .1
        points[8:10, :, 2] = 0
        self.assertFalse(stable_support_mask(points, .02).any())

    def test_height_hysteresis_keeps_support_until_clear_lift(self):
        points = np.zeros((30, 1, 3))
        points[:, 0, 2] = np.r_[np.full(6, .018), np.linspace(.018, .024, 6),
                                   np.full(6, .024), np.linspace(.024, .05, 12)]
        support = stable_support_mask(points, .02)
        self.assertTrue(support[:18].all())
        self.assertFalse(support[-5:].any())

    def test_toe_pivot_supports_with_moving_ankle_and_floor_offset(self):
        model = mujoco.MjModel.from_xml_string(FOOT_XML.replace('name="floor"', 'name="floor" pos="0 0 .4"'))
        qpos = np.tile(model.qpos0, (40, 1))
        angles = np.linspace(0, .6, 40)
        # Rotate around the fixed front edge (x=.1, z=.4).
        qpos[:, 3] = np.cos(angles / 2)
        qpos[:, 5] = np.sin(angles / 2)
        qpos[:, 0] = .1 - .1 * np.cos(angles) + .02 * np.sin(angles)
        qpos[:, 2] = .4 + .1 * np.sin(angles) + .02 * np.cos(angles)
        before = qpos.copy()
        support = infer_support(model, qpos, .02, 0)
        self.assertTrue(support[1].all())
        np.testing.assert_array_equal(qpos, before)
        qpos[:, 0] += np.arange(40) * .02
        self.assertFalse(infer_support(model, qpos, .02, 0)[1].any())

    def test_missing_feet_and_missing_floor_are_empty(self):
        model = mujoco.MjModel.from_xml_string(FOOT_XML.replace('left_ankle_roll_link', 'other'))
        self.assertEqual(infer_support(model, np.tile(model.qpos0, (10, 1)), .02, 0), {})
        self.assertEqual(infer_support(model, np.tile(model.qpos0, (10, 1)), .02, -1), {})


if __name__ == '__main__':
    unittest.main()
