from __future__ import annotations

import unittest

import mujoco
import numpy as np

from any4hdmi.scripts.viewer_issues import MotionIssueScan, issue_intervals, next_issue_frame, timeline_image


class ViewerIssuesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = mujoco.MjModel.from_xml_string('''
          <mujoco><worldbody><geom type="plane" size="5 5 .1"/>
            <body name="pelvis" pos="0 0 1"><freejoint/><geom size=".03"/>
              <body name="left_ankle_roll_link" pos="0 .08 -.9"><joint/><geom size=".1"/></body>
              <body name="right_ankle_roll_link" pos="0 -.08 -.9"><joint/><geom size=".1"/></body>
            </body>
          </worldbody></mujoco>
        ''')
        self.qpos = np.tile(self.model.qpos0, (5, 1))
        self.qpos[:, 2] = [1.1, .98, .98, 1.1, .98]

    def test_incremental_scan_checks_skipped_frames_and_keeps_input_unchanged(self) -> None:
        before = self.qpos.copy()
        scan = MotionIssueScan(self.model, self.qpos, 0, feet_only=True)
        scan.advance(2)
        self.assertEqual(scan.completed, 2)
        self.assertFalse(scan.done)
        scan.advance(10)
        self.assertTrue(scan.done)
        np.testing.assert_allclose(scan.depth_m, [0, .02, .02, 0, .02], atol=1e-7)
        self.assertEqual(issue_intervals(scan.mask('Ground penetration')), [(1, 2), (4, 4)])
        self.assertEqual(issue_intervals(scan.mask('Self collision')), [(0, 4)])
        self.assertIn('left_ankle_roll_link', scan.details(1))
        self.assertIn('20.0 mm', scan.details(1))
        np.testing.assert_array_equal(self.qpos, before)

    def test_interval_boundaries_wrap_and_no_issues(self) -> None:
        self.assertEqual(issue_intervals(np.array([1, 1, 0, 1, 0, 1], dtype=bool)), [(0, 1), (3, 3), (5, 5)])
        self.assertEqual(issue_intervals(np.zeros(1, dtype=bool)), [])
        self.assertEqual(next_issue_frame([(0, 1), (3, 3), (5, 5)], 0, 1), 3)
        self.assertEqual(next_issue_frame([(0, 1), (3, 3), (5, 5)], 5, 1), 0)
        self.assertEqual(next_issue_frame([(0, 1), (3, 3), (5, 5)], 0, -1), 5)
        self.assertIsNone(next_issue_frame([], 0, 1))

    def test_clean_single_frame_and_timeline(self) -> None:
        xml = '<mujoco><worldbody><geom type="plane" size="5 5 .1"/><body name="foot" pos="0 0 1"><freejoint/><geom size=".1"/></body></worldbody></mujoco>'
        model = mujoco.MjModel.from_xml_string(xml)
        scan = MotionIssueScan(model, model.qpos0[None], 0, feet_only=False)
        scan.advance()
        self.assertEqual(issue_intervals(scan.mask('Either')), [])
        self.assertEqual(timeline_image(scan, 0).shape, (24, 400, 3))
        self.assertTrue(np.isfinite(timeline_image(scan, 0)).all())


if __name__ == '__main__':
    unittest.main()
