from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import mujoco
import numpy as np

from any4hdmi.core.format import load_manifest, load_motion, save_motion, write_manifest
from any4hdmi.core.grounding import correct_foot_penetration, measure_foot_heights
from any4hdmi.scripts.ground import correct_dataset
from any4hdmi.utils.mjcf import qpos_names_from_model


MODEL_XML = '''
<mujoco><worldbody>
  <geom name="floor" type="plane" size="5 5 .1"/>
  <body name="extra"><joint name="extra_joint"/><geom size=".01"/></body>
  <body name="pelvis"><freejoint name="floating_base_joint"/>
    <geom size=".1"/>
    <body name="left_ankle_roll_link" pos="0 .15 -.8">
      <joint name="ankle" axis="0 1 0"/>
      <geom type="box" size=".12 .05 .04"/>
      <body name="toe" pos=".1 0 -.03">
        <geom type="sphere" size=".03" contype="0" conaffinity="0"/>
      </body>
    </body>
    <body name="right_ankle_roll_link" pos="0 -.15 -.8">
      <geom type="box" size=".12 .05 .04"/>
    </body>
  </body>
</worldbody></mujoco>
'''


class GroundingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model = mujoco.MjModel.from_xml_string(MODEL_XML)
        # An unrelated joint precedes the free joint: root Z is not column 2.
        self.z_index = 3

    def motion(self, heights) -> np.ndarray:
        qpos = np.tile(self.model.qpos0, (len(heights), 1)).astype(np.float32)
        qpos[:, self.z_index] = heights
        return qpos

    def test_measures_rotated_foot_surfaces_and_visual_descendants(self) -> None:
        qpos = self.motion([.86, .86])
        qpos[1, -1] = np.pi / 2
        heights = measure_foot_heights(self.model, qpos)
        np.testing.assert_allclose(heights[0], [0, .02], atol=1e-7)
        np.testing.assert_allclose(heights[1], [-.07, .02], atol=1e-7)

    def test_geometry_heights_match_mujoco_plane_distance(self) -> None:
        mesh = '<asset><mesh name="m" scale=".1 .2 .3" vertex="-1 -1 -1 1 -1 -1 -1 1 -1 -1 -1 1 1 1 1"/></asset>'
        for geom in [
            'type="sphere" size=".1"', 'type="box" size=".1 .2 .3"',
            'type="capsule" size=".1 .2"', 'type="cylinder" size=".1 .2"',
            'type="ellipsoid" size=".1 .2 .3"', 'type="mesh" mesh="m"',
        ]:
            with self.subTest(geom=geom):
                model = mujoco.MjModel.from_xml_string(
                    f'<mujoco>{mesh}<worldbody><geom type="plane" size="5 5 .1"/>'
                    '<body name="foot"><freejoint/><geom pos=".1 .2 .03" '
                    f'euler="20 30 40" {geom}/></body></worldbody></mujoco>'
                )
                qpos = np.tile(model.qpos0, (3, 1))
                qpos[:, 2] = [.7, 0, -.2]
                qpos[2, 3:7] = [np.cos(.3), np.sin(.3), 0, 0]
                actual = measure_foot_heights(model, qpos, foot_bodies=("foot",))[:, 0]
                data = mujoco.MjData(model)
                for frame in range(3):
                    data.qpos[:] = qpos[frame]
                    mujoco.mj_forward(model, data)
                    expected = mujoco.mj_geomDistance(model, data, 0, 1, 10, None)
                    self.assertAlmostEqual(actual[frame], expected, places=6)

    def test_correction_clears_spike_without_changing_pose_or_source(self) -> None:
        qpos = self.motion([.9, .9, .80, .9, .9, .9, 1.2, 1.3])
        original = qpos.copy()
        out, report = correct_foot_penetration(self.model, qpos, timestep=.1, max_correction_speed=.2)
        np.testing.assert_array_equal(qpos, original)
        np.testing.assert_array_equal(np.delete(out, self.z_index, axis=1), np.delete(qpos, self.z_index, axis=1))
        self.assertGreaterEqual(measure_foot_heights(self.model, out).min(), .001 - 1e-6)
        correction = out[:, self.z_index] - qpos[:, self.z_index]
        self.assertLessEqual(np.abs(np.diff(correction)).max(), .02 + 1e-6)
        self.assertGreater(correction[1], 0)  # Offline anticipation, no lag at the spike.
        np.testing.assert_array_equal(out[-2:], qpos[-2:])  # Airborne frames stay airborne.
        self.assertEqual(report["before"]["penetrating_frames"], 1)
        self.assertEqual(report["after"]["penetrating_frames"], 0)
        self.assertEqual(out.dtype, np.float32)
        repeated, _ = correct_foot_penetration(self.model, out, timestep=.1)
        np.testing.assert_array_equal(repeated, out)

    def test_clean_single_frame_and_nonzero_ground(self) -> None:
        clean = self.motion([1.1])
        out, _ = correct_foot_penetration(self.model, clean, timestep=.02)
        np.testing.assert_array_equal(out, clean)
        out, report = correct_foot_penetration(self.model, self.motion([.8]), timestep=.02, ground_z=.2)
        self.assertGreaterEqual(measure_foot_heights(self.model, out).min(), .201 - 1e-6)
        self.assertEqual(report["max_correction_speed_m_s"], 0)

    def test_nonfoot_surface_controls_lift_and_report_even_when_feet_are_clear(self) -> None:
        xml = MODEL_XML.replace(
            '<geom size=".1"/>',
            '<geom size=".1"/><body name="hand" pos=".3 0 -1"><geom size=".1" contype="0" conaffinity="0"/></body>',
        )
        model = mujoco.MjModel.from_xml_string(xml)
        qpos = np.tile(model.qpos0, (3, 1)).astype(np.float32)
        qpos[:, self.z_index] = [.9, 1.2, .9]
        original = qpos.copy()
        self.assertGreater(measure_foot_heights(model, qpos).min(), 0)
        out, report = correct_foot_penetration(model, qpos, timestep=.1)
        data = mujoco.MjData(model)
        hand_geom = int(model.body_geomadr[model.body('hand').id])
        for pose in out:
            data.qpos[:] = pose
            mujoco.mj_forward(model, data)
            distance = mujoco.mj_geomDistance(model, data, 0, hand_geom, 10, None)
            self.assertGreaterEqual(distance, .001 - 1e-6)
        np.testing.assert_array_equal(qpos, original)
        np.testing.assert_array_equal(np.delete(out, self.z_index, axis=1), np.delete(qpos, self.z_index, axis=1))
        self.assertEqual(report['before']['penetrating_frames'], 2)
        self.assertAlmostEqual(report['before']['max_penetration_m'], .2, places=6)
        self.assertEqual(report['after']['penetrating_frames'], 0)
        self.assertIn('hand', report['body_names'])

    def test_inverted_pose_is_grounded_using_torso_and_ignores_scene_geometry(self) -> None:
        model = self.model
        qpos = self.motion([.05])
        qpos[0, 4:8] = [0, 1, 0, 0]  # Upside down, feet high above torso.
        original = qpos.copy()
        self.assertGreater(measure_foot_heights(model, qpos).min(), .5)
        out, report = correct_foot_penetration(model, qpos, timestep=.02)
        self.assertAlmostEqual(float(out[0, self.z_index]), .101, places=6)
        self.assertAlmostEqual(report['before']['max_penetration_m'], .05, places=6)
        self.assertEqual(report['after']['penetrating_frames'], 0)
        np.testing.assert_array_equal(np.delete(out, self.z_index, axis=1), np.delete(original, self.z_index, axis=1))
        # The unrelated 'extra' body still intersects the floor; never lift the
        # robot in an attempt to repair world/other-body geometry.
        self.assertNotIn('extra', report['body_names'])

    def test_invalid_inputs_fail_explicitly(self) -> None:
        for kwargs in [{"timestep": 0}, {"timestep": .02, "clearance": -.01},
                       {"timestep": .02, "max_correction_speed": 0}, {"timestep": float("nan")}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                correct_foot_penetration(self.model, self.motion([.8]), **kwargs)
        for qpos in [self.motion([]), np.zeros((1, 2)), self.motion([float("nan")])]:
            with self.assertRaises(ValueError):
                correct_foot_penetration(self.model, qpos, timestep=.02)
        with self.assertRaisesRegex(ValueError, "missing"):
            measure_foot_heights(self.model, self.motion([.8]), foot_bodies=("missing",))

    def test_rejects_feet_outside_robot_and_multiple_floating_bases(self) -> None:
        model = mujoco.MjModel.from_xml_string('''
            <mujoco><worldbody>
              <body name="fixed_foot"><geom size=".1"/></body>
              <body><freejoint/><geom size=".1"/></body>
            </worldbody></mujoco>
        ''')
        with self.assertRaisesRegex(ValueError, "belong"):
            correct_foot_penetration(model, model.qpos0[None], timestep=.02, foot_bodies=("fixed_foot",))
        model = mujoco.MjModel.from_xml_string('''
            <mujoco><worldbody>
              <body><freejoint/><geom size=".1"/></body>
              <body><freejoint/><geom size=".1"/></body>
            </worldbody></mujoco>
        ''')
        with self.assertRaisesRegex(ValueError, "exactly one"):
            correct_foot_penetration(model, model.qpos0[None], timestep=.02)

    def test_dataset_round_trip_selection_report_and_no_overwrite(self) -> None:
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            path = save_motion(source / "motions/group/clip.npz", self.motion([.8, .9]))
            save_motion(source / "motions/other.npz", self.motion([1.1]))
            (source / "model.xml").write_text(MODEL_XML)
            write_manifest(
                source, dataset_name="example", mjcf=source / "model.xml", timestep=.02,
                qpos_names=qpos_names_from_model(self.model), num_motions=2,
                source={"original": True}, total_hours=3 * .02 / 3600,
            )
            original = path.read_bytes()
            for selection in [source, path.parent, path]:
                output = root / f"out_{selection.name}"
                report_path = correct_dataset(selection, output)
                report = json.loads(report_path.read_text())
                manifest = load_manifest(output)
                count = 2 if selection == source else 1
                self.assertEqual(manifest.payload["num_motions"], count)
                self.assertEqual(manifest.timestep, .02)
                self.assertEqual(manifest.mjcf_path, source / "model.xml")
                self.assertEqual(len(report["motions"]), count)
                self.assertEqual(report["summary"]["after"]["penetrating_frames"], 0)
                self.assertEqual(report["settings"]["scope"], "whole_robot")
                self.assertIn("pelvis", report["summary"]["body_names"])
                self.assertNotIn("extra", report["summary"]["body_names"])
                self.assertGreaterEqual(min(report["summary"]["after"]["min_body_z_m"]), .001 - 1e-6)
                corrected = load_motion(output / "motions/group/clip.npz")
                self.assertGreaterEqual(measure_foot_heights(self.model, corrected).min(), .001 - 1e-6)
                with self.assertRaises(FileExistsError):
                    correct_dataset(selection, output)
            self.assertEqual(path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
