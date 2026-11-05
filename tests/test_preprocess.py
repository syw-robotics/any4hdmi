from __future__ import annotations

import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np

from any4hdmi.core.format import load_manifest, load_motion
from any4hdmi.core.model import G1_JOINT_ORDER, base_qpos_adr, joint_qpos_adrs
from any4hdmi.scripts.preprocess import axellwppr, common, lafan, sonic


class PreprocessConversionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # An extra joint and reversed order catch assumptions about qpos addresses.
        bodies = "".join(
            f'<body><joint name="{name}"/><geom size="0.01"/>'
            for name in reversed(G1_JOINT_ORDER)
        ) + "</body>" * len(G1_JOINT_ORDER)
        cls.model_xml = (
            '<mujoco><worldbody><body><joint name="extra"/><geom size="0.01"/></body>'
            '<body><freejoint name="floating_base_joint"/><geom size="0.01"/>'
            f'{bodies}</body></worldbody></mujoco>'
        )
        cls.model = mujoco.MjModel.from_xml_string(cls.model_xml)
        cls.base_adr = base_qpos_adr(cls.model)
        cls.joint_adrs = joint_qpos_adrs(cls.model, G1_JOINT_ORDER)

    def expected_qpos(self, positions, quaternions, joints) -> np.ndarray:
        qpos = np.zeros((len(positions), self.model.nq), dtype=np.float32)
        qpos[:, self.base_adr : self.base_adr + 3] = positions
        qpos[:, self.base_adr + 3 : self.base_adr + 7] = quaternions
        qpos[:, self.joint_adrs] = joints
        return qpos

    def test_lafan_preserves_positions_and_reorders_xyzw_and_joints(self) -> None:
        positions = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        xyzw = np.array([[0, 0, 0, 1], [0, 0, 1, 0]], dtype=np.float32)
        joints = np.arange(58, dtype=np.float32).reshape(2, 29) / 10
        motion = np.concatenate((positions, xyzw, joints), axis=1)
        actual = lafan._build_qpos_sequence(self.model, motion)
        expected = self.expected_qpos(positions, [[1, 0, 0, 0], [0, 0, 0, 1]], joints)
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual.dtype, np.float32)

    def test_sonic_maps_named_columns_units_and_sliced_output(self) -> None:
        header = sonic._required_columns()
        motion = np.zeros((5, len(header)), dtype=np.float32)
        motion[:, 0] = np.arange(5)
        motion[:, 1:4] = np.arange(15).reshape(5, 3) * 100
        motion[:, 6] = 90
        motion[:, 7:] = np.arange(29)
        expected = self.expected_qpos(
            motion[1:4:2, 1:4] * np.float32(0.01),
            np.tile([2**-0.5, 0, 0, 2**-0.5], (2, 1)),
            np.deg2rad(motion[1:4:2, 7:]),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            csv_path = root / "input/group/clip.csv"
            csv_path.parent.mkdir(parents=True)
            # Column order is intentionally unrelated to the model or source order.
            np.savetxt(
                csv_path, motion[:, ::-1], delimiter=",",
                header=",".join(reversed(header)), comments="",
            )
            frames = sonic._convert_one(
                csv_path, csv_dir=root / "input", out_dir=root / "output",
                qpos_dim=self.model.nq, base_adr=self.base_adr,
                hinge_qpos_adrs=self.joint_adrs, translation_scale=0.01,
                angle_unit="deg", euler_order="xyz", euler_frame="extrinsic",
                start=1, end=4, stride=2,
            )
            actual = load_motion(root / "output/motions/group/clip.npz")
        self.assertEqual(frames, 2)
        np.testing.assert_allclose(actual, expected, atol=1e-7)

    def test_axellwppr_normalizes_wxyz_and_maps_joints(self) -> None:
        positions = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        joints = np.arange(58, dtype=np.float32).reshape(2, 29)
        actual = axellwppr._build_qpos_sequence(
            self.model, positions, np.array([[2, 0, 0, 0], [0, 0, 0, 3]]), joints,
            base_adr=self.base_adr, hinge_qpos_adrs=self.joint_adrs,
        )
        expected = self.expected_qpos(positions, [[1, 0, 0, 0], [0, 0, 0, 1]], joints)
        np.testing.assert_array_equal(actual, expected)

    def test_csv_commands_write_dataset_and_announce_model_before_fetch(self) -> None:
        positions = np.arange(12, dtype=np.float32).reshape(4, 3)
        joints = np.zeros((4, 29), dtype=np.float32)
        quaternions = np.tile([1, 0, 0, 0], (4, 1))
        expected = self.expected_qpos(positions[1::2], quaternions[1::2], joints[1::2])
        for module in (lafan, sonic):
            with self.subTest(converter=module.__name__), tempfile.TemporaryDirectory() as temp_dir:
                root = Path(temp_dir)
                model_path = root / "model.xml"
                model_path.write_text(self.model_xml)
                csv_path = root / "input/group/clip.csv"
                csv_path.parent.mkdir(parents=True)
                if module is lafan:
                    motion = np.concatenate((positions, quaternions[:, [1, 2, 3, 0]], joints), axis=1)
                    np.savetxt(csv_path, motion, delimiter=",")
                else:
                    motion = np.concatenate(
                        (np.arange(4)[:, None], positions * 100, np.zeros((4, 3)), joints), axis=1
                    )
                    np.savetxt(
                        csv_path, motion, delimiter=",",
                        header=",".join(sonic._required_columns()), comments="",
                    )
                output = io.StringIO()

                def resolve_model(reference):
                    self.assertIn(reference, output.getvalue())
                    self.assertIn("models and mesh assets", output.getvalue())
                    self.assertIn("reuses local cache", output.getvalue())
                    return model_path

                argv = [
                    module.__name__, "--csv-dir", str(root / "input"),
                    "--out-dir", str(root / "output"), "--fps", "60",
                    "--start", "1", "--end", "100", "--stride", "2",
                ]
                with patch("sys.argv", argv), redirect_stdout(output), \
                     patch.object(common, "resolve_mjcf_path", side_effect=resolve_model):
                    module.main()
                actual = load_motion(root / "output/motions/group/clip.npz")
                np.testing.assert_allclose(actual, expected, atol=1e-6)
                manifest = load_manifest(root / "output")
                self.assertEqual(manifest.qpos_dim, self.model.nq)
                self.assertEqual(manifest.payload["num_motions"], 1)
                self.assertEqual(manifest.timestep, 1 / 60)
                self.assertAlmostEqual(manifest.payload["total_hours"], 2 / 60 / 3600)

    def test_format_specific_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "Expected 36 columns"):
            lafan._build_qpos_sequence(self.model, np.zeros((2, 35)))
        with self.assertRaisesRegex(ValueError, "Missing required columns"):
            sonic._build_qpos_sequence(
                [], np.zeros((2, 0)), qpos_dim=self.model.nq,
                base_adr=self.base_adr, hinge_qpos_adrs=self.joint_adrs,
                translation_scale=0.01, angle_unit="deg",
                euler_order="xyz", euler_frame="extrinsic",
            )
        with self.assertRaisesRegex(ValueError, "zero-norm root quaternion"):
            axellwppr._build_qpos_sequence(
                self.model, np.zeros((2, 3)), np.zeros((2, 4)), np.zeros((2, 29)),
                base_adr=self.base_adr, hinge_qpos_adrs=self.joint_adrs,
            )


if __name__ == "__main__":
    unittest.main()
