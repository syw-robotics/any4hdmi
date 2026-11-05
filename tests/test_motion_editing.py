from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import numpy as np

from any4hdmi.core.editing import export_motion_clip
from any4hdmi.core.format import load_manifest, load_motion, save_motion
from any4hdmi.scripts.viewer import _run_mjviser, dataset_motions


class MotionEditingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dataset = self.root / "source"
        self.qpos = np.arange(30, dtype=np.float32).reshape(10, 3)
        self.source = save_motion(self.dataset / "motions/group/clip.npz", self.qpos)
        self.payload = {
            "dataset_name": "example", "format_version": 2,
            "mjcf": "model.xml", "timestep": 0.02, "qpos_dim": 3,
            "qpos_names": ["x", "y", "z"], "num_motions": 1,
            "source": {"fps": 50},
        }
        (self.dataset / "model.xml").write_text("<mujoco/>")
        (self.dataset / "manifest.json").write_text(json.dumps(self.payload))

    def test_export_round_trip_preserves_frames_timing_and_model(self) -> None:
        original = self.source.read_bytes()
        destination = self.root / "edited"
        path = export_motion_clip(self.source, destination, 2, 6)
        np.testing.assert_array_equal(load_motion(path), self.qpos[2:7])
        manifest = load_manifest(destination)
        self.assertEqual(manifest.timestep, 0.02)
        self.assertEqual(manifest.mjcf_path, self.dataset / "model.xml")
        self.assertEqual(manifest.payload["num_motions"], 1)
        self.assertAlmostEqual(manifest.payload["total_hours"], 5 * 0.02 / 3600)
        self.assertEqual(manifest.payload["source"]["start_frame"], 2)
        self.assertEqual(manifest.payload["source"]["end_frame_inclusive"], 6)
        self.assertEqual(manifest.payload["source"]["motion"], "motions/group/clip.npz")
        self.assertEqual(dataset_motions(destination, manifest, destination), [("clip.npz", path)])
        self.assertEqual(self.source.read_bytes(), original)
        self.assertEqual(json.loads((self.dataset / "manifest.json").read_text()), self.payload)

    def test_single_frame_full_motion_and_hf_model(self) -> None:
        self.payload["mjcf"] = "hf://example/models@main/model.xml"
        (self.dataset / "manifest.json").write_text(json.dumps(self.payload))
        for start, end in [(9, 9), (0, 9)]:
            destination = self.root / f"edited_{start}_{end}"
            path = export_motion_clip(self.source, destination, start, end)
            np.testing.assert_array_equal(load_motion(path), self.qpos[start:end + 1])
            self.assertEqual(load_manifest(destination).payload["mjcf"], self.payload["mjcf"])

    def test_reject_invalid_bounds_and_existing_directory(self) -> None:
        destination = self.root / "edited"
        for start, end in [(-1, 2), (5, 4), (0, 10)]:
            with self.subTest(start=start, end=end), self.assertRaises(ValueError):
                export_motion_clip(self.source, destination, start, end)
            self.assertFalse(destination.exists())
        with self.assertRaises(FileExistsError):
            export_motion_clip(self.source, self.dataset, 0, 1)
        np.testing.assert_array_equal(load_motion(self.source), self.qpos)

    def test_viewer_trim_preview_reset_switch_and_export(self) -> None:
        controls = {}
        server = mock.MagicMock()

        def add_control(label, *args, **kwargs):
            handle = mock.MagicMock()
            handle.value = kwargs.get("initial_value")
            handle.on_click.side_effect = lambda callback: setattr(handle, "click_callback", callback) or callback
            handle.on_update.side_effect = lambda callback: setattr(handle, "update_callback", callback) or callback
            handle.on_trigger.side_effect = lambda callback: setattr(handle, "trigger_callback", callback) or callback
            controls[label] = handle
            return handle

        for method in ["add_button", "add_number", "add_slider", "add_text", "add_dropdown", "add_command"]:
            getattr(server.gui, method).side_effect = add_control
        destination = self.root / "gui_export"
        data = mock.MagicMock()
        data.qpos = np.zeros(3)
        data.qvel = np.zeros(3)
        second = save_motion(self.dataset / "motions/second.npz", self.qpos[:4])

        def interact(_):
            self.assertEqual(controls["Motion"].label, "Motion (1/2)")
            self.assertEqual(controls["Source frame (0-based)"].value, 3)
            server.gui.add_command.assert_called_once_with("Start / Pause", hotkey="space")
            slider_args = {call.args[0]: call.kwargs for call in server.gui.add_slider.call_args_list}
            self.assertEqual(slider_args["Frame"]["step"], 1)
            self.assertGreater(slider_args["Frame"]["max"], slider_args["Frame"]["min"])
            self.assertEqual(slider_args["Speed"]["step"], 0.5)
            self.assertEqual(slider_args["Speed"]["min"], 0.5)

            # Space works without first clicking the playback button.
            controls["Start / Pause"].trigger_callback(None)
            self.assertEqual(controls["Status"].value, "Paused")
            controls["Start / Pause"].trigger_callback(None)
            self.assertEqual(controls["Status"].value, "Playing")
            # Playback's own slider update must not pause playback.
            frame = controls["Frame"]
            frame.update_callback(mock.Mock(client_id=None, target=frame))
            self.assertEqual(controls["Status"].value, "Playing")
            # A user seek to the last programmatically set value still pauses.
            frame.update_callback(mock.Mock(client_id=0, target=frame))
            self.assertEqual(controls["Status"].value, "Paused")
            controls["Frame"].value = 2
            frame.update_callback(mock.Mock(client_id=0, target=frame))

        def edit(_):
            # With --start=1 and --stride=2, playback cursor 2 is source frame 5.
            self.assertEqual(controls["Source frame (0-based)"].value, 5)
            self.assertEqual(controls["Status"].value, "Paused")
            controls["Set start to current frame"].click_callback(None)
            self.assertEqual(controls["Start frame (inclusive)"].value, 5)
            controls["Start frame (inclusive)"].value = 3
            controls["End frame (inclusive)"].value = 3
            controls["Preview selection"].click_callback(None)
            self.assertTrue(controls["Frame"].disabled)
            self.assertEqual(controls["Frame"].max, 1)
            controls["End frame (inclusive)"].value = 7
            controls["Preview selection"].click_callback(None)
            self.assertFalse(controls["Frame"].disabled)
            self.assertEqual(controls["Frame"].max, 2)
            np.testing.assert_array_equal(data.qpos, self.qpos[3])
            controls["New dataset directory"].value = str(destination)
            controls["Export clip"].click_callback(None)
            np.testing.assert_array_equal(load_motion(destination / "motions/clip.npz"), self.qpos[3:8])
            controls["Export clip"].click_callback(None)
            self.assertIn("Export failed", controls["Edit status"].value)
            controls["Start frame (inclusive)"].value = 8
            controls["Preview selection"].click_callback(None)
            self.assertIn("Expected", controls["Edit status"].value)
            controls["Reset selection"].click_callback(None)
            self.assertEqual(controls["Frame"].max, 4)
            np.testing.assert_array_equal(data.qpos, self.qpos[0])
            controls["Next"].click_callback(None)
            self.assertEqual(controls["Motion"].label, "Motion (2/2)")
            self.assertEqual(controls["Start frame (inclusive)"].value, 1)
            self.assertEqual(controls["End frame (inclusive)"].value, 3)
            controls["Next"].click_callback(None)
            self.assertEqual(controls["Motion"].label, "Motion (1/2)")
            controls["Previous"].click_callback(None)
            self.assertEqual(controls["Motion"].label, "Motion (2/2)")
            controls["Motion"].value = "group/clip.npz"
            controls["Motion"].update_callback(None)
            self.assertEqual(controls["Motion"].label, "Motion (1/2)")
            raise StopIteration

        with mock.patch("viser.ViserServer", return_value=server), \
             mock.patch("any4hdmi.scripts.viewer._create_motion_scene"), \
             mock.patch("any4hdmi.scripts.viewer.mujoco.mj_forward"), \
             mock.patch("any4hdmi.scripts.viewer.time.sleep") as sleep:
            # A function side_effect lets each playback iteration drive one UI action.
            actions = iter([interact, edit])
            sleep.side_effect = lambda delay: next(actions)(delay)
            with self.assertRaises(StopIteration):
                _run_mjviser(
                    mock.Mock(nq=3), data,
                    [("group/clip.npz", self.source), ("second.npz", second)], "group/clip.npz",
                    start=1, end=-1, stride=2, frame_dt=1000, loop=False, port=8080,
                )


if __name__ == "__main__":
    unittest.main()
