from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from any4hdmi.core.format import find_dataset_root, load_manifest
from any4hdmi.scripts.viewer import _resolve_motion_path, dataset_motions, initial_motion_label


class ViewerTest(unittest.TestCase):
    def test_resolve_motion_path_accepts_hf_uri(self) -> None:
        expected = Path("/tmp/hf-cache/snapshot/motions/example.npz")
        motion_uri = "hf://example/dataset/motions/example.npz"

        with mock.patch(
            "any4hdmi.scripts.viewer.resolve_input_paths",
            return_value=[expected],
        ) as resolve_input_paths:
            resolved = _resolve_motion_path(motion_uri)

        self.assertEqual(resolved, expected)
        resolve_input_paths.assert_called_once_with(Path.cwd(), motion_uri)

    def test_find_dataset_root_preserves_snapshot_symlink_path(self) -> None:
        with TemporaryDirectory() as temp_dir:
            cache_root = Path(temp_dir)
            dataset_root = cache_root / "snapshots" / "revision"
            motions_root = dataset_root / "motions"
            motions_root.mkdir(parents=True)
            (dataset_root / "manifest.json").write_text("{}")
            blob = cache_root / "blobs" / "hash"
            blob.parent.mkdir()
            blob.write_bytes(b"motion")
            motion = motions_root / "clip.npz"
            motion.symlink_to(blob)

            self.assertEqual(find_dataset_root(motion), dataset_root)

    def test_manifest_mjcf_preserves_snapshot_symlink_path(self) -> None:
        with TemporaryDirectory() as temp_dir:
            cache_root = Path(temp_dir)
            dataset_root = cache_root / "snapshots" / "revision"
            dataset_root.mkdir(parents=True)
            (dataset_root / "manifest.json").write_text('{"mjcf": "model.xml"}')
            blob = cache_root / "blobs" / "hash"
            blob.parent.mkdir()
            blob.write_text("<mujoco/>")
            mjcf = dataset_root / "model.xml"
            mjcf.symlink_to(blob)

            self.assertEqual(load_manifest(dataset_root).mjcf_path, mjcf)

    def test_dataset_motions_lists_nested_clips_and_selects_requested_file(self) -> None:
        with TemporaryDirectory() as temp_dir:
            dataset_root = Path(temp_dir)
            motions_root = dataset_root / "motions"
            nested = motions_root / "group"
            nested.mkdir(parents=True)
            (dataset_root / "manifest.json").write_text('{"motions_subdir": "motions"}')
            (motions_root / "b.npz").write_bytes(b"")
            (nested / "a.npz").write_bytes(b"")

            motions = dataset_motions(dataset_root, load_manifest(dataset_root))

            self.assertEqual([label for label, _ in motions], ["b.npz", "group/a.npz"])
            self.assertEqual(initial_motion_label(nested / "a.npz", motions), "group/a.npz")
            self.assertEqual(initial_motion_label(dataset_root, motions), "b.npz")

if __name__ == "__main__":
    unittest.main()
