from __future__ import annotations

import io
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import mujoco
import numpy as np
import trimesh

from any4hdmi.core.format import find_dataset_root, load_manifest
from any4hdmi.scripts.viewer.app import _create_motion_scene, _resolve_motion_path, dataset_motions, initial_motion_label


class ViewerTest(unittest.TestCase):
    def test_collision_layers_render_and_follow_motion(self) -> None:
        model = mujoco.MjModel.from_xml_string('''
            <mujoco><asset><material name="hidden" rgba="1 0 0 0"/></asset><worldbody>
              <geom name="floor" type="plane" size="5 5 .1" group="5"/>
              <geom name="obstacle" type="box" size=".1 .1 .1" pos="3 0 0"/>
              <body name="robot" pos="0 0 1"><freejoint/>
                <geom name="visual" type="sphere" size=".2" group="4" contype="0" conaffinity="0"/>
                <geom name="collision" type="capsule" size=".1 .3" group="4" material="hidden" rgba="1 0 0 0"/>
                <body name="fixed_link" pos="0 0 .5">
                  <geom name="affinity_only" type="sphere" size=".1" contype="0" conaffinity="2"/>
                </body>
              </body>
              <body name="fixed_robot" pos="2 0 1"><geom type="box" size=".1 .1 .1"/></body>
            </worldbody></mujoco>
        ''')
        original_contype = model.geom_contype.copy()
        original_conaffinity = model.geom_conaffinity.copy()
        server = mock.MagicMock()
        originals = {}

        def add_mesh(name, mesh, **kwargs):
            handle = mock.Mock(glb_data=mesh.export(file_type="glb"), **kwargs)
            originals[id(handle)] = handle.glb_data
            return handle

        server.scene.add_mesh_trimesh.side_effect = add_mesh
        server.scene.add_batched_meshes_trimesh.side_effect = add_mesh
        checkboxes = {}

        def add_checkbox(label, *args, initial_value, **kwargs):
            checkboxes[label] = mock.Mock(value=initial_value)
            return checkboxes[label]

        for method in ("add_checkbox", "add_dropdown", "add_rgb", "add_slider"):
            getattr(server.gui, method).side_effect = add_checkbox
        scene = _create_motion_scene(server, model)
        scene.camera_tracking_enabled = False
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        scene.update_from_mjdata(data)

        visual_handle = next(group.handle for group in scene._mesh_groups if group.group_id == 0)
        ghost = visual_handle.glb_data
        self.assertEqual(checkboxes["Appearance"].value, "Ghost")
        self.assertEqual(scene.visual_opacity, .45)
        meshes = trimesh.load(io.BytesIO(ghost), file_type="glb", force="scene")
        for mesh in meshes.geometry.values():
            self.assertEqual(mesh.visual.material.alphaMode, "BLEND")
            np.testing.assert_array_equal(mesh.visual.material.baseColorFactor, [38, 166, 255, 115])
        checkboxes["Appearance"].value = "Texture"
        checkboxes["Appearance"].on_update.call_args.args[0](None)
        self.assertEqual(visual_handle.glb_data, originals[id(visual_handle)])
        self.assertEqual(scene.visual_opacity, 1.)
        checkboxes["Appearance"].value = "Ghost"
        checkboxes["Appearance"].on_update.call_args.args[0](None)
        self.assertEqual(visual_handle.glb_data, ghost)

        np.testing.assert_array_equal(model.geom_group, [2, 2, 0, 1, 1, 1])
        np.testing.assert_array_equal(model.geom_contype, original_contype)
        np.testing.assert_array_equal(model.geom_conaffinity, original_conaffinity)
        collision_handles = [group.handle for group in scene._mesh_groups if group.group_id == 1]
        collision_handles += [handle for (_, group, _), handle in scene._fixed_geom_handles.items() if group == 1]
        self.assertTrue(collision_handles)
        for handle in collision_handles:
            self.assertFalse(handle.visible)
            mesh_scene = trimesh.load(io.BytesIO(handle.glb_data), file_type="glb", force="scene")
            for mesh in mesh_scene.geometry.values():
                self.assertEqual(mesh.visual.material.alphaMode, "BLEND")
                np.testing.assert_array_equal(mesh.visual.material.baseColorFactor, [40, 200, 90, 120])

        # Switch layers without advancing playback: the cached frame must refresh.
        checkboxes["Visual"].value = False
        checkboxes["Collision"].value = True
        checkboxes["Collision"].on_update.call_args.args[0](None)
        self.assertTrue(all(handle.visible for handle in collision_handles))
        self.assertTrue(all(not group.handle.visible for group in scene._mesh_groups if group.group_id == 0))
        self.assertTrue(all(handle.visible for (_, group, _), handle in scene._fixed_geom_handles.items() if group == 2))

        group = next(group for group in scene._mesh_groups if group.group_id == 1)
        before = group.handle.batched_positions.copy()
        data.qpos[0] += 1.5
        mujoco.mj_forward(model, data)
        scene.update_from_mjdata(data)
        np.testing.assert_allclose(group.handle.batched_positions - before, [[1.5, 0, 0]])

    def test_collision_only_robot_is_visible_by_default(self) -> None:
        model = mujoco.MjModel.from_xml_string('''
            <mujoco><worldbody>
              <geom type="plane" size="5 5 .1" contype="0" conaffinity="0"/>
              <body><freejoint/><geom type="sphere" size=".1"/></body>
            </worldbody></mujoco>
        ''')
        with mock.patch("any4hdmi.scripts.viewer.scene.ViserMujocoScene"):
            scene = _create_motion_scene(mock.MagicMock(), model)
        self.assertEqual(scene.geom_groups_visible, [False, True, True, False, False, False])

    def test_resolve_motion_path_accepts_hf_uri(self) -> None:
        expected = Path("/tmp/hf-cache/snapshot/motions/example.npz")
        motion_uri = "hf://example/dataset/motions/example.npz"

        with mock.patch(
            "any4hdmi.scripts.viewer.app.resolve_input_paths",
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

            manifest = load_manifest(dataset_root)
            motions = dataset_motions(dataset_root, manifest, dataset_root)

            self.assertEqual([label for label, _ in motions], ["b.npz", "group/a.npz"])
            self.assertEqual(initial_motion_label(nested / "a.npz", motions), "group/a.npz")
            self.assertEqual(initial_motion_label(dataset_root, motions), "b.npz")
            self.assertEqual(dataset_motions(dataset_root, manifest, nested / "a.npz"), motions)

            selected = dataset_motions(dataset_root, manifest, nested)
            self.assertEqual(selected, [("a.npz", nested / "a.npz")])
            self.assertEqual(initial_motion_label(nested, selected), "a.npz")
            self.assertEqual(dataset_motions(dataset_root, manifest, motions_root), motions)

            deeper = nested / "subgroup"
            deeper.mkdir()
            (deeper / "a.npz").write_bytes(b"")
            selected = dataset_motions(dataset_root, manifest, nested)
            self.assertEqual([label for label, _ in selected], ["a.npz", "subgroup/a.npz"])

            empty = motions_root / "empty"
            empty.mkdir()
            with self.assertRaisesRegex(FileNotFoundError, "No motion .npz files found"):
                dataset_motions(dataset_root, manifest, empty)

if __name__ == "__main__":
    unittest.main()
