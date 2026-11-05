from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

from any4hdmi.core.format import MOTIONS_SUBDIR, find_dataset_root, load_manifest, load_motion
from any4hdmi.scripts.viewer_contacts import ContactHighlights


def load_comparison_motion(current_path: Path, current_qpos: np.ndarray, reference: Path | None = None) -> tuple[np.ndarray | None, str]:
    try:
        current_manifest = load_manifest(find_dataset_root(current_path))
        if reference is None:
            source = current_manifest.payload.get("source", {})
            if "foot_grounding" not in source or not source.get("dataset"):
                return None, "Use --compare <original dataset or motion> to compare."
            reference = Path(source["dataset"])
        reference = reference.expanduser().absolute()
        original_manifest = load_manifest(find_dataset_root(reference))
        if reference.is_dir():
            # Resolve directory aliases without dereferencing HF motion-file
            # symlinks into the cache's unrelated blob directory.
            current = current_path.parent.resolve() / current_path.name
            relative = current.relative_to((current_manifest.root / current_manifest.payload.get("motions_subdir", MOTIONS_SUBDIR)).resolve())
            reference = original_manifest.root / original_manifest.payload.get("motions_subdir", MOTIONS_SUBDIR) / relative
        if not np.isclose(current_manifest.timestep, original_manifest.timestep, rtol=1e-6, atol=0):
            return None, "Comparison unavailable: timestep differs."
        if current_manifest.payload["qpos_names"] != original_manifest.payload["qpos_names"]:
            return None, "Comparison unavailable: qpos_names differ."
        if current_manifest.mjcf_path != original_manifest.mjcf_path:
            # Separate copies of the same MJCF are fine; a different robot is not.
            if current_manifest.mjcf_path.read_bytes() != original_manifest.mjcf_path.read_bytes():
                return None, "Comparison unavailable: MJCF models differ."
        qpos = load_motion(reference)
        if qpos.shape != current_qpos.shape or not np.isfinite(qpos).all():
            return None, "Comparison unavailable: frame count, qpos width or finite values differ."
        return qpos, str(reference)
    except (OSError, ValueError, KeyError) as exc:
        return None, f"Comparison unavailable: {exc}"


class ComparisonView:
    """A reference copy in the primary scene's camera frame; never recenter Z."""

    def __init__(self, server, model: mujoco.MjModel, scene, contacts, reference: Path | None = None) -> None:
        self.server, self.model, self.scene, self.reference = server, model, scene, reference
        self.qpos = None
        self.current_qpos = None
        self.last_key = None
        self.data = mujoco.MjData(model)
        self._mesh_groups = []
        self._fixed_geom_handles = {}
        self._scene_offset = np.zeros(3)
        self.contacts = ContactHighlights(server, model, self.data, self, controls=contacts, prefix="/comparison/contact_links")
        self.label = server.scene.add_label("/comparison/label", "Original", visible=False)
        self.current_label = server.scene.add_label("/comparison/current_label", "Current", visible=False)
        with server.gui.add_folder("Comparison", order=3, expand_by_default=False):
            self.enabled = server.gui.add_checkbox("Show original", initial_value=True)
            self.status = server.gui.add_text("Original motion", initial_value="", disabled=True)
            server.gui.add_markdown("Current and original share the source frame and camera. The original is shifted sideways only.")

    @property
    def geom_groups_visible(self):
        return self.scene.geom_groups_visible

    def load(self, path: Path, qpos: np.ndarray) -> None:
        self.last_key = None
        self.current_qpos = qpos
        self.qpos, message = load_comparison_motion(path, qpos, self.reference)
        self.status.value = message
        self.enabled.disabled = self.qpos is None
        if self.qpos is not None:
            self.contacts.load_motion(self.qpos, load_manifest(find_dataset_root(path)).timestep)
        if self.qpos is not None and not self._mesh_groups:
            for index, group in enumerate(self.scene._mesh_groups):
                handle = self.server.scene.add_batched_glb(
                    f"/comparison/bodies/{index}", group.handle.glb_data,
                    batched_positions=np.zeros((len(group.body_ids), 3)),
                    batched_wxyzs=np.tile([1., 0, 0, 0], (len(group.body_ids), 1)), visible=False,
                )
                self._mesh_groups.append(SimpleNamespace(handle=handle, body_ids=group.body_ids, group_id=group.group_id))

    def update(self, frame: int) -> None:
        visible = self.qpos is not None and bool(self.enabled.value)
        key = (frame, visible, tuple(self.geom_groups_visible), tuple(self.scene._scene_offset),
               bool(self.contacts.ground.value), self.contacts.scope.value, bool(self.contacts.self_collision.value))
        if key == self.last_key:
            return
        self.last_key = key
        self.label.visible = self.current_label.visible = visible
        if not visible:
            for group in self._mesh_groups:
                group.handle.visible = False
            for handle in self.contacts.handles.values():
                handle.visible = False
            return
        self.data.qpos[:] = self.qpos[frame]
        mujoco.mj_forward(self.model, self.data)
        self._scene_offset = self.scene._scene_offset + np.array([0., 1.4, 0.])
        for group in self._mesh_groups:
            group.handle.visible = bool(self.geom_groups_visible[group.group_id])
            group.handle.batched_positions = self.data.xpos[group.body_ids] + self._scene_offset
            group.handle.batched_wxyzs = self.data.xquat[group.body_ids]
        roots = np.flatnonzero(self.model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        body = int(self.model.jnt_bodyid[roots[0]]) if len(roots) else 1
        self.label.position = self.data.xpos[body] + self._scene_offset + np.array([0, 0, .8])
        adr = int(self.model.jnt_qposadr[roots[0]]) if len(roots) else 0
        self.current_label.position = self.current_qpos[frame, adr:adr + 3] + self.scene._scene_offset + np.array([0, 0, .8])
        self.contacts.update(force=True, frame=frame)
