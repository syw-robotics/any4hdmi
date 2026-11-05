from __future__ import annotations

import mujoco
import numpy as np

from any4hdmi.core.grounding import DEFAULT_FOOT_BODIES, HEIGHT_TOLERANCE
from any4hdmi.scripts.viewer_support import infer_support


GROUND_COLOR = (255, 205, 40)
SELF_COLOR = (240, 60, 60)
GROUND_CONTACT_DISTANCE = .005  # Visual near-contact tolerance, in metres.


def ground_link_distances(model: mujoco.MjModel, data: mujoco.MjData, floor_id: int, *, feet_only: bool) -> dict[int, float]:
    feet = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in DEFAULT_FOOT_BODIES} - {-1}
    for body in range(1, model.nbody):
        if int(model.body_parentid[body]) in feet:
            feet.add(body)
    distances = {}
    for geom in range(model.ngeom):
        body = int(model.geom_bodyid[geom])
        if body == 0 or (feet_only and body not in feet):
            continue
        if model.geom_type[geom] in (mujoco.mjtGeom.mjGEOM_PLANE, mujoco.mjtGeom.mjGEOM_HFIELD):
            continue
        # Include visual surfaces so mesh penetration is visible even when the
        # model's smaller collision primitives do not yet touch the ground.
        distance = mujoco.mj_geomDistance(model, data, floor_id, geom, 1.0, None)
        distances[body] = min(distances.get(body, 1.), float(distance))
    return distances


def ground_contact_bodies(model: mujoco.MjModel, data: mujoco.MjData, floor_id: int, *, feet_only: bool) -> set[int]:
    return {body for body, distance in ground_link_distances(model, data, floor_id, feet_only=feet_only).items() if distance <= GROUND_CONTACT_DISTANCE + HEIGHT_TOLERANCE}


def self_collision_bodies(model: mujoco.MjModel, data: mujoco.MjData) -> set[int]:
    colliding = set()
    for contact in data.contact:
        if contact.dist > 0 or min(contact.geom) < 0:
            continue
        left, right = (int(model.geom_bodyid[geom]) for geom in contact.geom)
        # Separate scene objects and the floor do not count as self collision.
        if left != right and model.body_rootid[left] != 0 and model.body_rootid[left] == model.body_rootid[right]:
            colliding.update((left, right))
    return colliding


class ContactHighlights:
    def __init__(self, server, model: mujoco.MjModel, data: mujoco.MjData, scene, *, controls=None, prefix="/contact_links") -> None:
        self.server, self.model, self.data, self.scene = server, model, data, scene
        self.prefix = prefix
        self.dirty = True
        self.visibility = None
        self.offset = None
        self.support = {}
        self.frame = 0
        self.handles = {}
        self.geoms = {}
        for geom in range(model.ngeom):
            body, group = int(model.geom_bodyid[geom]), int(model.geom_group[geom])
            if body and group in (0, 1) and model.geom_rgba[geom, 3] > 0:
                self.geoms.setdefault((body, group), []).append(geom)
        planes = np.flatnonzero(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE)
        named_floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.floor_id = named_floor if named_floor in planes else int(planes[0]) if len(planes) else -1
        if controls is None:
            with server.gui.add_folder("Contacts", order=2):
                self.ground = server.gui.add_checkbox("Stable foot support", initial_value=False, disabled=self.floor_id < 0)
                self.scope = server.gui.add_dropdown("Penetration scan links", ("Feet", "All links"), initial_value="Feet")
                self.self_collision = server.gui.add_checkbox("Self collision", initial_value=False)
                server.gui.add_markdown("🟡 Inferred stable foot support (not sliding) · 🔴 Self collision (takes priority)")
        else:
            self.ground, self.scope, self.self_collision = controls.ground, controls.scope, controls.self_collision

        def mark_dirty(_) -> None:
            self.dirty = True

        for control in (self.ground, self.scope, self.self_collision):
            control.on_update(mark_dirty)

    def load_motion(self, qpos: np.ndarray, timestep: float) -> None:
        self.support = infer_support(self.model, qpos, timestep, self.floor_id)
        self.frame = 0
        self.dirty = True

    def update(self, *, force: bool = False, frame: int | None = None) -> None:
        if frame is not None and frame != self.frame:
            self.frame = frame
            self.dirty = True
        visibility = tuple(self.scene.geom_groups_visible)
        offset = tuple(self.scene._scene_offset)
        if not force and not self.dirty and visibility == self.visibility and offset == self.offset:
            return
        self.dirty, self.visibility = False, visibility
        self.offset = offset
        colors = {}
        if self.ground.value and self.floor_id >= 0:
            bodies = {body for body, mask in self.support.items() if mask[self.frame]}
            colors.update(dict.fromkeys(bodies, GROUND_COLOR))
        if self.self_collision.value:
            colors.update(dict.fromkeys(self_collision_bodies(self.model, self.data), SELF_COLOR))
        active = {key for key in self.geoms if key[0] in colors and visibility[key[1]]}
        for key, handle in self.handles.items():
            if key not in active:
                handle.visible = False
        for (body, group), geoms in self.geoms.items():
            if body not in colors or not visibility[group]:
                continue
            key = (body, group)
            if key not in self.handles:
                from mjviser.conversions import merge_geoms

                mesh = merge_geoms(self.model, geoms)
                self.handles[key] = self.server.scene.add_mesh_simple(
                    f"{self.prefix}/{body}/{group}", mesh.vertices, mesh.faces,
                    color=colors[body], opacity=.6 if group == 1 else 1.0, visible=False,
                )
            handle = self.handles[key]
            handle.color = colors[body]
            handle.position = self.data.xpos[body] + self.scene._scene_offset
            handle.wxyz = self.data.xquat[body]
            handle.visible = True
        # GLB batches may contain several links with identical geometry. Hide
        # only highlighted instances; keep the original materials for the rest.
        for group in self.scene._mesh_groups:
            group.handle.batched_scales = np.asarray([
                0.0 if (int(body), group.group_id) in active else 1.0 for body in group.body_ids
            ], dtype=np.float32)
        for (body, group, _), handle in self.scene._fixed_geom_handles.items():
            handle.visible = visibility[group] and (body, group) not in active
