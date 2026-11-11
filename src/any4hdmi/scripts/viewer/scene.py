"""Shared motion geometry and appearance controls for playback and live previews."""

import io

import mujoco
import trimesh
from mjviser import ViserMujocoScene


def create_motion_scene(server, model: mujoco.MjModel):
    # Model vendors use different geom groups; use actual collision flags.
    robot = (model.geom_bodyid != 0) & (model.geom_type != mujoco.mjtGeom.mjGEOM_PLANE)
    collision = robot & ((model.geom_contype != 0) | (model.geom_conaffinity != 0))
    visual = robot & ~collision
    model.geom_group[:] = 2
    model.geom_group[visual] = 0
    model.geom_group[collision] = 1
    model.geom_matid[collision] = -1
    model.geom_rgba[collision] = (40 / 255, 200 / 255, 90 / 255, 120 / 255)

    scene = ViserMujocoScene(server, model, num_envs=1)
    handles = [(group.group_id, group.handle) for group in scene._mesh_groups]
    handles += [(group, handle) for (_, group, _), handle in scene._fixed_geom_handles.items()]
    originals = [(group, handle, handle.glb_data) for group, handle in handles if group in (0, 1)]
    scene.appearance_revision = 0
    scene.visual_opacity = .45
    has_visual = bool(visual.any())
    scene.geom_groups_visible = [has_visual, not has_visual and bool(collision.any()), True, False, False, False]
    scene._sync_visibilities()

    with server.gui.add_folder("Robot"):
        visual_checkbox = server.gui.add_checkbox("Visual", initial_value=scene.geom_groups_visible[0])
        collision_checkbox = server.gui.add_checkbox("Collision", initial_value=scene.geom_groups_visible[1])
        appearance = server.gui.add_dropdown("Appearance", ("Ghost", "Texture"), initial_value="Ghost")
        color = server.gui.add_rgb("Ghost color", initial_value=(38, 166, 255))
        opacity = server.gui.add_slider("Ghost opacity", min=.05, max=1., step=.05, initial_value=.45)

    def apply_appearance():
        ghost = appearance.value == "Ghost"
        scene.visual_opacity = float(opacity.value) if ghost else 1.
        color.disabled = opacity.disabled = not ghost
        for group, handle, original in originals:
            if group == 0 and not ghost:
                handle.glb_data = original
                continue
            rgba = (*color.value, round(255 * scene.visual_opacity)) if group == 0 else (40, 200, 90, 120)
            meshes = trimesh.load(io.BytesIO(original), file_type="glb", force="scene")
            for mesh in meshes.geometry.values():
                material = trimesh.visual.material.PBRMaterial(
                    baseColorFactor=rgba, alphaMode="BLEND", doubleSided=True,
                )
                mesh.visual = trimesh.visual.TextureVisuals(material=material)
            handle.glb_data = meshes.export(file_type="glb")
        scene.appearance_revision += 1

    def update_appearance(_):
        scene._apply_visualization_change(apply_appearance)

    def update_layers(_):
        def mutate():
            scene.geom_groups_visible[:2] = [bool(visual_checkbox.value), bool(collision_checkbox.value)]
            scene._sync_visibilities()
        scene._apply_visualization_change(mutate)

    for control in (appearance, color, opacity):
        control.on_update(update_appearance)
    for control in (visual_checkbox, collision_checkbox):
        control.on_update(update_layers)
    apply_appearance()
    return scene
