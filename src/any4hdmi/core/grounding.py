from __future__ import annotations

import mujoco
import numpy as np


DEFAULT_FOOT_BODIES = ("left_ankle_roll_link", "right_ankle_roll_link")
HEIGHT_TOLERANCE = 1e-6


def _floating_base_joint(model: mujoco.MjModel) -> int:
    joints = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if len(joints) != 1:
        raise ValueError("Grounding requires exactly one floating-base free joint")
    joint = int(joints[0])
    if model.body_parentid[model.jnt_bodyid[joint]] != 0:
        raise ValueError("Floating base must be attached to world")
    return joint


def _robot_geom_ids(model: mujoco.MjModel) -> list[list[int]]:
    root = int(model.jnt_bodyid[_floating_base_joint(model)])
    bodies = {}
    for geom, body in enumerate(model.geom_bodyid):
        if model.body_rootid[body] == root:
            bodies.setdefault(int(body), []).append(geom)
    if not bodies:
        raise ValueError("No geometry under the floating-base robot")
    return list(bodies.values())


def _foot_geom_ids(model: mujoco.MjModel, foot_bodies: tuple[str, ...]) -> list[list[int]]:
    if not foot_bodies:
        raise ValueError("At least one foot body is required")
    feet = []
    for name in foot_bodies:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id <= 0:
            raise ValueError(f"Foot body not found: {name}")
        descendants = {body_id}
        for child in range(body_id + 1, model.nbody):
            if int(model.body_parentid[child]) in descendants:
                descendants.add(child)
        # Include visual and collision surfaces, including toe/heel child bodies.
        geoms = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) in descendants]
        if not geoms:
            raise ValueError(f"No geometry under foot body: {name}")
        feet.append(geoms)
    return feet


def _geom_min_z(model: mujoco.MjModel, data: mujoco.MjData, geom_id: int) -> float:
    kind = model.geom_type[geom_id]
    size = model.geom_size[geom_id]
    # World Z projected onto the geom's local axes; mesh vertices are already
    # compiled/scaled into the geom frame by MuJoCo.
    axis = data.geom_xmat[geom_id].reshape(3, 3)[2]
    z = float(data.geom_xpos[geom_id, 2])
    types = mujoco.mjtGeom
    if kind == types.mjGEOM_MESH:
        mesh_id = model.geom_dataid[geom_id]
        start = model.mesh_vertadr[mesh_id]
        vertices = model.mesh_vert[start : start + model.mesh_vertnum[mesh_id]]
        return z + float((vertices @ axis).min())
    if kind == types.mjGEOM_SPHERE:
        depth = size[0]
    elif kind == types.mjGEOM_BOX:
        depth = np.abs(axis) @ size
    elif kind == types.mjGEOM_CAPSULE:
        depth = size[0] + size[1] * abs(axis[2])
    elif kind == types.mjGEOM_CYLINDER:
        depth = size[0] * np.linalg.norm(axis[:2]) + size[1] * abs(axis[2])
    elif kind == types.mjGEOM_ELLIPSOID:
        depth = np.linalg.norm(axis * size)
    else:
        raise ValueError(f"Unsupported robot geom type {kind} at geom {geom_id}")
    return z - float(depth)


def measure_foot_heights(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    *,
    foot_bodies: tuple[str, ...] = DEFAULT_FOOT_BODIES,
) -> np.ndarray:
    """Lowest world Z of each foot's visual/collision surfaces, shape (T, feet)."""
    return _measure_geom_heights(model, qpos, _foot_geom_ids(model, foot_bodies))


def measure_robot_heights(model: mujoco.MjModel, qpos: np.ndarray) -> np.ndarray:
    """Lowest world Z per robot body with geometry, excluding world/other bodies."""
    return _measure_geom_heights(model, qpos, _robot_geom_ids(model))


def _measure_geom_heights(model: mujoco.MjModel, qpos: np.ndarray, groups: list[list[int]]) -> np.ndarray:
    qpos = np.asarray(qpos)
    if qpos.ndim != 2 or qpos.shape[1] != model.nq or not len(qpos) or not np.isfinite(qpos).all():
        raise ValueError(f"Expected nonempty finite qpos[T, {model.nq}]")
    data = mujoco.MjData(model)
    heights = np.empty((len(qpos), len(groups)), dtype=np.float64)
    for frame, pose in enumerate(qpos):
        data.qpos[:] = pose
        mujoco.mj_kinematics(model, data)
        for body, geoms in enumerate(groups):
            heights[frame, body] = min(_geom_min_z(model, data, geom) for geom in geoms)
    return heights


def penetration_stats(heights: np.ndarray, ground_z: float) -> dict:
    depth = np.maximum(0.0, ground_z - heights.min(axis=1))
    count = int(np.count_nonzero(depth > HEIGHT_TOLERANCE))
    return {
        "frames": len(heights),
        "penetrating_frames": count,
        "penetrating_frame_ratio": count / len(heights),
        "max_penetration_m": float(depth.max()),
    }


def correct_foot_penetration(
    model: mujoco.MjModel,
    qpos: np.ndarray,
    *,
    timestep: float,
    foot_bodies: tuple[str, ...] = DEFAULT_FOOT_BODIES,
    ground_z: float = 0.0,
    clearance: float = 0.001,
    max_correction_speed: float = 0.2,
) -> tuple[np.ndarray, dict]:
    """Lift only root Z to clear every robot surface, including non-foot links.

    Two offline passes anticipate/release corrections without allowing the
    temporal rate limit to leave penetration unresolved. No downward snap is used.
    """
    if not np.isfinite([timestep, ground_z, clearance, max_correction_speed]).all():
        raise ValueError("Grounding parameters must be finite")
    if timestep <= 0 or clearance < 0 or max_correction_speed <= 0:
        raise ValueError("timestep and max_correction_speed must be positive; clearance must be nonnegative")
    joint = _floating_base_joint(model)
    root_body = int(model.jnt_bodyid[joint])
    for geoms in _foot_geom_ids(model, foot_bodies):
        body = int(model.geom_bodyid[geoms[0]])
        while body != 0 and body != root_body:
            body = int(model.body_parentid[body])
        if body != root_body:
            raise ValueError("Foot bodies must belong to the floating-base robot")
    z_index = int(model.jnt_qposadr[joint]) + 2
    out = np.array(qpos, dtype=np.float32, copy=True)
    groups = _robot_geom_ids(model)
    body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geoms[0]])) for geoms in groups]
    before = _measure_geom_heights(model, out, groups)
    foot_before = measure_foot_heights(model, out, foot_bodies=foot_bodies)
    correction = np.maximum(0.0, ground_z + clearance - before.min(axis=1))
    # Ignore sub-micrometre numerical residue so repeated exports are stable.
    correction[correction <= HEIGHT_TOLERANCE] = 0
    step = max_correction_speed * timestep
    for frame in range(1, len(correction)):
        correction[frame] = max(correction[frame], correction[frame - 1] - step)
    for frame in range(len(correction) - 2, -1, -1):
        correction[frame] = max(correction[frame], correction[frame + 1] - step)
    old_z = out[:, z_index].copy()
    out[:, z_index] = old_z.astype(np.float64) + correction
    after = _measure_geom_heights(model, out, groups)
    if after.min() < ground_z + clearance - HEIGHT_TOLERANCE:
        raise ValueError("Corrected float32 motion cannot meet robot clearance within 1e-6 m")
    applied = out[:, z_index].astype(np.float64) - old_z
    # A root translation moves foot surfaces by exactly the same amount.
    foot_after = foot_before + applied[:, None]

    def peak_difference(values: np.ndarray, order: int) -> float:
        return float(np.abs(np.diff(values, n=order)).max() / timestep**order) if len(values) > order else 0.0

    return out, {
        "body_names": body_names,
        "before": {
            **penetration_stats(before, ground_z),
            "min_body_z_m": before.min(axis=0).tolist(),
            "min_foot_z_m": foot_before.min(axis=0).tolist(),
        },
        "after": {
            **penetration_stats(after, ground_z),
            "min_body_z_m": after.min(axis=0).tolist(),
            "min_foot_z_m": foot_after.min(axis=0).tolist(),
        },
        "max_lift_m": float(applied.max()),
        "lifted_frames": int(np.count_nonzero(applied > HEIGHT_TOLERANCE)),
        "max_correction_speed_m_s": peak_difference(applied, 1),
        "max_root_vertical_speed_before_m_s": peak_difference(old_z.astype(np.float64), 1),
        "max_root_vertical_speed_after_m_s": peak_difference(out[:, z_index].astype(np.float64), 1),
        "max_root_vertical_acceleration_before_m_s2": peak_difference(old_z.astype(np.float64), 2),
        "max_root_vertical_acceleration_after_m_s2": peak_difference(out[:, z_index].astype(np.float64), 2),
    }
