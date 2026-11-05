from __future__ import annotations

import math

import mujoco
import numpy as np

from any4hdmi.core.grounding import DEFAULT_FOOT_BODIES


def stable_support_mask(points: np.ndarray, timestep: float) -> np.ndarray:
    """Infer stable support from fixed sole points in the static floor frame."""
    result = np.zeros(len(points), dtype=bool)
    if len(points) < 2:
        return result
    # Taking the larger adjacent speed avoids a false zero at swing reversals.
    edge_speed = np.linalg.norm(np.diff(points, axis=0), axis=-1) / timestep
    speed = np.maximum(np.concatenate((edge_speed[:1], edge_speed)),
                       np.concatenate((edge_speed, edge_speed[-1:])))
    height = points[:, :, 2]
    enter = ((height >= -.01) & (height <= .02) & (speed <= .15)).any(axis=1)
    hold = ((height >= -.015) & (height <= .03) & (speed <= .25)).any(axis=1)
    # Offline confirmation backfills the qualifying run, without playback delay.
    minimum_frames = max(2, math.ceil(.06 / timestep))
    run = 0
    active = False
    for frame in range(len(points)):
        if active and hold[frame]:
            result[frame] = True
            continue
        active = False
        run = run + 1 if enter[frame] else 0
        if run >= minimum_frames:
            result[frame - run + 1:frame + 1] = True
            active = True
            run = 0
    return result


def infer_support(model: mujoco.MjModel, qpos: np.ndarray, timestep: float, floor_id: int) -> dict[int, np.ndarray]:
    """Full-source-clip masks for G1 feet and descendants; never modify qpos."""
    if floor_id < 0:
        return {}
    from mjviser.conversions import merge_geoms

    feet = {}
    samples = {}
    for name in DEFAULT_FOOT_BODIES:
        root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if root < 0:
            continue
        bodies = {root}
        for body in range(root + 1, model.nbody):
            if int(model.body_parentid[body]) in bodies:
                bodies.add(body)
        feet[root] = bodies
        for body in sorted(bodies):
            geoms = [g for g in range(model.ngeom) if model.geom_bodyid[g] == body
                     and model.geom_type[g] not in (mujoco.mjtGeom.mjGEOM_PLANE, mujoco.mjtGeom.mjGEOM_HFIELD)]
            if not geoms:
                continue
            vertices = np.asarray(merge_geoms(model, geoms).vertices)
            # G1 foot-local Z points up. Select actual surface vertices near the
            # sole, rather than virtual bounding-box corners outside the mesh.
            sole = vertices[vertices[:, 2] <= vertices[:, 2].min() + .005]
            low, high = sole[:, :2].min(axis=0), sole[:, :2].max(axis=0)
            corners = np.array([[x, y] for x in (low[0], high[0]) for y in (low[1], high[1])])
            indices = np.unique(np.argmin(np.linalg.norm(sole[:, None, :2] - corners, axis=-1), axis=0))
            samples[body] = sole[indices]
    if not samples:
        return {}
    trajectories = {body: np.empty((len(qpos), len(points), 3)) for body, points in samples.items()}
    data = mujoco.MjData(model)
    for frame, pose in enumerate(qpos):
        data.qpos[:] = pose
        mujoco.mj_kinematics(model, data)
        floor_rotation = data.geom_xmat[floor_id].reshape(3, 3)
        for body, points in samples.items():
            world = points @ data.xmat[body].reshape(3, 3).T + data.xpos[body]
            trajectories[body][frame] = (world - data.geom_xpos[floor_id]) @ floor_rotation
    masks = {}
    for bodies in feet.values():
        points = [trajectories[body] for body in sorted(bodies) if body in trajectories]
        if points:
            mask = stable_support_mask(np.concatenate(points, axis=1), timestep)
            masks.update(dict.fromkeys(bodies, mask))
    return masks
