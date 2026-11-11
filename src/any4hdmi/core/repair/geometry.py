from __future__ import annotations

import copy

import mujoco
import numpy as np

from any4hdmi.core.grounding import _floating_base_joint, _robot_geom_ids


def lowest_point(model, data, geom: int) -> np.ndarray:
    """Support point in world -Z, including visual mesh surfaces."""
    rotation = data.geom_xmat[geom].reshape(3, 3)
    axis = rotation[2]
    size = model.geom_size[geom]
    kind = model.geom_type[geom]
    types = mujoco.mjtGeom
    if kind == types.mjGEOM_MESH:
        mesh = model.geom_dataid[geom]
        start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
        vertices = model.mesh_vert[start:start + count]
        local = vertices[np.argmin(vertices @ axis)]
    elif kind == types.mjGEOM_BOX:
        local = -np.sign(axis) * size
    elif kind in (types.mjGEOM_SPHERE, types.mjGEOM_CAPSULE):
        local = -size[0] * axis
        if kind == types.mjGEOM_CAPSULE:
            local[2] -= size[1] * np.sign(axis[2])
    elif kind == types.mjGEOM_CYLINDER:
        norm = np.linalg.norm(axis[:2])
        local = np.r_[-size[0] * axis[:2] / max(norm, 1e-12), -size[1] * np.sign(axis[2])]
    elif kind == types.mjGEOM_ELLIPSOID:
        local = -size**2 * axis / np.linalg.norm(size * axis)
    else:
        raise ValueError(f'Unsupported repair geometry type {kind}')
    return data.geom_xpos[geom] + rotation @ local


class RepairGeometry:
    """Independent MuJoCo state; no playback model or MjData is mutated."""

    def __init__(self, model, settings):
        self.model = copy.copy(model)
        self.settings = settings
        self.root_joint = _floating_base_joint(model)
        self.root = int(model.jnt_bodyid[self.root_joint])
        self.groups = _robot_geom_ids(model)
        self.root_qadr = int(model.jnt_qposadr[self.root_joint])
        self.root_dadr = int(model.jnt_dofadr[self.root_joint])
        self.joints = [j for j in range(model.njnt)
                       if model.body_rootid[model.jnt_bodyid[j]] == self.root and j != self.root_joint]
        if any(model.jnt_type[j] not in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE) for j in self.joints):
            raise ValueError('Repair supports a free base with hinge/slide joints')
        self.dofs = np.array(list(range(self.root_dadr, self.root_dadr + 6)) +
                             [int(model.jnt_dofadr[j]) for j in self.joints])
        self.qadrs = np.array([int(model.jnt_qposadr[j]) for j in self.joints], dtype=int)
        self.limited = [j for j in self.joints if model.jnt_limited[j]]
        # Near contacts enter the same collision pipeline, respecting MJCF masks/excludes.
        self.model.geom_margin[:] = np.maximum(self.model.geom_margin, .005)
        self.data = mujoco.MjData(self.model)
        self.jac = np.zeros((3, model.nv))

    def forward(self, qpos):
        self.data.qpos[:] = qpos
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)
        mujoco.mj_collision(self.model, self.data)

    def point_jacobian(self, point, body):
        mujoco.mj_jac(self.model, self.data, self.jac, None, point, body)
        return self.jac.copy()

    def ground_rows(self):
        rows = []
        for geoms in self.groups:
            points = [lowest_point(self.model, self.data, g) for g in geoms]
            index = min(range(len(points)), key=lambda i: points[i][2])
            point, geom = points[index], geoms[index]
            jac = self.point_jacobian(point, int(self.model.geom_bodyid[geom]))[2]
            rows.append((float(point[2] - self.settings.ground_z), jac))
        return rows

    def collision_rows(self):
        rows = []
        for contact in self.data.contact:
            if min(contact.geom) < 0:
                continue
            a, b = (int(self.model.geom_bodyid[g]) for g in contact.geom)
            if a == b or self.model.body_rootid[a] != self.root or self.model.body_rootid[b] != self.root:
                continue
            ja = self.point_jacobian(contact.pos, a)
            jb = self.point_jacobian(contact.pos, b)
            jac = contact.frame[:3] @ (jb - ja)
            rows.append((float(contact.dist), jac))
        return rows

    def joint_violation(self, qpos):
        return max((max(self.model.jnt_range[j, 0] - qpos[self.model.jnt_qposadr[j]],
                        qpos[self.model.jnt_qposadr[j]] - self.model.jnt_range[j, 1], 0.)
                    for j in self.limited), default=0.)
