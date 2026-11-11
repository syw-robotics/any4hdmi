from __future__ import annotations

from dataclasses import asdict, dataclass
from collections.abc import Callable

import mujoco
import numpy as np
from scipy import sparse
from scipy.optimize import lsq_linear
from scipy.sparse.linalg import LinearOperator

from any4hdmi.core.repair.geometry import RepairGeometry
from any4hdmi.core.support import sole_samples, stable_support_mask


@dataclass(frozen=True)
class RepairSettings:
    ground: bool = True
    self_collision: bool = True
    ground_z: float = 0.
    clearance: float = .001
    tolerance: float = .001
    transition_seconds: float = .3
    max_root_change: float = .15
    max_joint_change: float = .35
    max_support_drift: float = .005
    max_correction_speed: float = 2.
    max_correction_acceleration: float = 30.
    iterations: int = 20

    def __post_init__(self):
        values = asdict(self)
        if not all(np.isfinite(v) for v in values.values()):
            raise ValueError('Repair settings must be finite')
        positive = ('tolerance', 'transition_seconds', 'max_root_change', 'max_joint_change',
                    'max_support_drift', 'max_correction_speed', 'max_correction_acceleration', 'iterations')
        if any(values[key] <= 0 for key in positive) or self.clearance < 0:
            raise ValueError('Repair limits must be positive and clearance nonnegative')


class RepairCancelled(Exception):
    pass


def checkpoint(cancel):
    if cancel is not None and cancel():
        raise RepairCancelled('Cancelled')


def validate_motion(model, qpos, timestep):
    qpos = np.asarray(qpos)
    if qpos.ndim != 2 or qpos.shape[1] != model.nq or not len(qpos) or not np.isfinite(qpos).all():
        raise ValueError(f'Expected nonempty finite qpos[T, {model.nq}]')
    if not np.isfinite(timestep) or timestep <= 0:
        raise ValueError('timestep must be positive and finite')
    for joint in np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE):
        adr = int(model.jnt_qposadr[joint]) + 3
        if not np.allclose(np.linalg.norm(qpos[:, adr:adr + 4], axis=1), 1, atol=1e-4):
            raise ValueError('Root quaternions must be normalized')


def differences(model, source, candidate):
    delta = np.zeros((len(source), model.nv))
    for frame in range(len(source)):
        mujoco.mj_differentiatePos(model, delta[frame], 1., source[frame], candidate[frame])
    return delta


def scan(geometry, qpos, cancel=None):
    depths = np.zeros((len(qpos), 3))
    for frame, pose in enumerate(qpos):
        checkpoint(cancel)
        geometry.forward(pose)
        depths[frame] = [max(0., -min(d for d, _ in geometry.ground_rows())),
                         max(0., -min((d for d, _ in geometry.collision_rows()), default=0.)),
                         geometry.joint_violation(pose)]
    return depths


def metrics(depths, tolerance):
    result = {}
    for col, kind in enumerate(('ground', 'self', 'joint')):
        result[f'{kind}_frames'] = int(np.count_nonzero(depths[:, col] > tolerance))
        result[f'{kind}_depth_m' if col < 2 else 'joint_violation'] = float(depths[:, col].max())
    return result


def support_targets(geometry, source, timestep, cancel):
    feet, samples = sole_samples(geometry.model)
    targets = {body: np.empty((len(source), len(points), 3)) for body, points in samples.items()}
    for frame, pose in enumerate(source):
        checkpoint(cancel)
        geometry.forward(pose)
        for body, points in samples.items():
            targets[body][frame] = points @ geometry.data.xmat[body].reshape(3, 3).T + geometry.data.xpos[body]
    anchors = []
    for body, points in samples.items():
        for i, point in enumerate(points):
            trajectory = targets[body][:, i]
            floor_points = trajectory[:, None].copy()
            floor_points[:, :, 2] -= geometry.settings.ground_z
            mask = stable_support_mask(floor_points, timestep)
            if mask.any():
                anchors.append((body, point, trajectory, mask))
    # Availability means the expected feet have usable samples, not that a clip is planted.
    available = len(feet) == 2 and all(any(body in samples for body in group) for group in feet.values())
    return anchors, available


def support_drift(geometry, candidate, anchors, cancel):
    maximum = 0.
    for frame, pose in enumerate(candidate):
        checkpoint(cancel)
        geometry.forward(pose)
        for body, point, target, mask in anchors:
            if mask[frame]:
                world = geometry.data.xpos[body] + geometry.data.xmat[body].reshape(3, 3) @ point
                maximum = max(maximum, float(np.linalg.norm(world - target[frame])))
    return maximum


def temporal_matrix(length, dofs, frames, timestep):
    """Penalize correction, its speed and acceleration, including fixed boundaries."""
    identity = sparse.eye(length, format='csr')
    first = (identity[1:] - identity[:-1]) * (.04 / timestep)
    second = (identity[2:] - 2 * identity[1:-1] + identity[:-2]) * (.004 / timestep**2)
    time = sparse.vstack((identity, first, second), format='csr')[:, frames]
    weights = np.ones(dofs)
    weights[:3] = 4.
    weights[3:6] = 2.
    return sparse.kron(time, sparse.diags(weights), format='csr')


def linear_rows(geometry, candidate, frames, anchors, before, settings, cancel):
    blocks, targets = [], []
    for frame in frames:
        checkpoint(cancel)
        geometry.forward(candidate[frame])
        rows, rhs = [], []
        for kind, contacts in enumerate((geometry.ground_rows(), geometry.collision_rows())):
            enabled = settings.ground if kind == 0 else settings.self_collision
            margin = settings.clearance if enabled else -before[frame, kind]
            for distance, jac in contacts:
                if distance < margin + .002:
                    rows.append(1000. * jac[geometry.dofs])
                    rhs.append(1000. * (margin - distance))
        for body, point, reference, mask in anchors:
            if not mask[frame]:
                continue
            world = geometry.data.xpos[body] + geometry.data.xmat[body].reshape(3, 3) @ point
            jac = geometry.point_jacobian(world, body)[:, geometry.dofs]
            rows.extend(200. * jac)
            rhs.extend(200. * (reference[frame] - world))
        blocks.append(sparse.csr_matrix(np.asarray(rows).reshape(-1, len(geometry.dofs))))
        targets.extend(rhs)
    return sparse.block_diag(blocks, format='csr'), np.asarray(targets)


def step_bounds(geometry, candidate, frames, correction, settings):
    limits = np.full(len(geometry.dofs), settings.max_joint_change)
    limits[:3] = settings.max_root_change
    lower = np.maximum(-.08, -limits - correction)
    upper = np.minimum(.08, limits - correction)
    for index, joint in enumerate(geometry.joints, start=6):
        if not geometry.model.jnt_limited[joint]:
            continue
        adr = geometry.model.jnt_qposadr[joint]
        lo, hi = geometry.model.jnt_range[joint]
        # An already invalid source can need several trust-region steps to reach its limit.
        lower[:, index] = np.maximum(lower[:, index], np.minimum(.079, lo - candidate[frames, adr]))
        upper[:, index] = np.minimum(upper[:, index], np.maximum(-.079, hi - candidate[frames, adr]))
    if np.any(lower >= upper):
        raise ValueError('Joint limits conflict with the allowed correction')
    return lower.ravel(), upper.ravel()


def cancellable_matrix(matrix, cancel):
    if cancel is None:
        return matrix
    def multiply(vector):
        checkpoint(cancel)
        return matrix @ vector
    def transpose_multiply(vector):
        checkpoint(cancel)
        return matrix.T @ vector
    return LinearOperator(matrix.shape, matvec=multiply, rmatvec=transpose_multiply, dtype=np.float64)


def solve(geometry, source, before, anchors, timestep, settings, cancel):
    bad = before[:, 2] > settings.tolerance
    if settings.ground:
        bad |= before[:, 0] > settings.tolerance
    if settings.self_collision:
        bad |= before[:, 1] > settings.tolerance
    radius = max(2, int(np.ceil(settings.transition_seconds / timestep)))
    active = np.zeros(len(source), dtype=bool)
    for frame in np.flatnonzero(bad):
        active[max(0, frame - radius):min(len(source), frame + radius + 1)] = True
    frames = np.flatnonzero(active)
    candidate = source.astype(np.float64, copy=True)
    if not len(frames):
        return candidate
    temporal = temporal_matrix(len(source), len(geometry.dofs), frames, timestep)
    for _ in range(settings.iterations):
        checkpoint(cancel)
        correction = differences(geometry.model, source[frames], candidate[frames])[:, geometry.dofs]
        constraints, targets = linear_rows(geometry, candidate, frames, anchors, before, settings, cancel)
        matrix = sparse.vstack((temporal, constraints), format='csr')
        rhs = np.r_[-temporal @ correction.ravel(), targets]
        bounds = step_bounds(geometry, candidate, frames, correction, settings)
        result = lsq_linear(cancellable_matrix(matrix, cancel), rhs, bounds=bounds,
                            tol=1e-5, lsmr_tol=1e-6, max_iter=30)
        checkpoint(cancel)
        if not result.success or not np.isfinite(result.x).all():
            raise ValueError(f'Repair least-squares solve failed: {result.message}')
        steps = result.x.reshape(len(frames), -1)
        for frame, step in zip(frames, steps):
            velocity = np.zeros(geometry.model.nv)
            velocity[geometry.dofs] = step
            mujoco.mj_integratePos(geometry.model, candidate[frame], velocity, 1.)
        if np.abs(steps).max() < 1e-5:
            break
    return candidate.astype(np.float32)


def result_report(geometry, source, candidate, before, after, anchors, available, timestep, settings, cancel):
    delta = differences(geometry.model, source, candidate)[:, geometry.dofs]
    speed = float(np.abs(np.diff(delta, axis=0)).max() / timestep) if len(delta) > 1 else 0.
    acceleration = float(np.abs(np.diff(delta, n=2, axis=0)).max() / timestep**2) if len(delta) > 2 else 0.
    drift = support_drift(geometry, candidate, anchors, cancel)
    changed = not np.array_equal(source, candidate)
    reasons = []
    for col, kind in enumerate(('ground penetration', 'self collision', 'joint limits')):
        if after[:, col].max() > settings.tolerance:
            reasons.append(f'Residual {kind}')
        if np.any(after[:, col] > before[:, col] + settings.tolerance):
            reasons.append(f'Increased {kind}')
    if changed and not available:
        reasons.append('Support inference unavailable: expected both G1 feet with surface samples')
    if drift > settings.max_support_drift:
        reasons.append('Support drift exceeds limit')
    if speed > settings.max_correction_speed or acceleration > settings.max_correction_acceleration:
        reasons.append('Correction changes too rapidly')
    if np.abs(delta[:, :3]).max() > settings.max_root_change + 1e-5 or np.abs(delta[:, 3:]).max() > settings.max_joint_change + 1e-5:
        reasons.append('Pose change exceeds limit')
    outcome = 'needs_review' if reasons else 'passed' if changed else 'unchanged'
    return dict(outcome=outcome, reasons=reasons, frames=len(source), settings=asdict(settings),
                before=metrics(before, settings.tolerance), after=metrics(after, settings.tolerance),
                support_available=available, support_drift_m=drift,
                root_change_m=float(np.linalg.norm(delta[:, :3], axis=1).max()),
                joint_change_max=float(np.abs(delta[:, 6:]).max()) if delta.shape[1] > 6 else 0.,
                correction_speed_max=speed, correction_acceleration_max=acceleration)


def inspect_motion(model, qpos, timestep, settings=RepairSettings(), cancel=None):
    validate_motion(model, qpos, timestep)
    checkpoint(cancel)
    geometry = RepairGeometry(model, settings)
    before = scan(geometry, qpos, cancel)
    anchors, available = support_targets(geometry, qpos, timestep, cancel)
    report = result_report(geometry, qpos, qpos, before, before, anchors, available, timestep, settings, cancel)
    report['operation'] = 'check'
    return report


def root_lift(geometry, source, depths, timestep, settings, cancel):
    lift = np.where(depths > settings.tolerance, depths + settings.clearance, 0.)
    step = .2 * timestep
    for frame in range(1, len(lift)):
        checkpoint(cancel)
        lift[frame] = max(lift[frame], lift[frame - 1] - step)
    for frame in range(len(lift) - 2, -1, -1):
        checkpoint(cancel)
        lift[frame] = max(lift[frame], lift[frame + 1] - step)
    candidate = source.copy()
    candidate[:, geometry.root_qadr + 2] += lift
    return candidate


def repair_motion(model, qpos, timestep, settings=RepairSettings(), cancel: Callable | None = None):
    validate_motion(model, qpos, timestep)
    checkpoint(cancel)
    source = np.array(qpos, dtype=np.float32, copy=True)
    geometry = RepairGeometry(model, settings)
    before = scan(geometry, source, cancel)
    anchors, available = support_targets(geometry, source, timestep, cancel)
    only_ground = settings.ground and before[:, 0].max() > settings.tolerance and before[:, 1:].max() <= settings.tolerance
    if only_ground:
        candidate = root_lift(geometry, source, before[:, 0], timestep, settings, cancel)
        after = scan(geometry, candidate, cancel)
        report = result_report(geometry, source, candidate, before, after, anchors, available, timestep, settings, cancel)
        if report['outcome'] == 'passed':
            report.update(operation='repair', method='root_lift')
            return candidate, report
    candidate = solve(geometry, source, before, anchors, timestep, settings, cancel)
    after = scan(geometry, candidate, cancel)
    report = result_report(geometry, source, candidate, before, after, anchors, available, timestep, settings, cancel)
    report.update(operation='repair', method='joint')
    return candidate, report
