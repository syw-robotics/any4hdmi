"""Interpretable screening events; thresholds are provisional, not action labels."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


FEATURE_VERSION = 1


@dataclass(frozen=True)
class SelectionPolicy:
    inversion: str = 'reject'
    ground_roll: str = 'reject'
    inversion_deg: float = 130.0
    min_duration_s: float = .08
    roll_deg: float = 150.0
    roll_window_s: float = 1.5
    ground_height_m: float = .45
    ground_z: float = 0.0

    def __post_init__(self):
        if self.inversion not in ('allow', 'review', 'reject') or self.ground_roll not in ('allow', 'review', 'reject'):
            raise ValueError('Event actions must be allow, review or reject')
        numbers = [v for v in asdict(self).values() if isinstance(v, (int, float))]
        if not all(math.isfinite(v) for v in numbers):
            raise ValueError('Thresholds must be finite')
        if not (90 < self.inversion_deg <= 180 and 0 < self.min_duration_s <= self.roll_window_s):
            raise ValueError('Inversion must be >90 and <=180 degrees; 0 < duration <= roll window')
        if self.roll_deg <= 0 or self.ground_height_m <= 0:
            raise ValueError('Roll angle and ground height must be positive')


def extract_features(model, qpos, *, timestep, torso_body=None):
    """Store only orientations and root height. Body-local +Z must point up."""
    qpos = np.asarray(qpos)
    if timestep <= 0 or not math.isfinite(timestep):
        raise ValueError('Timestep must be finite and positive')
    if qpos.ndim != 2 or qpos.shape[1] != model.nq or not len(qpos) or not np.isfinite(qpos).all():
        raise ValueError('Expected nonempty finite qpos with model.nq columns')
    roots = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if len(roots) != 1:
        raise ValueError('Selection requires exactly one free root joint')
    root = int(model.jnt_bodyid[roots[0]])
    torso = root
    if torso_body:
        torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, torso_body)
        if torso <= 0:
            raise ValueError(f'Torso body not found: {torso_body}')
    elif mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'torso_link') > 0:
        torso = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'torso_link')
    for joint in range(model.njnt):
        kind, address = model.jnt_type[joint], model.jnt_qposadr[joint]
        if kind in (mujoco.mjtJoint.mjJNT_FREE, mujoco.mjtJoint.mjJNT_BALL):
            start = address + (3 if kind == mujoco.mjtJoint.mjJNT_FREE else 0)
            if np.any(np.abs(np.linalg.norm(qpos[:, start:start + 4], axis=1) - 1) > .05):
                raise ValueError('Motion has non-unit quaternions')
    data = mujoco.MjData(model)
    root_quat = np.empty((len(qpos), 4), dtype=np.float32)
    torso_quat = np.empty_like(root_quat)
    height = np.empty(len(qpos), dtype=np.float32)
    for i, pose in enumerate(qpos):
        data.qpos[:] = pose
        mujoco.mj_kinematics(model, data)
        root_quat[i], torso_quat[i] = data.xquat[root], data.xquat[torso]
        height[i] = data.xpos[root, 2]
    return dict(root_quat=root_quat, torso_quat=torso_quat, root_height=height, fps=np.array(1 / timestep))


def _runs(mask):
    boundaries = np.flatnonzero(np.diff(np.r_[False, mask, False].astype(int)))
    return zip(boundaries[::2], boundaries[1::2])


def classify_features(features, policy):
    fps = float(features['fps'])
    count = len(features['root_height'])
    rotations = [Rotation.from_quat(features[key][:, [1, 2, 3, 0]]) for key in ('root_quat', 'torso_quat')]
    tilt = np.maximum(*[np.degrees(np.arccos(np.clip(r.as_matrix()[:, 2, 2], -1, 1))) for r in rotations])
    events = []
    min_frames = max(1, math.ceil(policy.min_duration_s * fps - 1e-8))
    for start, end in _runs(tilt >= policy.inversion_deg - 20):
        if end - start < min_frames:
            continue
        hard = any(b - a >= min_frames for a, b in _runs(tilt[start:end] >= policy.inversion_deg))
        events.append(dict(type='inversion', start_frame=int(start), end_frame_exclusive=int(end),
                           severity='definite' if hard else 'borderline',
                           evidence=dict(max_tilt_deg=float(tilt[start:end].max()), duration_s=(end - start) / fps)))

    # Integrate signed world-horizontal rotation over bounded windows. Pure yaw
    # vanishes and back-and-forth rocking cancels instead of accumulating forever.
    roll_angles = np.zeros(count)
    if count > 1:
        delta = (rotations[1][1:] * rotations[1][:-1].inv()).as_rotvec()[:, :2]
        cumulative = np.vstack((np.zeros((1, 2)), np.cumsum(delta, axis=0)))
        distance = np.r_[0., np.cumsum(np.linalg.norm(delta, axis=1))]
        ends = np.arange(count)
        starts = np.maximum(0, ends - max(1, round(policy.roll_window_s * fps)))
        net = np.linalg.norm(cumulative[ends] - cumulative[starts], axis=1)
        travelled = distance[ends] - distance[starts]
        low = features['root_height'] - policy.ground_z <= policy.ground_height_m
        low_sum = np.r_[0, np.cumsum(low)]
        low_fraction = (low_sum[ends + 1] - low_sum[starts]) / (ends - starts + 1)
        coherent = net >= .75 * np.maximum(travelled, 1e-8)
        eligible = coherent & (low_fraction >= .8)
        roll_angles = np.where(eligible, np.degrees(net), 0.)
        for a, b in _runs(roll_angles >= policy.roll_deg * .8):
            start, end = int(starts[a]), int(b)
            peak = float(roll_angles[a:b].max())
            event = dict(type='ground_roll', start_frame=start, end_frame_exclusive=end,
                         severity='definite' if peak >= policy.roll_deg else 'borderline',
                         evidence=dict(rotation_deg=peak, window_s=policy.roll_window_s))
            if events and events[-1]['type'] == 'ground_roll' and events[-1]['end_frame_exclusive'] >= start:
                events[-1]['end_frame_exclusive'] = end
                if event['severity'] == 'definite':
                    events[-1]['severity'] = 'definite'
                events[-1]['evidence']['rotation_deg'] = max(events[-1]['evidence']['rotation_deg'], peak)
            else:
                events.append(event)
    status = 'keep'
    reasons = set()
    for event in events:
        action = getattr(policy, event['type'])
        event['action'] = action
        if action == 'allow':
            continue
        reasons.add(event['type'])
        if action == 'reject' and event['severity'] == 'definite':
            status = 'reject'
        elif status != 'reject':
            status = 'review'
    return dict(auto_status=status, reasons=sorted(reasons), events=sorted(events, key=lambda e: e['start_frame']),
                num_frames=count, fps=fps, duration_s=max(0, count - 1) / fps,
                metrics=dict(max_tilt_deg=float(tilt.max()), max_ground_rotation_deg=float(roll_angles.max())))
