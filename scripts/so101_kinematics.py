"""Forward and inverse kinematics for the SO-101 follower arm (numpy only).

The joint table below is copied from the official URDF
(TheRobotStudio/SO-ARM100, Simulation/SO101/so101_new_calib.urdf).
Joint angles are in DEGREES and use the same zero and sign convention as
LeRobot with use_degrees=True: 0 deg is the middle of each joint's range.

Coordinate frame (metres), origin at the bottom centre-back of the base:
    x  forward, away from the base
    y  to the robot's left
    z  up
"""

from __future__ import annotations

import numpy as np

ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

# (xyz, rpy) of each joint origin in its parent link; every joint turns about its local z axis.
_JOINT_ORIGINS = [
    ((0.0388353, -8.97657e-09, 0.0624), (3.14159, 4.18253e-17, -3.14159)),  # shoulder_pan
    ((-0.0303992, -0.0182778, -0.0542), (-1.5708, -1.5708, 0.0)),  # shoulder_lift
    ((-0.11257, -0.028, 1.73763e-16), (-3.63608e-16, 8.74301e-16, 1.5708)),  # elbow_flex
    ((-0.1349, 0.0052, 3.62355e-17), (4.02456e-15, 8.67362e-16, -1.5708)),  # wrist_flex
    ((5.55112e-17, -0.0611, 0.0181), (1.5708, 0.0486795, 3.14159)),  # wrist_roll
]
# Fixed transform from the gripper link to the "gripper frame" at the fingertips.
_TIP_ORIGIN = ((-0.0079, -0.000218121, -0.0981274), (0.0, 3.14159, 0.0))

# Mechanical joint limits from the URDF, in degrees. The real arm may allow a
# little less; draw_shape.py also checks against your own calibration file.
URDF_LIMITS_DEG = {
    "shoulder_pan": (-110.0, 110.0),
    "shoulder_lift": (-100.0, 100.0),
    "elbow_flex": (-96.8, 96.8),
    "wrist_flex": (-95.0, 95.0),
    "wrist_roll": (-157.2, 162.8),
}


def _transform(xyz, rpy) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    t = np.eye(4)
    t[:3, :3] = [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]
    t[:3, 3] = xyz
    return t


def _rot_z(angle_rad: float) -> np.ndarray:
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    t = np.eye(4)
    t[:2, :2] = [[c, -s], [s, c]]
    return t


_FIXED = [_transform(*o) for o in _JOINT_ORIGINS]
_TIP = _transform(*_TIP_ORIGIN)


def gripper_pose(joints_deg) -> np.ndarray:
    """4x4 pose of the gripper frame (origin at the fingertips, z pointing out of the fingers)."""
    t = np.eye(4)
    for fixed, q in zip(_FIXED, joints_deg):
        t = t @ fixed @ _rot_z(np.radians(q))
    return t @ _TIP


def forward(joints_deg, stickout: float = 0.0, tool_dir=None) -> tuple[np.ndarray, np.ndarray]:
    """Return (tip position, tool direction) for 5 arm joint angles in degrees.

    The pen/chalk tip is `stickout` metres from the fingertips along the tool
    direction. By default that is the direction the gripper points (pen held
    between the jaws); `tool_dir` gives another direction in the gripper frame
    (pen strapped across the gripper).
    """
    t = gripper_pose(joints_deg)
    axis = t[:3, 2] if tool_dir is None else t[:3, :3] @ np.asarray(tool_dir, dtype=float)
    return t[:3, 3] + stickout * axis, axis


def _residual(q4, roll_deg, target, tilt_rad, stickout, tool_dir):
    joints = [q4[0], q4[1], q4[2], q4[3], roll_deg]
    pos, axis = forward(joints, stickout, tool_dir)
    radial = np.array([target[0], target[1], 0.0])
    norm = np.linalg.norm(radial)
    radial = radial / norm if norm > 1e-6 else np.array([1.0, 0.0, 0.0])
    # Position error (m) plus tool direction error (unitless, scaled to be comparable).
    return np.concatenate(
        [
            pos - target,
            0.1 * np.array([axis @ radial - np.sin(tilt_rad), axis[2] + np.cos(tilt_rad)]),
        ]
    )


def inverse(target_xyz, seed_deg, tilt_deg: float = 0.0, stickout: float = 0.0, roll_deg: float = 0.0, tool_dir=None):
    """Find joint angles that put the chalk tip on `target_xyz`.

    tilt_deg is the lean of the chalk away from vertical: 0 = straight down,
    positive = tip further from the base than the wrist (like a pencil pushed
    forward). wrist_roll is held at `roll_deg`.

    Returns (joints_deg[5], position_error_m). Uses damped least squares
    starting from `seed_deg`, so consecutive points stay on the same solution.
    """
    target = np.asarray(target_xyz, dtype=float)
    tilt = np.radians(tilt_deg)
    q = np.array(seed_deg[:4], dtype=float)
    damping = 1e-3
    res = _residual(q, roll_deg, target, tilt, stickout, tool_dir)
    for _ in range(200):
        jac = np.empty((5, 4))
        for i in range(4):
            dq = q.copy()
            dq[i] += 1e-3
            jac[:, i] = (_residual(dq, roll_deg, target, tilt, stickout, tool_dir) - res) / 1e-3
        step = np.linalg.solve(jac.T @ jac + damping * np.eye(4), -jac.T @ res)
        step = np.clip(step, -15.0, 15.0)
        new_res = _residual(q + step, roll_deg, target, tilt, stickout, tool_dir)
        if new_res @ new_res < res @ res:
            q, res, damping = q + step, new_res, max(damping * 0.3, 1e-9)
            if np.abs(step).max() < 1e-5:
                break
        else:
            damping *= 10.0
            if damping > 1e6:
                break
    joints = np.array([q[0], q[1], q[2], q[3], roll_deg])
    return joints, float(np.linalg.norm(res[:3]))


def link_points(joints_deg, stickout: float = 0.0, tool_dir=None) -> np.ndarray:
    """Positions (metres) along the arm for drawing it: base, shoulder pan, shoulder lift,
    elbow, wrist flex, wrist roll, fingertips and (if stickout > 0) the pen tip."""
    pts = [np.zeros(3)]
    t = np.eye(4)
    for fixed, q in zip(_FIXED, joints_deg):
        t = t @ fixed
        pts.append(t[:3, 3].copy())
        t = t @ _rot_z(np.radians(q))
    t = t @ _TIP
    pts.append(t[:3, 3].copy())
    if stickout > 0:
        axis = t[:3, 2] if tool_dir is None else t[:3, :3] @ np.asarray(tool_dir, dtype=float)
        pts.append(t[:3, 3] + stickout * axis)
    return np.array(pts)
