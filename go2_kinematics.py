"""
go2_kinematics.py - [S2R] Go2 leg kinematics for the comparisons in go2_analyze.py, applied the same way to the three
worlds (Genesis, unitree_mujoco, robot) so that an estimator bias cancels out when they are compared. numpy only.

    foot_positions(q)              foot-sphere centres in the base frame [N, 4, 3] (legs FR, FL, RR, RL)
    jacobians(q)                   d p_foot / d q_leg [N, 4, 3, 3]
    foot_forces(q, tau)            ground force on each foot from the joint torques, base frame: F = -J^-T tau
                                   (static: leg inertia ignored, a few N during a fast swing)
    contacts(fz)                   stance flags from the estimated vertical forces, with hysteresis
    leg_odometry(...)              base velocity in the base frame from the stance feet: v = -(J dq + w x p)
    base_height(...)               base height above the ground under the stance feet (flat ground)

Geometry from go2.urdf (Genesis) = go2.xml (unitree_mujoco): hips at (+-0.1934, +-0.0465, 0), thigh joint 0.0955 m out
from the hip joint, thigh and calf 0.213 m, foot sphere radius 0.022 m. Hip axis x, thigh and calf axes y.
"""

import numpy as np

LEGS = ("FR", "FL", "RR", "RL")  # policy / SDK leg order: joints 3 k .. 3 k + 2 belong to leg k
HIP = np.array([[0.1934, -0.0465, 0.0], [0.1934, 0.0465, 0.0], [-0.1934, -0.0465, 0.0], [-0.1934, 0.0465, 0.0]])
THIGH_Y = np.array([-0.0955, 0.0955, -0.0955, 0.0955])
L1 = L2 = 0.213
FOOT_R = 0.022
MASS = 15.0  # [kg] go2.urdf 15.02, go2.xml 15.21


def _legs(q):
    """[N, 12] -> hip, thigh, calf angles [N, 4] each."""
    q = np.asarray(q, dtype=float).reshape(-1, 4, 3)
    return q[..., 0], q[..., 1], q[..., 2]


def foot_positions(q):
    """Foot-sphere centres in the base frame [N, 4, 3]."""
    q0, q1, q2 = _legs(q)
    vx = -L1 * np.sin(q1) - L2 * np.sin(q1 + q2)
    vy = np.broadcast_to(THIGH_Y, q0.shape)
    vz = -L1 * np.cos(q1) - L2 * np.cos(q1 + q2)
    c0, s0 = np.cos(q0), np.sin(q0)
    return np.stack((HIP[:, 0] + vx, HIP[:, 1] + vy * c0 - vz * s0, HIP[:, 2] + vy * s0 + vz * c0), -1)


def jacobians(q):
    """d p_foot / d (hip, thigh, calf) in the base frame [N, 4, 3, 3] (rows x y z, columns joints)."""
    q0, q1, q2 = _legs(q)
    vy = np.broadcast_to(THIGH_Y, q0.shape)
    vz = -L1 * np.cos(q1) - L2 * np.cos(q1 + q2)
    c0, s0 = np.cos(q0), np.sin(q0)
    d1 = np.stack((-L1 * np.cos(q1) - L2 * np.cos(q1 + q2), np.zeros_like(q1), L1 * np.sin(q1) + L2 * np.sin(q1 + q2)), -1)
    d2 = np.stack((-L2 * np.cos(q1 + q2), np.zeros_like(q1), L2 * np.sin(q1 + q2)), -1)

    def rot_x(v):  # Rx(q0) v for v [..., 3]
        return np.stack((v[..., 0], v[..., 1] * c0 - v[..., 2] * s0, v[..., 1] * s0 + v[..., 2] * c0), -1)

    j0 = np.stack((np.zeros_like(q0), -vy * s0 - vz * c0, vy * c0 - vz * s0), -1)
    return np.stack((j0, rot_x(d1), rot_x(d2)), -1)


def foot_forces(q, tau):
    """Ground force ON each foot, base frame [N, 4, 3], from the joint torques (static balance tau = -J^T F)."""
    J = jacobians(q)
    t = np.asarray(tau, dtype=float).reshape(-1, 4, 3, 1)
    return -np.linalg.solve(np.swapaxes(J, -1, -2), t)[..., 0]


def world_vertical(quat):
    """Rows of R(quat) giving the world z axis in the base frame [N, 3] (= -projected gravity)."""
    w, x, y, z = np.moveaxis(np.asarray(quat, dtype=float).reshape(-1, 4), -1, 0)
    return np.stack((2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)), -1)


def rotate(quat, v):
    """R(quat) v, base frame -> world, for rows [N, 4] and [N, 3]."""
    w, x, y, z = np.moveaxis(np.asarray(quat, dtype=float).reshape(-1, 4), -1, 0)
    R = np.stack((np.stack((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)), -1),
                  np.stack((2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)), -1),
                  np.stack((2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)), -1)), -2)
    return np.einsum("nij,nj->ni", R, np.asarray(v, dtype=float).reshape(-1, 3))


def rotate_inv(quat, v):
    """R(quat)^T v, world -> base frame."""
    w, x, y, z = np.moveaxis(np.asarray(quat, dtype=float).reshape(-1, 4), -1, 0)
    return rotate(np.stack((w, -x, -y, -z), -1), v)


def contacts(fz, on=25.0, off=12.0):
    """Stance flags [N, 4] from vertical foot forces [N, 4] [N]: on above `on`, off below `off` (hysteresis).
    Standing, each foot carries about m g / 4 = 37 N; in a trot the two stance feet about 75 N."""
    fz = np.asarray(fz, dtype=float)
    out = np.zeros(fz.shape, dtype=bool)
    state = fz[0] > on
    for i in range(len(fz)):
        state = np.where(state, fz[i] > off, fz[i] > on)
        out[i] = state
    return out


def debounce(flags, n):
    """Stance flags [N, 4] without the runs shorter than n samples: gaps inside a stance are filled first (a foot
    bouncing at touchdown), then short stance blips are removed (a toe scuffing in swing). The first and last run of
    each foot are kept (the window may have cut them). Simulator contact forces chatter at impacts; this makes the
    true contacts and the estimate count the same steps."""
    out = np.array(flags, dtype=bool)
    if n <= 1:
        return out
    for k in range(out.shape[1]):
        s = out[:, k]  # a view: edits go to out
        for value in (False, True):  # short swings first, then short stances
            d = np.diff(np.concatenate(([0], (s == value).astype(np.int8), [0])))
            for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
                if b - a < n and a > 0 and b < len(s):
                    s[a:b] = not value
    return out


def leg_odometry(q, dq, gyro, stance):
    """Base linear velocity in the base frame [N, 3], from the feet in stance assumed not to slip:
    v_base = -(J dq + w x p) averaged over the stance feet (NaN with no foot in stance). Slip makes it overestimate."""
    p = foot_positions(q)
    J = jacobians(q)
    v_foot = np.einsum("nkij,nkj->nki", J, np.asarray(dq, dtype=float).reshape(-1, 4, 3))
    w = np.asarray(gyro, dtype=float)[:, None, :]
    v = -(v_foot + np.cross(np.broadcast_to(w, p.shape), p))
    s = np.asarray(stance, dtype=float)[..., None]
    n = s.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (v * s).sum(1) / n, np.nan)


def base_height(q, quat, stance):
    """Height of the base origin above the ground [N] (flat ground under the stance feet), NaN with no stance foot."""
    p = foot_positions(q)
    up = world_vertical(quat)  # world z in the base frame
    h = -np.einsum("nki,ni->nk", p, up) + FOOT_R  # base above each foot's contact point
    s = np.asarray(stance, dtype=float)
    n = s.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (h * s).sum(1) / n, np.nan)


def foot_heights(q, quat, stance):
    """Height of each foot's contact point above the ground estimated from the stance feet [N, 4]."""
    p = foot_positions(q)
    up = world_vertical(quat)
    z = np.einsum("nki,ni->nk", p, up)  # foot centre along world z, relative to the base
    s = np.asarray(stance, dtype=float)
    n = s.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ground = np.where(n > 0, (z * s).sum(1) / n, np.nan)
    return z - ground[:, None]
