"""Quaternion helpers (wxyz convention, matches Unitree / IsaacLab).

Port of math_utils.hpp. Vectorised with NumPy.
"""

import numpy as np


def quat_unit(q: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(q)
    return q / n if n > 1e-12 else np.array([1.0, 0.0, 0.0, 0.0])


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by quaternion q (wxyz).

    Same formula as quat_rotate_d in math_utils.hpp, so a zero quaternion (a zero-padded
    history entry) gives -v as it does there.
    """
    w = q[0]
    q_vec = np.asarray(q[1:4], dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    return v * (2.0 * w * w - 1.0) + np.cross(q_vec, v) * w * 2.0 + q_vec * np.dot(q_vec, v) * 2.0


def gravity_orientation(q: np.ndarray) -> np.ndarray:
    """Body-frame projection of [0,0,-1] (matches math_utils.hpp:94-107)."""
    w, x, y, z = q
    return np.array(
        [
            2.0 * (-z * x + w * y),
            -2.0 * (z * y + w * x),
            1.0 - 2.0 * (w * w + z * z),
        ]
    )


def calc_heading(q: np.ndarray) -> float:
    fwd = quat_rotate(q, np.array([1.0, 0.0, 0.0]))
    return float(np.arctan2(fwd[1], fwd[0]))


def euler_z_to_quat(yaw: float) -> np.ndarray:
    half = 0.5 * yaw
    return np.array([np.cos(half), 0.0, 0.0, np.sin(half)])


def quat_slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        return quat_unit(q0 + t * (q1 - q0))
    theta_0 = np.arccos(dot)
    sin_theta_0 = np.sin(theta_0)
    theta = theta_0 * t
    s0 = np.cos(theta) - dot * np.sin(theta) / sin_theta_0
    s1 = np.sin(theta) / sin_theta_0
    return s0 * q0 + s1 * q1


def quat_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    xx = x * x
    yy = y * y
    zz = z * z
    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (xx + zz), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (xx + yy)],
        ]
    )


def quat_to_6d_rowwise(q: np.ndarray) -> np.ndarray:
    """First 2 columns of the rotation matrix, ROW-WISE flattened.

    Matches the C++ GatherMotionAnchorOrientation packing
    (g1_deploy_onnx_ref.cpp:679-683):
        [R[0][0], R[0][1], R[1][0], R[1][1], R[2][0], R[2][1]]
    Python's scipy `as_matrix()[..., :2].reshape(1, -1)` is row-major.
    """
    R = quat_to_rotation_matrix(q)
    return np.array([R[0, 0], R[0, 1], R[1, 0], R[1, 1], R[2, 0], R[2, 1]])


def calc_heading_quat_inv(q: np.ndarray) -> np.ndarray:
    """Inverse of the yaw-only heading quaternion (negative yaw)."""
    return euler_z_to_quat(-calc_heading(q))


def calc_heading_quat(q: np.ndarray) -> np.ndarray:
    """Yaw-only heading quaternion."""
    return euler_z_to_quat(calc_heading(q))
