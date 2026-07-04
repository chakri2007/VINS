"""
math_utils.py

SO(3) Lie group utilities used throughout the VIO pipeline.

Implements:

    • skew()
    • vee()
    • exp_so3()
    • log_so3()

    • left_jacobian()
    • right_jacobian()

    • inverse_left_jacobian()
    • inverse_right_jacobian()

    • normalize_rotation()

References
----------
Forster et al.
IMU Preintegration on Manifold

Barfoot
State Estimation for Robotics
"""

import numpy as np

EPS = 1e-12


# ---------------------------------------------------------------------
# Basic operators
# ---------------------------------------------------------------------

def skew(v):
    """
    Convert a 3-vector into a skew-symmetric matrix.

    Parameters
    ----------
    v : (3,)

    Returns
    -------
    (3,3)
    """

    x, y, z = v

    return np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ]
    )


def vee(S):
    """
    Inverse of skew().

    Parameters
    ----------
    S : (3,3)

    Returns
    -------
    (3,)
    """

    return np.array(
        [
            S[2, 1],
            S[0, 2],
            S[1, 0],
        ]
    )


# ---------------------------------------------------------------------
# Rotation normalization
# ---------------------------------------------------------------------

def normalize_rotation(R):
    """
    Project a matrix onto SO(3)
    using SVD.

    Removes numerical drift.
    """

    U, _, Vt = np.linalg.svd(R)

    R = U @ Vt

    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt

    return R


# ---------------------------------------------------------------------
# Exponential map
# ---------------------------------------------------------------------

def exp_so3(phi):
    """
    Rodrigues exponential map.

    Parameters
    ----------
    phi : (3,)

    Returns
    -------
    R : (3,3)
    """

    theta = np.linalg.norm(phi)

    if theta < EPS:
        return np.eye(3) + skew(phi)

    A = np.sin(theta) / theta
    B = (1.0 - np.cos(theta)) / (theta ** 2)

    W = skew(phi)

    return (
        np.eye(3)
        + A * W
        + B * (W @ W)
    )


# ---------------------------------------------------------------------
# Log map
# ---------------------------------------------------------------------

def log_so3(R):
    """
    SO(3) logarithm.

    Returns

        rotation vector
    """

    R = normalize_rotation(R)

    cos_theta = (np.trace(R) - 1.0) * 0.5

    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    theta = np.arccos(cos_theta)

    if theta < EPS:
        return vee(R - R.T) * 0.5

    return (
        theta
        / (2.0 * np.sin(theta))
        * vee(R - R.T)
    )


# ---------------------------------------------------------------------
# Left Jacobian
# ---------------------------------------------------------------------

def left_jacobian(phi):
    """
    Left Jacobian of SO(3).
    """

    theta = np.linalg.norm(phi)

    W = skew(phi)

    if theta < EPS:
        return (
            np.eye(3)
            + 0.5 * W
            + (1.0 / 6.0) * W @ W
        )

    A = (1 - np.cos(theta)) / (theta ** 2)

    B = (theta - np.sin(theta)) / (theta ** 3)

    return (
        np.eye(3)
        + A * W
        + B * W @ W
    )


# ---------------------------------------------------------------------
# Right Jacobian
# ---------------------------------------------------------------------

def right_jacobian(phi):
    """
    Right Jacobian.
    """

    return left_jacobian(-phi)


# ---------------------------------------------------------------------
# Inverse Left Jacobian
# ---------------------------------------------------------------------

def inverse_left_jacobian(phi):
    """
    Inverse Left Jacobian.
    """

    theta = np.linalg.norm(phi)

    W = skew(phi)

    if theta < EPS:
        return (
            np.eye(3)
            - 0.5 * W
            + (1.0 / 12.0) * W @ W
        )

    half = 0.5

    cot = (
        1.0 / theta ** 2
        - (1 + np.cos(theta))
        / (2 * theta * np.sin(theta))
    )

    return (
        np.eye(3)
        - half * W
        + cot * W @ W
    )


# ---------------------------------------------------------------------
# Inverse Right Jacobian
# ---------------------------------------------------------------------

def inverse_right_jacobian(phi):
    """
    Inverse Right Jacobian.
    """

    return inverse_left_jacobian(-phi)