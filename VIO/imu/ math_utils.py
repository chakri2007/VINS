import numpy as np


def skew(v):
    """
    Skew symmetric matrix.
    """

    x, y, z = v

    return np.array([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0],
    ])


def exp_so3(phi):
    """
    Rodrigues exponential map.

    Parameters
    ----------
    phi : (3,)
        Rotation vector.

    Returns
    -------
    R : (3,3)
    """

    theta = np.linalg.norm(phi)

    if theta < 1e-12:
        return np.eye(3)

    axis = phi / theta

    K = skew(axis)

    return (
        np.eye(3)
        + np.sin(theta) * K
        + (1 - np.cos(theta)) * K @ K
    )