import numpy as np


def predict_state(
    R_i,
    t_i,
    v_i,
    gravity,
    preintegration,
):
    """
    Predict state at keyframe j from state at keyframe i.

    Parameters
    ----------
    R_i : (3,3)
        Camera-to-world rotation.

    t_i : (3,)
        Camera position.

    v_i : (3,)
        Velocity in world frame.

    gravity : (3,)

    preintegration : PreintegratedIMU

    Returns
    -------
    R_j
    t_j
    v_j
    """

    dt = preintegration.delta_t

    #
    # Rotation
    #
    R_j = R_i @ preintegration.delta_R

    #
    # Velocity
    #
    v_j = (
        v_i
        + gravity * dt
        + R_i @ preintegration.delta_v
    )

    #
    # Position
    #
    t_j = (
        t_i
        + v_i * dt
        + 0.5 * gravity * dt * dt
        + R_i @ preintegration.delta_p
    )

    return R_j, t_j, v_j