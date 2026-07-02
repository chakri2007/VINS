import numpy as np


def estimate_initial_velocities(
    view_set,
):
    """
    Estimate camera velocities from consecutive optimized poses.

    Parameters
    ----------
    view_set : ViewSet

    Returns
    -------
    velocities : dict

        Dictionary mapping

            view_id -> (3,) velocity

        expressed in the world frame.
    """

    velocities = {}

    view_ids = view_set.view_ids

    #
    # Need at least two poses
    #
    if len(view_ids) < 2:
        return velocities

    #
    # Forward finite difference
    #
    for i in range(len(view_ids) - 1):

        view_id_1 = view_ids[i]
        view_id_2 = view_ids[i + 1]

        #
        # Camera centers
        #
        _, C1 = view_set.get_pose(view_id_1)
        _, C2 = view_set.get_pose(view_id_2)

        #
        # Timestamps
        #
        t1 = view_set.get_timestamp(view_id_1)
        t2 = view_set.get_timestamp(view_id_2)

        dt = t2 - t1

        #
        # Invalid timestamps
        #
        if dt <= 0:
            continue

        #
        # World-frame velocity
        #
        velocity = (C2 - C1) / dt

        velocities[view_id_1] = velocity

    #
    # Last frame
    #
    last_view = view_ids[-1]

    if len(view_ids) >= 2:

        previous_view = view_ids[-2]

        if previous_view in velocities:

            velocities[last_view] = velocities[previous_view].copy()

    return velocities