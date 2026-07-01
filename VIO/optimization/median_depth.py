import numpy as np


def estimate_median_depth(
    view_set,
    sliding_window_state,
):
    """
    Estimate the median landmark depth.

    Parameters
    ----------
    view_set : ViewSet

    sliding_window_state : SlidingWindowState

    Returns
    -------
    median_depth : float
    """

    depths = []

    #
    # Use the newest camera in the window
    #
    current_view = sliding_window_state.current_view_id

    R, t = view_set.get_pose(current_view)

    #
    # Compute depth of every landmark
    #
    for landmark in sliding_window_state.landmarks.values():

        pc = R.T @ (landmark.xyz - t)

        if pc[2] > 0:
            depths.append(pc[2])

    if len(depths) == 0:
        return None

    return float(np.median(depths))

import numpy as np


def normalize_map(
    view_set,
    sliding_window_state,
):
    """
    Normalize the reconstruction so that the median landmark depth is 1.

    Parameters
    ----------
    median_depth : float

    view_set : ViewSet

    sliding_window_state : SlidingWindowState
    """
    median_depth = estimate_median_depth(
        view_set,
        sliding_window_state,)

    if median_depth is None or median_depth <= 1e-8:
        return

    scale = 1.0 / median_depth

    #
    # Scale camera translations
    #
    for view_id in view_set.view_ids:

        R, t = view_set.get_pose(view_id)

        view_set.update_pose(
            view_id,
            R,
            t * scale,
        )

    #
    # Scale landmarks
    #
    for landmark in sliding_window_state.landmarks.values():

        landmark.xyz *= scale

    print(
        f"[VIO] Map normalized (scale = {scale:.6f})"
    )