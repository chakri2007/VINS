import numpy as np


def update_state_from_graph(
    factor_graph,
    view_set,
    sliding_window_state,
):
    """
    Copy optimized poses and landmarks from the factor graph back into
    the ViewSet and SlidingWindowState.

    Equivalent to MATLAB:

        helperUpdateCameraPoseTable(...)
        updateView(...)
        setXYZPoints(...)
    """

    #
    # Update camera poses
    #
    for view_id, pose in factor_graph.pose_nodes.items():

        view_set.update_pose(
            view_id,
            pose["R"],
            pose["t"],
        )

    #
    # Update landmarks
    #
    for point_id, xyz in factor_graph.landmark_nodes.items():

        if point_id in sliding_window_state.landmarks:

            sliding_window_state.landmarks[point_id].xyz = xyz.copy()

    print(
        "[BA] Runtime state updated from factor graph."
    )