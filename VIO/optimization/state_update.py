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

    #
    # Update velocities (Phase 3 windowed VIO only -- vision-only Phase
    # 1/2 graphs never populate velocity_nodes, so this is a no-op there)
    #
    for view_id, velocity in factor_graph.velocity_nodes.items():

        sliding_window_state.velocities[view_id] = velocity.copy()

    #
    # Update per-keyframe biases (Phase 3 windowed VIO only)
    #
    for view_id, (bias_g, bias_a) in factor_graph.bias_nodes.items():

        sliding_window_state.biases[view_id] = (bias_g.copy(), bias_a.copy())

    print(
        "[BA] Runtime state updated from factor graph."
    )