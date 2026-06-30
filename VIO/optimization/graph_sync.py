def sync_graph_to_state(
    factor_graph,
    view_set,
    sliding_window,
):
    """
    Copy optimized values from the factor graph back into
    ViewSet and SlidingWindow.
    """

    #
    # Camera poses
    #
    for view_id, pose in factor_graph.pose_nodes.items():

        view_set.update_pose(
            view_id,
            pose["R"],
            pose["t"],
        )

    #
    # Landmarks
    #
    for point_id, xyz in factor_graph.landmark_nodes.items():

        if point_id in sliding_window.landmarks:

            sliding_window.landmarks[point_id].xyz = xyz.copy()