from optimization.camera_factor import CameraFactor
import numpy as np


class GraphBuilder:

    def __init__(self):

        self.information = np.eye(2)


    def update(
        self,
        factor_graph,
        view_set,
        sw_state,
        current_view_id,
    ):

        #
        # Add new pose node
        #
        if current_view_id not in factor_graph.pose_nodes:

            R, t = view_set.get_pose(current_view_id)

            factor_graph.add_pose(
                current_view_id,
                R,
                t,
            )

        #
        # Add newly triangulated landmarks
        #
        for landmark in sw_state.landmarks.values():

            if landmark.point_id not in factor_graph.landmark_nodes:

                factor_graph.add_landmark(
                    landmark.point_id,
                    landmark.xyz,
                )

        #
        # Add camera observation factors
        #
        ids = sw_state.all_ids[current_view_id][:,1]
        uv = sw_state.all_observations[current_view_id]
        tri = sw_state.all_triangulated[current_view_id]

        for pid, pixel, is_tri in zip(ids, uv, tri):

            if not is_tri:
                continue

            factor_graph.add_camera_factor(

                CameraFactor(
                    view_id=current_view_id,
                    point_id=int(pid),
                    measurement=pixel.astype(np.float64),
                    information=self.information,
                )

            )