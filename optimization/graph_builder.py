from optimization.factor_graph import FactorGraph
from optimization.camera_factor import CameraFactor
import numpy as np
from optimization.imu_factor import IMUFactor

from imu.preintegration import IMUPreintegrator

from memory_management.sliding_window import get_imu_measurements

class GraphBuilder:

    def __init__(self):

        self.information = np.eye(2)
        self.imu_information = np.eye(9)
        self.imu_preintegrator = IMUPreintegrator()

    def build(
        self,
        view_set,
        sw_state,
        K,
    ):
        """
        Build a fresh factor graph from the current sliding window.

        Returns
        -------
        FactorGraph
        """

        graph = FactorGraph(K)

        #
        # ------------------------------------------------------------------
        # Pose Nodes
        # ------------------------------------------------------------------
        #

        for view_id in sw_state.sliding_window_view_ids:

            R, t = view_set.get_pose(view_id)

            graph.add_pose(
                view_id,
                R,
                t,
            )

        #
        # ------------------------------------------------------------------
        # Landmark Nodes
        # ------------------------------------------------------------------
        #

        for landmark in sw_state.landmarks.values():
            if not landmark.is_triangulated:
                continue

            graph.add_landmark(
                landmark.point_id,
                landmark.xyz,
            )

        #
        # ------------------------------------------------------------------
        # Camera Factors
        # ------------------------------------------------------------------
        #

        for landmark in sw_state.landmarks.values():

            for obs in landmark.observations:

                #
                # Ignore observations outside the window
                #
                window_ids = set(sw_state.sliding_window_view_ids)
                if obs.view_id not in window_ids:
                    continue

                graph.add_camera_factor(

                    CameraFactor(
                        view_id=obs.view_id,
                        point_id=landmark.point_id,
                        measurement=obs.uv.copy(),
                        information=self.information,
                    )

                )
        
        #
        # ------------------------------------------------------------------
        # IMU Factors
        # ------------------------------------------------------------------
        #

        window_ids = sw_state.sliding_window_view_ids

        for from_view, to_view in zip(window_ids[:-1], window_ids[1:]):

            measurements = get_imu_measurements(
                sw_state,
                from_view,
                to_view,
            )

            #
            # No IMU data available
            #
            if len(measurements) == 0:
                continue

            preintegration = self.imu_preintegrator(
                measurements,
            )

            graph.add_imu_factor(

                IMUFactor(
                    from_view=from_view,
                    to_view=to_view,
                    preintegration=preintegration,
                    information=self.imu_information,
                )

            )

        return graph