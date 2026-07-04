from optimization.factor_graph import FactorGraph
from optimization.camera_factor import CameraFactor
import numpy as np

class GraphBuilder:
    """
    Builds the vision-only sliding-window factor graph (Phase 1/2:
    SfM init + camera-only BA during VI alignment).

    IMU factors are intentionally NOT added here. In the MATLAB
    reference, factorIMU is only added to the graph once IMU alignment
    has actually succeeded (isIMUAligned) — adding it earlier, against
    a not-yet-metric, not-yet-gravity-aligned vision-only map, doesn't
    match the reference and previously crashed here (imu_preintegrator
    was being called as a function; IMUPreintegrator has no __call__).
    IMU-factor wiring belongs in Phase 3, built on top of the
    timestamp-indexed IMU buffer in memory_management.sliding_window
    (extract_imu_between), once isVI_aligned is true.
    """

    def __init__(self):

        self.information = np.eye(2)

    def build(
        self,
        view_set,
        sw_state,
        K,
    ):
        """
        Build a fresh vision-only factor graph from the current sliding
        window (pose nodes, landmark nodes, camera factors — no IMU).

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

        return graph