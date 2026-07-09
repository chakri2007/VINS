from optimization.factor_graph import FactorGraph
from optimization.camera_factor import CameraFactor
from optimization.imu_factor import IMUFactor
from imu.preintegration_utils import preintegrate_between
from memory_management.sliding_window import get_bias
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

    def build_windowed_vio(
        self,
        view_set,
        sw_state,
        K,
        imu_calib,
        imu_information=None,
    ):
        """
        Build the Phase 3 windowed VIO factor graph: poses + velocities
        + biases + camera factors + IMU factors over the current
        sliding window.

        Unlike `build()` (vision-only, Phase 1/2), every pose node here
        also gets a paired velocity node and bias node, and consecutive
        keyframes in the window are linked by an IMUFactor built from
        the same preintegration helper used by
        VisualInertialOdometry._build_imu_preintegrations /
        _build_single_imu_preintegration (see
        imu/preintegration_utils.py) -- there is exactly one place that
        constructs an IMUPreintegrator and slices the timestamp-ordered
        buffer, this method just calls it once per consecutive pair.

        Parameters
        ----------
        view_set, sw_state, K : as in build()
        imu_calib : dict
            IMU noise-density calibration (VisualInertialOdometry.imu_calib),
            forwarded to preintegrate_between.
        imu_information : (15,15) ndarray, optional
            Base information matrix for IMU factors when the
            preintegration's own covariance can't be used (unused by
            default -- IMUFactor.sqrt_information is derived from each
            preintegration's own covariance, see
            ceres_bundle_adjustment_window.py). Kept as a parameter for
            symmetry with `information` above / future overrides.

        Returns
        -------
        FactorGraph, with pose_nodes/velocity_nodes/bias_nodes/
        camera_factors/imu_factors populated over the window. A
        keyframe pair with insufficient IMU coverage simply doesn't get
        an IMUFactor (mirrors the rest of this codebase's
        insufficient-coverage handling -- it's a gap, not a hard
        failure of the whole window).
        """

        graph = self.build(view_set, sw_state, K)

        window_ids = list(sw_state.sliding_window_view_ids)

        #
        # ------------------------------------------------------------------
        # Velocity / bias nodes
        # ------------------------------------------------------------------
        #

        for view_id in window_ids:

            velocity = sw_state.velocities.get(view_id)
            if velocity is None:
                # No velocity estimate yet for this view (e.g. it was
                # never touched by BA_motion, only by vision-only
                # phases) -- can't attach a velocity/bias node, so it
                # also can't anchor an IMU factor. Skip; the pose node
                # from build() above still lets it participate in
                # camera factors.
                continue

            bias_g, bias_a = get_bias(sw_state, view_id)

            graph.add_velocity(view_id, velocity)
            graph.add_bias(view_id, bias_g, bias_a)

        #
        # ------------------------------------------------------------------
        # IMU factors between consecutive window keyframes
        # ------------------------------------------------------------------
        #

        for from_id, to_id in zip(window_ids[:-1], window_ids[1:]):

            if from_id not in graph.velocity_nodes or to_id not in graph.velocity_nodes:
                continue

            t_from = view_set.get_timestamp(from_id)
            t_to = view_set.get_timestamp(to_id)

            bias_g, bias_a = get_bias(sw_state, from_id)

            preint = preintegrate_between(
                sw_state, imu_calib, t_from, t_to, bias_g, bias_a,
            )

            if preint is None:
                continue

            graph.add_imu_factor(
                IMUFactor(
                    from_view=from_id,
                    to_view=to_id,
                    preintegration=preint,
                    information=np.eye(15),
                )
            )

        return graph