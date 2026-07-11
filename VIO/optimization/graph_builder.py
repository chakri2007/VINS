from optimization.factor_graph import FactorGraph
from optimization.camera_factor import CameraFactor
from optimization.imu_factor import IMUFactor
from memory_management.sliding_window import build_preintegration
import numpy as np

class GraphBuilder:
    """
    Builds the vision-only sliding-window factor graph (Phase 1/2:
    SfM init + camera-only BA during VI alignment).

    IMU factors are intentionally NOT added by this method. In the
    MATLAB reference, factorIMU is only added to the graph once IMU
    alignment has actually succeeded (isIMUAligned) — adding it earlier,
    against a not-yet-metric, not-yet-gravity-aligned vision-only map,
    doesn't match the reference and previously crashed here
    (imu_preintegrator was being called as a function; IMUPreintegrator
    has no __call__). IMU-factor wiring is Phase 3's build_windowed_vio
    below, built on top of the timestamp-indexed IMU buffer in
    memory_management.sliding_window (extract_imu_between /
    build_preintegration), and is only ever called once isVI_aligned
    is true.
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
    ):
        """
        Build the Phase-3 windowed factor graph: everything build()
        builds (pose nodes, landmark nodes, camera factors) PLUS a
        velocity node and a bias node per window view, and one IMU
        factor per consecutive pair of keyframes in the window.

        This is the sliding-window FactorGraph / optimize(fg, ...)
        step the vision-only build()'s docstring flagged as deferred
        to Phase 3 -- IMU factors are only ever added here, once
        isVI_aligned is true and the caller (visual_inertial_
        optimization / should_run_windowed_optimization in vio_core.py)
        has decided this is a frame to run the full window optimizer
        on.

        Bias linearization for each IMU interval uses the "from" view's
        own per-view bias estimate (sw_state.biases), falling back to
        the sliding window's global bootstrap bias for any view that
        doesn't have one yet -- exactly the same convention as
        VisualInertialOdometry._get_bias / _build_imu_preintegrations,
        via the same shared build_preintegration helper (so there is
        one place, not three, that does this).

        Returns
        -------
        FactorGraph, or None if any consecutive window interval is
        missing IMU coverage (mirrors _build_imu_preintegrations'
        behaviour -- the caller should skip this optimization cycle
        and try again once the IMU stream has caught up).
        """

        graph = self.build(view_set, sw_state, K)

        window_ids = list(sw_state.sliding_window_view_ids)

        #
        # ------------------------------------------------------------------
        # Velocity / Bias Nodes
        # ------------------------------------------------------------------
        #

        for view_id in window_ids:

            velocity = sw_state.velocities.get(view_id)
            if velocity is None:
                # No velocity estimate for this view yet -- can't form
                # a complete [pose,vel,bias] node. Same defensive floor
                # visual_inertial_optimization uses.
                return None

            bias_g, bias_a = sw_state.biases.get(
                view_id,
                (sw_state.gyroscope_bias, sw_state.accelerometer_bias),
            )

            graph.add_velocity(view_id, velocity)
            graph.add_bias(view_id, bias_g, bias_a)

        #
        # ------------------------------------------------------------------
        # IMU Factors -- one per consecutive keyframe pair in the window
        # ------------------------------------------------------------------
        #

        for i, j in zip(window_ids[:-1], window_ids[1:]):

            t_i = view_set.get_timestamp(i)
            t_j = view_set.get_timestamp(j)

            bias_g, bias_a = sw_state.biases.get(
                i,
                (sw_state.gyroscope_bias, sw_state.accelerometer_bias),
            )

            preint = build_preintegration(
                sw_state, imu_calib, t_i, t_j, bias_g, bias_a,
            )

            if preint is None:
                # Missing IMU coverage for this interval -- can't build
                # a complete window graph this cycle.
                return None

            graph.add_imu_factor(
                IMUFactor(
                    from_view=i,
                    to_view=j,
                    preintegration=preint,
                )
            )

        return graph