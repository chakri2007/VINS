"""
ceres_bundle_adjustment_window.py

Ceres-backed windowed VIO bundle adjustment: N poses/velocities/biases,
M IMU factors, full camera factors -- the solver for the factor graph
built by GraphBuilder.build_windowed_vio.

Same public shape as CeresBundleAdjuster (ceres_bundle_adjustment.py):

    optimize(max_iterations, verbose)
    fix_pose(view_id) / unfix_pose(view_id) / clear_fixed_poses()

plus the analogous fix_velocity/fix_bias for the new node types, so
vio_core.py's run_window_bundle_adjustment() reads the same way as
VI_alignment()'s vision-only BA call site.
"""

import cv2
import numpy as np

from optimization.ceres_ba_window import ceres_ba_window


class CeresWindowResult:
    """Minimal result container, analogous to CeresResult /
    BAMotionResult in the other two wrapper modules."""

    def __init__(self, success, cost, iterations, message):
        self.success = success
        self.cost = cost
        self.iterations = iterations
        self.message = message


def _imu_sqrt_information(covariance, eps=1e-9):
    """
    Dense 15x15 sqrt-information from a preintegration's error-state
    covariance. Identical convention to
    ceres_bundle_adjustment_motion._imu_sqrt_information -- duplicated
    here (rather than imported) to keep this module independent of
    BA_motion's wrapper, which is single-frame-specific.
    """
    cov = np.asarray(covariance, dtype=np.float64)
    cov = 0.5 * (cov + cov.T) + eps * np.eye(cov.shape[0])
    information = np.linalg.inv(cov)
    return np.linalg.cholesky(information)


def _pose_to_vec(R, C):
    rvec, _ = cv2.Rodrigues(R)
    return np.concatenate([rvec.flatten(), np.asarray(C, dtype=np.float64).reshape(3)])


def _vec_to_pose(vec):
    vec = np.asarray(vec, dtype=np.float64)
    R, _ = cv2.Rodrigues(vec[0:3])
    C = vec[3:6].copy()
    return R, C


def _pack_bias(bias_g, bias_a):
    return np.concatenate([
        np.asarray(bias_g, dtype=np.float64).reshape(3),
        np.asarray(bias_a, dtype=np.float64).reshape(3),
    ])


class CeresWindowBundleAdjuster:

    def __init__(self, factor_graph, R_bs, t_bs, gravity,
                 num_threads=4, huber_delta=1.0,
                 max_solver_time_in_seconds=None):

        self.graph = factor_graph
        self.R_bs = np.asarray(R_bs, dtype=np.float64)
        self.t_bs = np.asarray(t_bs, dtype=np.float64)
        self.gravity = np.asarray(gravity, dtype=np.float64)

        self.fixed_pose_ids = set()
        self.fixed_velocity_ids = set()
        self.fixed_bias_ids = set()

        self.num_threads = num_threads
        self.huber_delta = huber_delta
        self.max_solver_time_in_seconds = max_solver_time_in_seconds

    # ------------------------------------------------------------ #
    # Fixed-node bookkeeping
    # ------------------------------------------------------------ #

    def fix_pose(self, view_id):
        self.fixed_pose_ids.add(view_id)

    def unfix_pose(self, view_id):
        self.fixed_pose_ids.discard(view_id)

    def fix_velocity(self, view_id):
        self.fixed_velocity_ids.add(view_id)

    def unfix_velocity(self, view_id):
        self.fixed_velocity_ids.discard(view_id)

    def fix_bias(self, view_id):
        self.fixed_bias_ids.add(view_id)

    def unfix_bias(self, view_id):
        self.fixed_bias_ids.discard(view_id)

    def fix_node(self, view_id):
        """Convenience: fix pose+velocity+bias together for one view_id
        (used to gauge-fix the oldest window keyframe)."""
        self.fix_pose(view_id)
        self.fix_velocity(view_id)
        self.fix_bias(view_id)

    def clear_fixed_poses(self):
        self.fixed_pose_ids.clear()
        self.fixed_velocity_ids.clear()
        self.fixed_bias_ids.clear()

    # ------------------------------------------------------------ #
    # Pack / unpack helpers
    # ------------------------------------------------------------ #

    def _pack_poses(self):
        poses = {}
        for view_id, pose in self.graph.pose_nodes.items():
            poses[view_id] = list(_pose_to_vec(pose["R"], pose["t"]))
        return poses

    def _pack_velocities(self):
        return {
            view_id: [float(v) for v in vel]
            for view_id, vel in self.graph.velocity_nodes.items()
        }

    def _pack_biases(self):
        return {
            view_id: list(_pack_bias(bias_g, bias_a))
            for view_id, (bias_g, bias_a) in self.graph.bias_nodes.items()
        }

    def _pack_points(self):
        return {
            point_id: [float(x) for x in xyz]
            for point_id, xyz in self.graph.landmark_nodes.items()
        }

    def _pack_observations(self):
        observations = []
        for factor in self.graph.camera_factors:
            if factor.view_id not in self.graph.pose_nodes:
                continue
            if factor.point_id not in self.graph.landmark_nodes:
                continue

            L = factor.sqrt_information

            obs = ceres_ba_window.WindowObservation()
            obs.view_id = int(factor.view_id)
            obs.point_id = int(factor.point_id)
            obs.u = float(factor.measurement[0])
            obs.v = float(factor.measurement[1])
            obs.L00 = float(L[0, 0])
            obs.L01 = float(L[0, 1])
            obs.L10 = float(L[1, 0])
            obs.L11 = float(L[1, 1])
            observations.append(obs)
        return observations

    def _pack_imu_factors(self):
        imu_factors = []
        for factor in self.graph.imu_factors:
            if factor.from_view not in self.graph.velocity_nodes:
                continue
            if factor.to_view not in self.graph.velocity_nodes:
                continue

            p = factor.preintegration

            f = ceres_ba_window.IMUFactorObs()
            f.from_view_id = int(factor.from_view)
            f.to_view_id = int(factor.to_view)
            f.delta_R = [float(x) for x in np.asarray(p.delta_R).reshape(9)]
            f.delta_v = [float(x) for x in np.asarray(p.delta_v).reshape(3)]
            f.delta_p = [float(x) for x in np.asarray(p.delta_p).reshape(3)]
            f.delta_t = float(p.delta_t)
            f.bias_lin = list(_pack_bias(p.bias_g, p.bias_a))
            f.J_R_bg = [float(x) for x in np.asarray(p.J_R_bg).reshape(9)]
            f.J_v_bg = [float(x) for x in np.asarray(p.J_v_bg).reshape(9)]
            f.J_v_ba = [float(x) for x in np.asarray(p.J_v_ba).reshape(9)]
            f.J_p_bg = [float(x) for x in np.asarray(p.J_p_bg).reshape(9)]
            f.J_p_ba = [float(x) for x in np.asarray(p.J_p_ba).reshape(9)]
            f.sqrt_information = [
                float(x) for x in _imu_sqrt_information(p.covariance).reshape(225)
            ]
            imu_factors.append(f)
        return imu_factors

    def _unpack(self, result):
        for view_id, vals in result["poses"].items():
            R, C = _vec_to_pose(np.array(vals, dtype=np.float64))
            self.graph.update_pose(view_id, R, C)

        for view_id, vals in result["velocities"].items():
            self.graph.update_velocity(view_id, np.array(vals, dtype=np.float64))

        for view_id, vals in result["biases"].items():
            vals = np.array(vals, dtype=np.float64)
            self.graph.update_bias(view_id, vals[0:3].copy(), vals[3:6].copy())

        for point_id, vals in result["points"].items():
            self.graph.update_landmark(point_id, np.array(vals, dtype=np.float64))

    # ------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------ #

    def optimize(self, max_iterations=8, verbose=False):

        poses = self._pack_poses()
        velocities = self._pack_velocities()
        biases = self._pack_biases()
        points = self._pack_points()
        observations = self._pack_observations()
        imu_factors = self._pack_imu_factors()

        if len(poses) == 0:
            print("[Window BA] Nothing to optimize.")
            return None

        K = self.graph.K
        K_vec = [float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])]

        fixed_pose_ids = [int(v) for v in self.fixed_pose_ids if v in poses]
        fixed_velocity_ids = [int(v) for v in self.fixed_velocity_ids if v in velocities]
        fixed_bias_ids = [int(v) for v in self.fixed_bias_ids if v in biases]

        solver_time_cap = self.max_solver_time_in_seconds or -1.0

        print("\n========== WINDOWED VIO BUNDLE ADJUSTMENT (Ceres) ==========")
        print(f"Poses         : {len(poses)}  (fixed: {len(fixed_pose_ids)})")
        print(f"Velocities    : {len(velocities)}  (fixed: {len(fixed_velocity_ids)})")
        print(f"Biases        : {len(biases)}  (fixed: {len(fixed_bias_ids)})")
        print(f"Landmarks     : {len(points)}")
        print(f"Observations  : {len(observations)}")
        print(f"IMU factors   : {len(imu_factors)}")
        if solver_time_cap > 0:
            print(f"Solver cap    : {solver_time_cap:.3f}s")

        result = ceres_ba_window.solve_windowed_bundle_adjustment(
            poses=poses,
            velocities=velocities,
            biases=biases,
            points=points,
            observations=observations,
            imu_factors=imu_factors,
            K_vec=K_vec,
            R_bs=list(self.R_bs.reshape(9)),
            t_bs=list(self.t_bs.reshape(3)),
            gravity=list(self.gravity.reshape(3)),
            fixed_pose_ids=fixed_pose_ids,
            fixed_velocity_ids=fixed_velocity_ids,
            fixed_bias_ids=fixed_bias_ids,
            max_iterations=max_iterations,
            verbose=verbose,
            huber_delta=self.huber_delta,
            num_threads=self.num_threads,
            max_solver_time_in_seconds=solver_time_cap,
        )

        print(f"Cost          : {result['initial_cost']:.4f} -> {result['final_cost']:.4f}")
        print(f"Iterations    : {result['iterations']}")
        print(f"Termination   : {result['termination']}")
        print("==============================================================\n")

        if not result["success"]:
            print(f"[Window BA] Ceres did not report a usable solution: {result['message']}")
            return None

        self._unpack(result)

        return CeresWindowResult(
            success=True,
            cost=result["final_cost"],
            iterations=result["iterations"],
            message=result["message"],
        )
