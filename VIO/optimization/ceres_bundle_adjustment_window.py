"""
ceres_bundle_adjustment_window.py

Ceres port of the sliding-window FactorGraph / optimize(fg, ...) step:
the full-window smoothing pass that build()'s and BA_motion's
docstrings both flagged as "next phase" -- jointly refines every
[pose, velocity, bias] in the current sliding window plus every
landmark observed from it, using both the reprojection factors
(build()) and one IMU preintegration factor per consecutive keyframe
pair (GraphBuilder.build_windowed_vio).

Same public-interface convention as CeresBundleAdjuster
(ceres_bundle_adjustment.py) and CeresBundleAdjusterMotion-style usage
(bundle_adjustment_motion in ceres_bundle_adjustment_motion.py):

    optimize(max_iterations=..., verbose=...)
    fix_pose(view_id) / fix_pose_translation(view_id) / unfix_pose(view_id)
    fix_velocity(view_id) / unfix_velocity(view_id)
    fix_bias(view_id) / unfix_bias(view_id)
    clear_fixed_poses() / clear_fixed_velocities() / clear_fixed_biases()

Camera <-> body convention (must match imu/vi_alignment.py and
ceres_bundle_adjustment_motion.py exactly):

    p_body = R_bs @ p_camera + t_bs
    R_wb = R_wc @ R_bs.T
    t_wb = C   - R_wb @ t_bs        (body origin, world)
"""

import cv2
import numpy as np

from optimization.ceres_ba_window import ceres_ba_window

# NOTE: this used to say the C++ cost functor "clamps z to >= 1e-6 and
# divides", describing a since-fixed bug. The actual current behavior
# (ceres_ba_window.cpp's ReprojectionErrorFree, matching ceres_ba.cpp
# and ceres_ba_motion.cpp exactly) is: for z < 1e-3, report a bounded
# zero residual/zero Jacobian for that one observation instead of
# dividing by a clamped near-zero z -- a defense-in-depth backstop for
# any point that drifts behind the camera *during* the solve itself,
# between one LM step and the next (verified empirically in
# tests/test_depth_backstop.py: cost stays bounded and the solve
# converges cleanly even when a point crosses this boundary mid-solve).
# This Python-side filter's job is different: it keeps such points out
# of the packed problem *before* solving at all, so they don't cost a
# wasted residual block/Jacobian evaluation every single iteration.
MIN_PROJECTION_DEPTH = 1e-3


class CeresWindowResult:
    """Minimal result container, analogous to CeresResult / BAMotionResult."""

    def __init__(self, success, cost, iterations, message):
        self.success = success
        self.cost = cost
        self.iterations = iterations
        self.message = message


def _pack_bias(bias_g, bias_a):
    return np.concatenate([
        np.asarray(bias_g, dtype=np.float64).reshape(3),
        np.asarray(bias_a, dtype=np.float64).reshape(3),
    ])


class CeresBundleAdjusterWindow:

    def __init__(
        self,
        factor_graph,
        R_bs,
        t_bs,
        gravity=np.array([0.0, 0.0, -9.81]),
        num_threads=4,
        huber_delta=1.0,
        max_solver_time_in_seconds=None,
    ):

        self.graph = factor_graph
        self.R_bs = np.asarray(R_bs, dtype=np.float64).reshape(3, 3)
        self.t_bs = np.asarray(t_bs, dtype=np.float64).reshape(3)
        self.gravity = np.asarray(gravity, dtype=np.float64).reshape(3)

        self.fixed_pose_ids = set()
        # Same VINS-Mono GlobalSFM-style minimal gauge fix as
        # CeresBundleAdjuster -- translation only, rotation stays free.
        self.fixed_translation_only_pose_ids = set()
        self.fixed_velocity_ids = set()
        self.fixed_bias_ids = set()

        self.num_threads = num_threads
        self.huber_delta = huber_delta
        self.max_solver_time_in_seconds = max_solver_time_in_seconds

    # ------------------------------------------------------------ #
    # Fixed-node bookkeeping
    # ------------------------------------------------------------ #

    def fix_pose(self, view_id):
        """Fully fix a pose (rotation + translation)."""
        self.fixed_pose_ids.add(view_id)
        self.fixed_translation_only_pose_ids.discard(view_id)

    def fix_pose_translation(self, view_id):
        """Fix only the translation (camera center) of a pose."""
        if view_id in self.fixed_pose_ids:
            return
        self.fixed_translation_only_pose_ids.add(view_id)

    def unfix_pose(self, view_id):
        self.fixed_pose_ids.discard(view_id)
        self.fixed_translation_only_pose_ids.discard(view_id)

    def clear_fixed_poses(self):
        self.fixed_pose_ids.clear()
        self.fixed_translation_only_pose_ids.clear()

    def fix_velocity(self, view_id):
        self.fixed_velocity_ids.add(view_id)

    def unfix_velocity(self, view_id):
        self.fixed_velocity_ids.discard(view_id)

    def clear_fixed_velocities(self):
        self.fixed_velocity_ids.clear()

    def fix_bias(self, view_id):
        self.fixed_bias_ids.add(view_id)

    def unfix_bias(self, view_id):
        self.fixed_bias_ids.discard(view_id)

    def clear_fixed_biases(self):
        self.fixed_bias_ids.clear()

    def fix_oldest_pose_only(self):
        """
        Convenience matching vio_core.py's chosen gauge-fixing
        convention for the windowed optimizer (see
        should_run_windowed_optimization / run_windowed_optimization):
        with IMU factors in the graph, scale and roll/pitch are
        already observable, so the only remaining gauge freedom is
        global position + yaw -- fully fixing just the oldest pose in
        the window resolves that, same as the single-pose-fix branch
        CeresBundleAdjuster uses for the vision-only init BA. Nothing
        else (velocities/biases, or any other pose) needs fixing: they
        are all constrained by the IMU factors chained through the
        window.
        """
        self.clear_fixed_poses()
        self.clear_fixed_velocities()
        self.clear_fixed_biases()
        if len(self.graph.pose_nodes) == 0:
            return
        oldest_view_id = min(self.graph.pose_nodes.keys())
        self.fix_pose(oldest_view_id)

    # ------------------------------------------------------------ #
    # Pack / unpack helpers
    # ------------------------------------------------------------ #

    def _pack_poses(self):
        poses = {}
        for view_id, pose in self.graph.pose_nodes.items():
            rvec, _ = cv2.Rodrigues(pose["R"])
            rvec = rvec.flatten()
            C = pose["t"]
            poses[view_id] = [
                float(rvec[0]), float(rvec[1]), float(rvec[2]),
                float(C[0]), float(C[1]), float(C[2]),
            ]
        return poses

    def _pack_velocities(self):
        return {
            view_id: [float(v[0]), float(v[1]), float(v[2])]
            for view_id, v in self.graph.velocity_nodes.items()
        }

    def _pack_biases(self):
        biases = {}
        for view_id, (bias_g, bias_a) in self.graph.bias_nodes.items():
            vec = _pack_bias(bias_g, bias_a)
            biases[view_id] = [float(x) for x in vec]
        return biases

    def _pack_points(self):
        return {
            point_id: [float(xyz[0]), float(xyz[1]), float(xyz[2])]
            for point_id, xyz in self.graph.landmark_nodes.items()
        }

    def _pack_observations(self):
        observations = []
        skipped_depth = 0

        # Landmark-culling bookkeeping (see memory_management.sliding_window.
        # cull_stale_landmarks, called from vio_core.run_windowed_optimization
        # after optimize() returns): sw_state.landmarks never had any
        # removal path, so the exact same chronically-degenerate points
        # (near/behind-camera depth every cycle) kept re-entering this
        # graph and being skipped here forever. Track both outcomes per
        # point_id so the caller can act on them once the solve is done.
        self.last_skipped_point_ids = set()
        self.last_seen_point_ids = set()

        for factor in self.graph.camera_factors:

            if factor.view_id not in self.graph.pose_nodes:
                continue
            if factor.point_id not in self.graph.landmark_nodes:
                continue

            pose = self.graph.pose_nodes[factor.view_id]
            R, C = pose["R"], pose["t"]
            xyz = self.graph.landmark_nodes[factor.point_id]

            z = float((R.T @ (xyz - C))[2])
            if z <= MIN_PROJECTION_DEPTH:
                skipped_depth += 1
                self.last_skipped_point_ids.add(factor.point_id)
                continue

            self.last_seen_point_ids.add(factor.point_id)

            L = factor.sqrt_information

            obs = ceres_ba_window.Observation()
            obs.view_id = int(factor.view_id)
            obs.point_id = int(factor.point_id)
            obs.u = float(factor.measurement[0])
            obs.v = float(factor.measurement[1])
            obs.L00 = float(L[0, 0])
            obs.L01 = float(L[0, 1])
            obs.L10 = float(L[1, 0])
            obs.L11 = float(L[1, 1])

            observations.append(obs)

        if skipped_depth:
            print(
                f"[BA_window] Skipped {skipped_depth} observation(s) with "
                f"near/behind-camera depth (<= {MIN_PROJECTION_DEPTH} m) "
                f"before solving."
            )

        return observations

    def _pack_imu_factors(self):
        imu_factors = []

        for factor in self.graph.imu_factors:

            if factor.from_view not in self.graph.pose_nodes:
                continue
            if factor.to_view not in self.graph.pose_nodes:
                continue

            p = factor.preintegration

            f = ceres_ba_window.IMUFactorData()
            f.from_view = int(factor.from_view)
            f.to_view = int(factor.to_view)
            f.delta_R = [float(x) for x in np.asarray(p.delta_R, dtype=np.float64).reshape(9)]
            f.delta_v = [float(x) for x in np.asarray(p.delta_v, dtype=np.float64).reshape(3)]
            f.delta_p = [float(x) for x in np.asarray(p.delta_p, dtype=np.float64).reshape(3)]
            f.delta_t = float(p.delta_t)
            f.bias_lin = [float(x) for x in _pack_bias(p.bias_g, p.bias_a)]
            f.J_R_bg = [float(x) for x in np.asarray(p.J_R_bg, dtype=np.float64).reshape(9)]
            f.J_v_bg = [float(x) for x in np.asarray(p.J_v_bg, dtype=np.float64).reshape(9)]
            f.J_v_ba = [float(x) for x in np.asarray(p.J_v_ba, dtype=np.float64).reshape(9)]
            f.J_p_bg = [float(x) for x in np.asarray(p.J_p_bg, dtype=np.float64).reshape(9)]
            f.J_p_ba = [float(x) for x in np.asarray(p.J_p_ba, dtype=np.float64).reshape(9)]
            # Same dense 15x15 sqrt-information convention as BA_motion
            # (_imu_sqrt_information in ceres_bundle_adjustment_motion.py)
            # -- IMUFactor.sqrt_information derives it from
            # preintegration.covariance the same way.
            f.sqrt_information = [float(x) for x in factor.sqrt_information.reshape(225)]

            imu_factors.append(f)

        return imu_factors

    def _unpack(self, result):
        for view_id, vals in result["poses"].items():
            rvec = np.array(vals[0:3], dtype=np.float64)
            C = np.array(vals[3:6], dtype=np.float64)
            R, _ = cv2.Rodrigues(rvec)
            self.graph.update_pose(view_id, R, C)

        for view_id, vals in result["velocities"].items():
            self.graph.update_velocity(view_id, np.asarray(vals, dtype=np.float64))

        for view_id, vals in result["biases"].items():
            vec = np.asarray(vals, dtype=np.float64)
            self.graph.update_bias(view_id, vec[0:3].copy(), vec[3:6].copy())

        for point_id, vals in result["points"].items():
            self.graph.update_landmark(point_id, np.asarray(vals, dtype=np.float64))

    # ------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------ #

    def optimize(self, max_iterations=100, verbose=True):

        poses = self._pack_poses()
        velocities = self._pack_velocities()
        biases = self._pack_biases()
        points = self._pack_points()
        observations = self._pack_observations()
        imu_factors = self._pack_imu_factors()

        if len(poses) == 0:
            print("Nothing to optimize.")
            return None

        K = self.graph.K
        K_vec = [float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])]

        R_bs_vec = [float(x) for x in self.R_bs.reshape(9)]
        t_bs_vec = [float(x) for x in self.t_bs.reshape(3)]
        gravity_vec = [float(x) for x in self.gravity.reshape(3)]

        fixed_ids = [int(v) for v in self.fixed_pose_ids if v in poses]
        fixed_translation_only_ids = [
            int(v) for v in self.fixed_translation_only_pose_ids if v in poses
        ]
        fixed_velocity_ids = [int(v) for v in self.fixed_velocity_ids if v in velocities]
        fixed_bias_ids = [int(v) for v in self.fixed_bias_ids if v in biases]

        solver_time_cap = self.max_solver_time_in_seconds or -1.0

        print("\n========== WINDOWED BUNDLE ADJUSTMENT (Ceres) ==========")
        print(f"Poses         : {len(poses)}  (fixed: {len(fixed_ids)}, "
              f"translation-only fixed: {len(fixed_translation_only_ids)})")
        print(f"Velocities    : {len(velocities)}  (fixed: {len(fixed_velocity_ids)})")
        print(f"Biases        : {len(biases)}  (fixed: {len(fixed_bias_ids)})")
        print(f"Landmarks     : {len(points)}")
        print(f"Observations  : {len(observations)}")
        print(f"IMU factors   : {len(imu_factors)}")
        if solver_time_cap > 0:
            print(f"Solver cap    : {solver_time_cap:.3f}s")

        result = ceres_ba_window.solve_bundle_adjustment_window(
            poses=poses,
            velocities=velocities,
            biases=biases,
            points=points,
            observations=observations,
            imu_factors=imu_factors,
            R_bs=R_bs_vec,
            t_bs=t_bs_vec,
            gravity=gravity_vec,
            K_vec=K_vec,
            fixed_pose_ids=fixed_ids,
            fixed_translation_only_pose_ids=fixed_translation_only_ids,
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
        print("==========================================================\n")

        if not result["success"]:
            print(f"[BA_window] Ceres did not report a usable solution: {result['message']}")
            return None

        self._unpack(result)

        return CeresWindowResult(
            success=True,
            cost=result["final_cost"],
            iterations=result["iterations"],
            message=result["message"],
        )
