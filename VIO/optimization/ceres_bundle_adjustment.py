"""
ceres_bundle_adjustment.py

Drop-in replacement for optimization.bundle_adjustment.BundleAdjuster that
calls the compiled Ceres solver (optimization/ceres_ba/) instead of
scipy.optimize.least_squares.

Same public interface as BundleAdjuster:
    optimize(max_iterations=100)
    fix_pose(view_id)
    unfix_pose(view_id)
    clear_fixed_poses()

so call sites (vio_core.py) don't need to change beyond the import.
"""

import cv2
import numpy as np

from optimization.ceres_ba import ceres_ba

# Minimum acceptable depth (meters, along the camera's +Z / viewing axis)
# for a landmark observation to be handed to the solver.
#
# ceres_ba.cpp's ReprojectionError clamps z to >= 1e-6 instead of rejecting
# the residual outright ("Ceres has no branchless reject"), on the stated
# assumption that "bad-depth points should be filtered out in Python before
# building the problem". That filtering previously only happened once, at
# triangulation time (vio_core/triangulate.py) -- nothing re-checked depth
# against the CURRENT pose/landmark estimate on every later optimize() call.
# A landmark that is still nominally "triangulated" can easily end up
# behind or nearly in the camera plane of some view after a few LM steps
# (especially with triangulate.py's permissive MIN_TRIANGULATION_ANGLE),
# and once z is clamped near 1e-6 the 1/z reprojection term explodes into
# an enormous, badly-scaled residual AND Jacobian -- that's what was
# driving the multi-order-of-magnitude cost spikes, deeply negative
# tr_ratio steps, and "CHOLMOD: Matrix not positive definite" failures
# seen throughout the run. Skipping such observations here, right before
# packing, is the actual "filter in Python" step the C++ comment assumes
# exists.
MIN_PROJECTION_DEPTH = 1e-3


class CeresResult:
    """
    Minimal stand-in for scipy's OptimizeResult, carrying just the
    fields the rest of the codebase inspects/logs.
    """

    def __init__(self, success, cost, nfev, message):
        self.success = success
        self.cost = cost
        self.nfev = nfev
        self.message = message


class CeresBundleAdjuster:

    def __init__(self, factor_graph, num_threads=4, huber_delta=1.0,
                 max_solver_time_in_seconds=None):

        self.graph = factor_graph
        self.fixed_pose_ids = set()
        # VINS-Mono GlobalSFM-style minimal gauge fix: poses here only have
        # their translation (camera center) held constant, rotation is free.
        self.fixed_translation_only_pose_ids = set()
        self.num_threads = num_threads
        self.huber_delta = huber_delta
        # Bounded one-shot solver budget, e.g. 0.2s for the vision-only
        # init/window BA (VINS-Mono's max_solver_time_in_seconds). None/<=0
        # means unbounded (run to max_iterations/convergence).
        self.max_solver_time_in_seconds = max_solver_time_in_seconds

    # ------------------------------------------------------------ #
    # Fixed-pose bookkeeping (same as BundleAdjuster)
    # ------------------------------------------------------------ #

    def fix_pose(self, view_id):
        """Fully fix a pose (rotation + translation)."""
        self.fixed_pose_ids.add(view_id)
        self.fixed_translation_only_pose_ids.discard(view_id)

    def fix_pose_translation(self, view_id):
        """
        Fix only the translation (camera center) of a pose, leaving its
        rotation free. This is what VINS-Mono's GlobalSFM::construct() does
        to the newest frame in the window: combined with fully fixing the
        reference frame l, it removes exactly the 7 unobservable DOF
        (6 gauge + 1 scale) of vision-only SfM without over-constraining
        every other pose in the window.
        """
        if view_id in self.fixed_pose_ids:
            # already fully fixed — translation-only would be redundant
            return
        self.fixed_translation_only_pose_ids.add(view_id)

    def unfix_pose(self, view_id):
        self.fixed_pose_ids.discard(view_id)
        self.fixed_translation_only_pose_ids.discard(view_id)

    def clear_fixed_poses(self):
        self.fixed_pose_ids.clear()
        self.fixed_translation_only_pose_ids.clear()

    # ------------------------------------------------------------ #
    # Pack / unpack helpers
    # ------------------------------------------------------------ #

    def _pack_poses(self):
        """
        view_id -> [rx, ry, rz, cx, cy, cz]

        R is camera-to-world (Rodrigues -> angle-axis).
        t stored on the pose node is the camera center in world coords
        (see camera_factor.py: pc = R.T @ (xyz - t)).
        """
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

    def _pack_points(self):
        return {
            point_id: [float(xyz[0]), float(xyz[1]), float(xyz[2])]
            for point_id, xyz in self.graph.landmark_nodes.items()
        }

    def _pack_observations(self):
        observations = []
        skipped_depth = 0

        for factor in self.graph.camera_factors:

            if factor.view_id not in self.graph.pose_nodes:
                continue
            if factor.point_id not in self.graph.landmark_nodes:
                continue

            pose = self.graph.pose_nodes[factor.view_id]
            R, C = pose["R"], pose["t"]
            xyz = self.graph.landmark_nodes[factor.point_id]

            # Re-check cheirality against the CURRENT estimate (see
            # MIN_PROJECTION_DEPTH above) -- not just at triangulation
            # time.
            z = float((R.T @ (xyz - C))[2])
            if z <= MIN_PROJECTION_DEPTH:
                skipped_depth += 1
                continue

            L = factor.sqrt_information

            obs = ceres_ba.Observation()
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
                f"[BA] Skipped {skipped_depth} observation(s) with "
                f"near/behind-camera depth (<= {MIN_PROJECTION_DEPTH} m) "
                f"before solving."
            )

        return observations

    def _unpack(self, result):
        """
        Write Ceres' refined poses/points back into the factor graph.
        """
        for view_id, vals in result["poses"].items():

            rvec = np.array(vals[0:3], dtype=np.float64)
            C = np.array(vals[3:6], dtype=np.float64)

            R, _ = cv2.Rodrigues(rvec)

            self.graph.update_pose(view_id, R, C)

        for point_id, vals in result["points"].items():
            xyz = np.array(vals, dtype=np.float64)
            self.graph.update_landmark(point_id, xyz)

    # ------------------------------------------------------------ #
    # Main entry point
    # ------------------------------------------------------------ #

    def optimize(self, max_iterations=100, verbose=True):

        poses = self._pack_poses()
        points = self._pack_points()
        observations = self._pack_observations()

        if len(poses) == 0 or len(points) == 0 or len(observations) == 0:
            print("Nothing to optimize.")
            return None

        K = self.graph.K
        K_vec = [float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])]

        fixed_ids = [int(v) for v in self.fixed_pose_ids if v in poses]
        fixed_translation_only_ids = [
            int(v) for v in self.fixed_translation_only_pose_ids if v in poses
        ]
        solver_time_cap = self.max_solver_time_in_seconds or -1.0

        print("\n========== BUNDLE ADJUSTMENT (Ceres) ==========")
        print(f"Poses         : {len(poses)}  (fixed: {len(fixed_ids)}, "
              f"translation-only fixed: {len(fixed_translation_only_ids)})")
        print(f"Landmarks     : {len(points)}")
        print(f"Observations  : {len(observations)}")
        if solver_time_cap > 0:
            print(f"Solver cap    : {solver_time_cap:.3f}s")

        result = ceres_ba.solve_bundle_adjustment(
            poses=poses,
            points=points,
            observations=observations,
            K_vec=K_vec,
            fixed_pose_ids=fixed_ids,
            max_iterations=max_iterations,
            verbose=verbose,
            huber_delta=self.huber_delta,
            num_threads=self.num_threads,
            fixed_translation_only_pose_ids=fixed_translation_only_ids,
            max_solver_time_in_seconds=solver_time_cap,
        )

        print(f"Cost          : {result['initial_cost']:.4f} -> {result['final_cost']:.4f}")
        print(f"Iterations    : {result['iterations']}")
        print(f"Termination   : {result['termination']}")
        print("================================================\n")

        if not result["success"]:
            print(f"[BA] Ceres did not report a usable solution: {result['message']}")
            return None

        self._unpack(result)

        return CeresResult(
            success=True,
            cost=result["final_cost"],
            nfev=result["iterations"],
            message=result["message"],
        )
