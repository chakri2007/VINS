# Reads from:  LandmarkMap, KeyframeSelector
# Writes to:   LandmarkMap (refined 3D positions)
#              KeyframeSelector (refined poses)

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional

# Import the compiled Ceres module.
# If import fails, BA is disabled gracefully — pipeline continues without it.
try:
    from motion_estimation.bundle_adjustment import ba_solver
    CERES_AVAILABLE = True
except ImportError:
    print("[BA] Warning: ba_solver not compiled. BA disabled.")
    CERES_AVAILABLE = False


def _rvec_from_R(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to angle-axis (3,) vector."""
    rvec, _ = cv2.Rodrigues(R.astype(np.float64))
    return rvec.flatten()


def _R_from_rvec(rvec: np.ndarray) -> np.ndarray:
    """Convert angle-axis (3,) to 3x3 rotation matrix."""
    R, _ = cv2.Rodrigues(rvec.reshape(3, 1).astype(np.float64))
    return R


class BundleAdjustment:
    def __init__(
        self,
        K: np.ndarray,
        max_iterations: int = 50,
        min_landmarks: int = 10,
        min_keyframes: int = 3,
        verbose: bool = False,
    ):
        self.K              = K.astype(np.float64)
        self.fx = K[0, 0]; self.fy = K[1, 1]
        self.cx = K[0, 2]; self.cy = K[1, 2]
        self.max_iterations = max_iterations
        self.min_landmarks  = min_landmarks
        self.min_keyframes  = min_keyframes
        self.verbose        = verbose

        # Tracks how many times BA has run — for logging
        self._run_count = 0

    def run(
        self,
        landmark_map,       # your LandmarkMap instance
        keyframe_selector,  # your KeyframeSelector instance
    ) -> bool:
        if not CERES_AVAILABLE:
            return False

        # ── Step 1: Pull data from your existing structures ────────────────
        keyframes    = keyframe_selector.get_all_keyframes()
        landmarks    = landmark_map.get_landmarks_for_ba()
        observations = landmark_map.get_observations_for_ba()

        if len(keyframes) < self.min_keyframes:
            return False
        if len(landmarks) < self.min_landmarks:
            return False

        # ── Step 2: Build index maps ───────────────────────────────────────
        # frame_idx → position in keyframes list (cam_idx for Ceres)
        frame_to_cam = {
            kf['frame_idx']: i for i, kf in enumerate(keyframes)
        }

        # lm_id → position in points list (pt_idx for Ceres)
        lm_ids    = list(landmarks.keys())
        lm_to_idx = {lm_id: i for i, lm_id in enumerate(lm_ids)}

        # ── Step 3: Pack poses into Ceres format ───────────────────────────
        # Ceres wants: [rx, ry, rz, tx, ty, tz] per camera
        # Your KeyframeSelector stores: {'R': (3,3), 't': (3,1), ...}
        ceres_poses = []
        for kf in keyframes:
            rvec = _rvec_from_R(kf['R'])           # (3,)
            tvec = kf['t'].flatten()                # (3,)
            ceres_poses.append([
                float(rvec[0]), float(rvec[1]), float(rvec[2]),
                float(tvec[0]), float(tvec[1]), float(tvec[2]),
            ])

        # ── Step 4: Pack 3D points into Ceres format ──────────────────────
        ceres_points = []
        for lm_id in lm_ids:
            X = landmarks[lm_id]
            ceres_points.append([float(X[0]), float(X[1]), float(X[2])])

        # ── Step 5: Build observations list ───────────────────────────────
        # Ceres format: (cam_idx, pt_idx, u_obs, v_obs)
        # Your LandmarkMap stores: {lm_id: [(frame_idx, u, v), ...]}
        ceres_obs = []
        for lm_id, obs_list in observations.items():
            pt_idx = lm_to_idx.get(lm_id)
            if pt_idx is None:
                continue

            for (frame_idx, u, v) in obs_list:
                cam_idx = frame_to_cam.get(frame_idx)
                if cam_idx is None:
                    continue  # observation is outside current window — skip

                ceres_obs.append((
                    int(cam_idx),
                    int(pt_idx),
                    float(u),
                    float(v),
                ))

        if len(ceres_obs) < self.min_landmarks * 2:
            if self.verbose:
                print(f"[BA] Too few observations ({len(ceres_obs)}), skipping.")
            return False

        if self.verbose:
            print(f"[BA] Running: {len(keyframes)} cams, "
                  f"{len(lm_ids)} points, {len(ceres_obs)} obs")

        # ── Step 6: Call Ceres ─────────────────────────────────────────────
        result = ba_solver.solve_bundle_adjustment(
            poses        = ceres_poses,
            points       = ceres_points,
            observations = ceres_obs,
            K_vec        = [self.fx, self.fy, self.cx, self.cy],
            fix_first    = True,        # anchor first keyframe
            max_iters    = self.max_iterations,
        )

        if not result['success']:
            if self.verbose:
                print(f"[BA] Solver did not converge: {result['message']}")
            # Still accept the result — NO_CONVERGENCE just means
            # it hit max_iters, the estimates are still improved.

        self._run_count += 1

        if self.verbose:
            print(f"[BA] Run #{self._run_count}: "
                  f"cost {result['initial_cost']:.2f} → {result['final_cost']:.2f} "
                  f"({result['iterations']} iters)")

        # ── Step 7: Write refined poses back to KeyframeSelector ──────────
        refined_poses = result['poses']
        for i, kf in enumerate(keyframe_selector.keyframes):
            pose = refined_poses[i]
            rvec = np.array(pose[:3], dtype=np.float64)
            tvec = np.array(pose[3:], dtype=np.float64)
            kf['R'] = _R_from_rvec(rvec)
            kf['t'] = tvec.reshape(3, 1)

        # Also update KeyframeSelector's world pose to last keyframe
        if keyframe_selector.keyframes:
            last_kf = keyframe_selector.keyframes[-1]
            keyframe_selector._world_R = last_kf['R'].copy()
            keyframe_selector._world_t = last_kf['t'].copy()

        # ── Step 8: Write refined 3D points back to LandmarkMap ───────────
        refined_points = result['points']
        for i, lm_id in enumerate(lm_ids):
            if lm_id in landmark_map.landmarks:
                landmark_map.landmarks[lm_id] = np.array(
                    refined_points[i], dtype=np.float64
                )

        return True