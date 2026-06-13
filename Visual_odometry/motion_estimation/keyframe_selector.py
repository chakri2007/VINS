import numpy as np
from typing import Optional


class KeyframeSelector:

    def __init__(
        self,
        min_parallax_px: float = 30.0,      # median pixel displacement
        min_rotation_deg: float = 3.0,       # rotation angle from last KF
        min_frames_gap: int = 3,             # minimum frames between KFs
        min_inliers: int = 50,               # minimum RANSAC inliers
    ):
        self.min_parallax_px  = min_parallax_px
        self.min_rotation_deg = min_rotation_deg
        self.min_frames_gap   = min_frames_gap
        self.min_inliers      = min_inliers

        # Pose chain — world frame = frame 0 camera frame
        self._world_R = np.eye(3,     dtype=np.float64)  # R of last keyframe
        self._world_t = np.zeros((3, 1), dtype=np.float64)  # t of last keyframe

        self._last_kf_frame_idx: Optional[int] = None
        self._frame_counter = 0   # incremented every process() call

        # Public: list of accepted keyframe records
        self.keyframes = []

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def process(
        self,
        frame_idx: int,
        ransac_result: Optional[dict],
    ) -> bool:
        self._frame_counter += 1

        if ransac_result is None:
            return False

        R_rel = ransac_result['R']          # relative rotation  (3,3)
        t_rel = ransac_result['t']          # relative translation (3,1) unit
        inlier_ids = ransac_result['inlier_ids']
        pts_b      = ransac_result['pts_b'] # undistorted inlier pts in curr frame

        # ── Rule 1: minimum frame gap ──────────────────────────────────
        if self._last_kf_frame_idx is not None:
            gap = frame_idx - self._last_kf_frame_idx
            if gap < self.min_frames_gap:
                return False

        # ── Rule 2: minimum inliers ────────────────────────────────────
        if len(inlier_ids) < self.min_inliers:
            return False

        # ── Rule 3: minimum parallax ───────────────────────────────────
        pts_a = ransac_result['pts_a']
        parallax = self._median_parallax(pts_a, pts_b)
        if parallax < self.min_parallax_px:
            return False

        # ── Rule 4: minimum rotation ───────────────────────────────────
        rot_deg = self._rotation_angle_deg(R_rel)
        if rot_deg < self.min_rotation_deg:
            # Allow through if parallax is very large (fast sideways motion)
            if parallax < self.min_parallax_px * 2.0:
                return False

        # ── Accepted — update pose chain ───────────────────────────────
        # Chain: R_world_curr = R_rel @ R_world_prev
        #        t_world_curr = R_rel @ t_world_prev + t_rel
        new_R = R_rel @ self._world_R
        new_t = R_rel @ self._world_t + t_rel

        self._world_R = new_R
        self._world_t = new_t
        self._last_kf_frame_idx = frame_idx

        record = {
            'frame_idx' : frame_idx,
            'R'         : new_R.copy(),
            't'         : new_t.copy(),
            'pts'       : pts_b.copy(),          # undistorted, inliers only
            'feat_ids'  : inlier_ids.copy(),
        }
        self.keyframes.append(record)

        print(f"[KeyframeSelector] KF accepted: frame={frame_idx} "
              f"parallax={parallax:.1f}px rot={rot_deg:.1f}° "
              f"inliers={len(inlier_ids)} total_kfs={len(self.keyframes)}")

        return True

    # ------------------------------------------------------------------
    # Query interface for downstream modules
    # ------------------------------------------------------------------

    def get_last_two_keyframes(self):
        """
        Return the last two keyframe records, or None if fewer than 2.
        Used by Triangulator to get a valid stereo pair.

        Returns
        -------
        (kf_prev, kf_curr) both are keyframe record dicts, or None.
        """
        if len(self.keyframes) < 2:
            return None
        return self.keyframes[-2], self.keyframes[-1]

    def get_all_keyframes(self):
        """Return full keyframe list. Used by IMU aligner later."""
        return list(self.keyframes)

    def num_keyframes(self) -> int:
        return len(self.keyframes)

    @staticmethod
    def _median_parallax(pts_a: np.ndarray, pts_b: np.ndarray) -> float:
        if len(pts_a) == 0:
            return 0.0
        return float(np.median(np.linalg.norm(pts_b - pts_a, axis=1)))

    @staticmethod
    def _rotation_angle_deg(R: np.ndarray) -> float:
        cos_theta = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_theta)))