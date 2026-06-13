import numpy as np
from typing import Optional


class KeyframeSelector:

    def __init__(
        self,
        min_parallax_px: float = 30.0,
        min_rotation_deg: float = 1.0,
        min_frames_gap: int = 3,
        min_inliers: int = 20,
    ):
        self.min_parallax_px = min_parallax_px
        self.min_rotation_deg = min_rotation_deg
        self.min_frames_gap = min_frames_gap
        self.min_inliers = min_inliers

        self._world_R = np.eye(3, dtype=np.float64)
        self._world_t = np.zeros((3, 1), dtype=np.float64)

        self._last_kf_frame_idx: Optional[int] = None
        self._frame_counter = 0
        self.sliding_window_size=10

        # Accumulate frame-to-frame parallax
        self._accumulated_parallax = 0.0

        self.keyframes = []

    def process(
        self,
        frame_idx: int,
        ransac_result: Optional[dict],
    ) -> bool:

        self._frame_counter += 1

        if ransac_result is None:
            #print("RANSAC result is NONE")
            return False

        R_rel = ransac_result['R']
        t_rel = ransac_result['t']
        inlier_ids = ransac_result['inlier_ids']

        pts_a = ransac_result['pts_a']
        pts_b = ransac_result['pts_b']

        # ----------------------------------------------------------
        # Rule 1: minimum frame gap
        # ----------------------------------------------------------

        if self._last_kf_frame_idx is not None:
            gap = frame_idx - self._last_kf_frame_idx

            if gap < self.min_frames_gap:
                #print("Gap between frames is less")
                return False

        # ----------------------------------------------------------
        # Rule 2: minimum inliers
        # ----------------------------------------------------------

        if len(inlier_ids) < self.min_inliers:
            #print("Min inliers are less")
            return False

        # ----------------------------------------------------------
        # Rule 3: accumulated parallax
        # ----------------------------------------------------------

        frame_parallax = self._median_parallax(pts_a, pts_b)

        self._accumulated_parallax += frame_parallax

        parallax = self._accumulated_parallax

        if parallax < self.min_parallax_px:
            #print(f"Parallax is not satisfied {parallax}")
            return False

        # ----------------------------------------------------------
        # Rule 4: minimum rotation
        # ----------------------------------------------------------

        rot_deg = self._rotation_angle_deg(R_rel)

        if rot_deg < self.min_rotation_deg:
            if parallax < self.min_parallax_px * 2.0:
                #print("Min rotation not satisfied")
                return False

        # ----------------------------------------------------------
        # Accepted keyframe
        # ----------------------------------------------------------

        new_R = R_rel @ self._world_R
        new_t = R_rel @ self._world_t + t_rel

        self._world_R = new_R
        self._world_t = new_t

        self._last_kf_frame_idx = frame_idx

        record = {
            'frame_idx': frame_idx,
            'R': new_R.copy(),
            't': new_t.copy(),
            'pts': pts_b.copy(),
            'feat_ids': inlier_ids.copy(),
        }

        self.keyframes.append(record)
        while len(self.keyframes) > self.sliding_window_size :
            self.keyframes.pop(0)

        # Reset accumulator after accepting keyframe
        self._accumulated_parallax = 0.0

        # print(
        #     f"[KeyframeSelector] KF accepted: "
        #     f"frame={frame_idx} "
        #     f"parallax={parallax:.4f} "
        #     f"rot={rot_deg:.2f}deg "
        #     f"inliers={len(inlier_ids)} "
        #     f"total_kfs={len(self.keyframes)}"
        # )

        return True

    def get_last_two_keyframes(self):

        if len(self.keyframes) < 2:
            return None

        return self.keyframes[-2], self.keyframes[-1]

    def get_all_keyframes(self):
        return list(self.keyframes)

    def num_keyframes(self) -> int:
        return len(self.keyframes)

    @staticmethod
    def _median_parallax(
        pts_a: np.ndarray,
        pts_b: np.ndarray
    ) -> float:

        if len(pts_a) == 0:
            return 0.0

        return float(
            np.median(
                np.linalg.norm(
                    pts_b - pts_a,
                    axis=1
                )
            )
        )

    @staticmethod
    def _rotation_angle_deg(R: np.ndarray) -> float:

        cos_theta = np.clip(
            (np.trace(R) - 1.0) / 2.0,
            -1.0,
            1.0
        )

        return float(
            np.degrees(
                np.arccos(cos_theta)
            )
        )