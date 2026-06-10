# motion_estimation/ransac.py
import numpy as np
import cv2


class RANSACMotionEstimator:
    def __init__(self, K: np.ndarray, dist_coeffs: np.ndarray,
                 ransac_prob: float = 0.999,
                 ransac_threshold: float = 1.0,
                 min_inlier_ratio: float = 0.5,
                 min_points: int = 8):
        self.K               = K
        self.dist_coeffs     = dist_coeffs
        self.ransac_prob      = ransac_prob
        self.ransac_threshold = ransac_threshold
        self.min_inlier_ratio = min_inlier_ratio
        self.min_points       = min_points

    def estimate(self, histories: dict) -> dict | None:
        prev_pts, curr_pts, valid_ids = self._extract_correspondences(histories)

        if prev_pts is None or len(prev_pts) < self.min_points:
            return None

        # Undistort before passing to Essential matrix solver
        prev_ud = self._undistort(prev_pts)
        curr_ud = self._undistort(curr_pts)

        E, e_mask = self._compute_essential(prev_ud, curr_ud)
        if E is None:
            return None

        R, t, final_mask = self._recover_pose(E, prev_ud, curr_ud, e_mask)
        if R is None:
            return None

        inlier_ratio = final_mask.sum() / len(valid_ids)
        if inlier_ratio < self.min_inlier_ratio:
            return None

        inlier_ids  = valid_ids[final_mask]
        outlier_ids = valid_ids[~final_mask]

        return {
            'R':            R,
            't':            t,
            'inlier_ids':   inlier_ids,
            'outlier_ids':  outlier_ids,
            'inlier_ratio': inlier_ratio,
        }

    def _extract_correspondences(self, histories):
        prev_pts  = []
        curr_pts  = []
        valid_ids = []

        for fid, history in histories.items():
            if len(history) < 2:
                continue
            curr_pts.append(history[0])   # most recent observation
            prev_pts.append(history[1])   # one frame prior
            valid_ids.append(fid)

        if len(prev_pts) < self.min_points:
            return None, None, None

        return (
            np.array(prev_pts,  dtype=np.float32),
            np.array(curr_pts,  dtype=np.float32),
            np.array(valid_ids, dtype=np.int64),
        )

    def _undistort(self, pts: np.ndarray) -> np.ndarray:
        undistorted = cv2.undistortPoints(
            pts.reshape(-1, 1, 2),
            self.K,
            self.dist_coeffs,
            P=self.K,          # re-project to pixel space
        )
        return undistorted.reshape(-1, 2)

    def _compute_essential(self, prev_ud, curr_ud):
        E, mask = cv2.findEssentialMat(
            prev_ud,
            curr_ud,
            self.K,
            method=cv2.RANSAC,
            prob=self.ransac_prob,
            threshold=self.ransac_threshold,
        )

        # OpenCV can return (9,3) when multiple E solutions found — validate
        if E is None or E.shape != (3, 3):
            return None, None

        return E, mask.ravel().astype(bool)

    def _recover_pose(self, E, prev_ud, curr_ud, e_mask):
        n_inliers, R, t, pose_mask = cv2.recoverPose(
            E,
            prev_ud,
            curr_ud,
            self.K,
            mask=e_mask.astype(np.uint8).reshape(-1, 1),  # pass E mask in
        )

        if n_inliers < self.min_points:
            return None, None, None

        # pose_mask is over the full input — compose with e_mask
        final_mask = e_mask & pose_mask.ravel().astype(bool)

        return R, t, final_mask