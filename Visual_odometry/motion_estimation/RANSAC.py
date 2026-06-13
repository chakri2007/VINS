import numpy as np
import cv2


class RANSACMotionEstimator:

    def __init__(
        self,
        K: np.ndarray,
        dist_coeffs: np.ndarray,
        ransac_prob: float = 0.99,
        ransac_threshold: float = 1.0,   # pixels
        min_inlier_ratio: float = 0.05,
        min_points: int = 8,
    ):
        self.K                = K.astype(np.float64)
        self.dist_coeffs      = dist_coeffs.astype(np.float64)
        self.ransac_prob      = ransac_prob
        self.ransac_threshold = ransac_threshold
        self.min_inlier_ratio = min_inlier_ratio
        self.min_points       = min_points


    def estimate(self, histories: dict) -> dict | None:

        pts_a, pts_b, valid_ids, frame_a, frame_b = \
            self._extract_correspondences(histories)

        if pts_a is None or len(pts_a) < self.min_points:
            return None

        pts_a_ud = self._undistort(pts_a)
        pts_b_ud = self._undistort(pts_b)

        E, e_mask = self._compute_essential(pts_a_ud, pts_b_ud)
        if E is None:
            return None

        R, t, final_mask = self._recover_pose(E, pts_a_ud, pts_b_ud, e_mask)
        if R is None:
            return None

        inlier_ratio = final_mask.sum() / len(valid_ids)
        if inlier_ratio < self.min_inlier_ratio:
            return None

        inlier_ids  = valid_ids[ final_mask]
        outlier_ids = valid_ids[~final_mask]

        return {
            'R'            : R,
            't'            : t,
            'inlier_ids'   : inlier_ids,
            'outlier_ids'  : outlier_ids,
            'inlier_ratio' : inlier_ratio,
            'frame_a'      : frame_a,
            'frame_b'      : frame_b,
            'pts_a'        : pts_a_ud[final_mask],   # undistorted inliers
            'pts_b'        : pts_b_ud[final_mask],
        }

    def _extract_correspondences(self, histories):
        pts_a_list  = []
        pts_b_list  = []
        valid_ids   = []
        frame_a_set = set()
        frame_b_set = set()

        for feat_id, obs_list in histories.items():
            if len(obs_list) < 2:
                continue
            # obs_list is newest-first  →  [0] = newest, [1] = one before
            frame_b_idx, ub, vb = obs_list[0]
            frame_a_idx, ua, va = obs_list[1]

            pts_b_list.append([ub, vb])
            pts_a_list.append([ua, va])
            valid_ids.append(feat_id)
            frame_a_set.add(frame_a_idx)
            frame_b_set.add(frame_b_idx)

        if len(pts_a_list) < self.min_points:
            return None, None, None, None, None


        frame_a = max(frame_a_set, key=lambda f: sum(
            1 for obs in histories.values()
            if len(obs) >= 2 and obs[1][0] == f
        ))
        frame_b = max(frame_b_set, key=lambda f: sum(
            1 for obs in histories.values()
            if len(obs) >= 1 and obs[0][0] == f
        ))

        return (
            np.array(pts_a_list, dtype=np.float32),
            np.array(pts_b_list, dtype=np.float32),
            np.array(valid_ids,  dtype=np.int64),
            frame_a,
            frame_b,
        )

    def _undistort(self, pts: np.ndarray) -> np.ndarray:
        undist = cv2.undistortPoints(
            pts.reshape(-1, 1, 2).astype(np.float64),
            self.K,
            self.dist_coeffs,
            P=self.K,
        )
        return undist.reshape(-1, 2).astype(np.float64)

    def _compute_essential(self, pts_a, pts_b):
        E, mask = cv2.findEssentialMat(
            pts_a, pts_b,
            self.K,
            method=cv2.RANSAC,
            prob=self.ransac_prob,
            threshold=self.ransac_threshold,
        )


        if E is None or E.shape != (3, 3):
            return None, None

        return E, mask.ravel().astype(bool)

    def _recover_pose(self, E, pts_a, pts_b, e_mask):
        n_inliers, R, t, pose_mask = cv2.recoverPose(
            E,
            pts_a,
            pts_b,
            self.K,
            mask=e_mask.astype(np.uint8).reshape(-1, 1),
        )

        if n_inliers < self.min_points:
            return None, None, None
        
        
        final_mask = e_mask & pose_mask.ravel().astype(bool)

        return R, t, final_mask