import numpy as np
import cv2


class RANSACMotionEstimator:

    def __init__(
        self,
        K: np.ndarray,
        dist_coeffs: np.ndarray,
        ransac_prob: float = 0.99,
        min_inlier_ratio: float = 0.05,
        min_points: int = 8,
    ):
        self.K = K.astype(np.float64)
        self.dist_coeffs = dist_coeffs.astype(np.float64)
        self.ransac_prob = ransac_prob
        self.min_inlier_ratio = min_inlier_ratio
        self.min_points = min_points

    def estimate(self, histories: dict) -> dict | None:

        pts_a, pts_b, valid_ids, frame_a, frame_b = \
            self._extract_correspondences(histories)

        if pts_a is None:
            return None

        motion = np.linalg.norm(pts_b - pts_a, axis=1)

        median_motion = np.median(motion)

        if median_motion < 0.5:
            #print("SKIP: motion too small")
            return None

        pts_a_ud = self._undistort(pts_a)
        pts_b_ud = self._undistort(pts_b)

        E, e_mask = self._compute_essential(pts_a_ud, pts_b_ud)

        if E is None:
            #print("FAIL: Essential matrix")
            return None

        essential_inliers = np.count_nonzero(e_mask)

        U, S, Vt = np.linalg.svd(E)


        R, t, final_mask = self._recover_pose(
            E,
            pts_a_ud,
            pts_b_ud,
            e_mask,
        )

        if R is None:
            return None

        final_inliers = np.count_nonzero(final_mask)

        inlier_ratio = final_inliers / len(valid_ids)

        if inlier_ratio < self.min_inlier_ratio:
            #print("FAIL: low inlier ratio")
            return None

        inlier_ids = valid_ids[final_mask]
        outlier_ids = valid_ids[~final_mask]

        return {
            "R": R,
            "t": t,
            "inlier_ids": inlier_ids,
            "outlier_ids": outlier_ids,
            "inlier_ratio": inlier_ratio,
            "frame_a": frame_a,
            "frame_b": frame_b,
            #"pts_a": pts_a_ud[final_mask],
            #"pts_b": pts_b_ud[final_mask],
            "pts_a": pts_a[final_mask],
            "pts_b": pts_b[final_mask],
        }

    def _extract_correspondences(self, histories):

        pts_a = []
        pts_b = []
        valid_ids = []

        frame_a_set = set()
        frame_b_set = set()

        for feat_id, history in histories.items():

            if len(history) < 2:
                continue

            frame_b_idx, ub, vb = history[0]
            frame_a_idx, ua, va = history[1]

            pts_b.append([ub, vb])
            pts_a.append([ua, va])

            valid_ids.append(feat_id)

            frame_a_set.add(frame_a_idx)
            frame_b_set.add(frame_b_idx)

        if len(pts_a) < self.min_points:
            return None, None, None, None, None

        frame_a = max(
            frame_a_set,
            key=lambda f: sum(
                1
                for obs in histories.values()
                if len(obs) >= 2 and obs[1][0] == f
            )
        )

        frame_b = max(
            frame_b_set,
            key=lambda f: sum(
                1
                for obs in histories.values()
                if len(obs) >= 1 and obs[0][0] == f
            )
        )

        return (
            np.array(pts_a, dtype=np.float32),
            np.array(pts_b, dtype=np.float32),
            np.array(valid_ids, dtype=np.int64),
            frame_a,
            frame_b,
        )

    def _undistort(self, pts):

        pts_ud = cv2.undistortPoints(
            pts.reshape(-1, 1, 2),
            self.K,
            self.dist_coeffs
        )

        return pts_ud.reshape(-1, 2)

    def _compute_essential(self, pts_a_ud, pts_b_ud):

        E, mask = cv2.findEssentialMat(
            pts_a_ud,
            pts_b_ud,
            focal=1.0,
            pp=(0.0, 0.0),
            method=cv2.RANSAC,
            prob=self.ransac_prob,
            threshold=1.0,
        )

        if E is None:
            return None, None

        if E.shape != (3, 3):
            return None, None

        return E, mask.ravel().astype(bool)

    def _recover_pose(self, E, pts_a_ud, pts_b_ud, e_mask):

        n_inliers, R, t, pose_mask = cv2.recoverPose(
            E,
            pts_b_ud,
            pts_a_ud,
            mask=e_mask.astype(np.uint8).reshape(-1, 1),
        )

        #print("recoverPose inliers :", n_inliers)

        if n_inliers < self.min_points:
            #print("FAIL: recoverPose")
            return None, None, None

        final_mask = e_mask & pose_mask.ravel().astype(bool)

        return R, t, final_mask