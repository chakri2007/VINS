import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional


class Triangulator:

    def __init__(
        self,
        K: np.ndarray,
        min_angle_deg: float = 1.0,   # minimum bearing angle between rays
    ):
        self.K             = K.astype(np.float64)
        self.min_angle_deg = min_angle_deg
        self._next_lm_id   = 0        # landmark ID counter — increments globally

    def triangulate(
        self,
        kf_a: dict,
        kf_b: dict,
    ) -> Optional[dict]:
        # Find features common to both keyframes
        common = self._find_common_features(kf_a, kf_b)
        if len(common) == 0:
            return None

        feat_ids = np.array([c[0] for c in common], dtype=np.int64)
        pts_a    = np.array([[c[1], c[2]] for c in common], dtype=np.float64)
        pts_b    = np.array([[c[3], c[4]] for c in common], dtype=np.float64)

        # Build projection matrices  P = K @ [R | t]
        Ra, ta = kf_a['R'], kf_a['t']
        Rb, tb = kf_b['R'], kf_b['t']
        Pa = self.K @ np.hstack([Ra, ta])   # (3,4)
        Pb = self.K @ np.hstack([Rb, tb])   # (3,4)

        # Triangulate
        pts4d = cv2.triangulatePoints(
            Pa.astype(np.float32),
            Pb.astype(np.float32),
            pts_a.T.astype(np.float32),
            pts_b.T.astype(np.float32),
        )  # (4, N)

        # Homogeneous → 3D
        w = pts4d[3, :]
        valid_w = np.abs(w) > 1e-6
        pts3d = np.where(valid_w, pts4d[:3, :] / np.where(valid_w, w, 1.0), 0.0).T
        # pts3d : (N, 3)

        landmarks    = {}
        observations = {}
        feat_to_lm   = {}

        for i in range(len(feat_ids)):
            if not valid_w[i]:
                continue

            X       = pts3d[i]         # (3,)
            feat_id = feat_ids[i]

            # ── Filter 1: Cheirality ───────────────────────────────────
            X_a = Ra @ X + ta.ravel()
            X_b = Rb @ X + tb.ravel()
            if X_a[2] <= 0.0 or X_b[2] <= 0.0:
                continue

            # ── Filter 2: Triangulation angle ─────────────────────────
            angle_deg = self._triangulation_angle(X, ta.ravel(), tb.ravel())
            if angle_deg < self.min_angle_deg:
                continue

            # ── Store ─────────────────────────────────────────────────
            lm_id = self._next_lm_id
            self._next_lm_id += 1

            landmarks[lm_id]  = X

            obs_a = (kf_a['frame_idx'], pts_a[i, 0], pts_a[i, 1])
            obs_b = (kf_b['frame_idx'], pts_b[i, 0], pts_b[i, 1])
            observations[lm_id] = [obs_a, obs_b]

            feat_to_lm[feat_id] = lm_id

        if len(landmarks) == 0:
            return None

        return {
            'landmarks'    : landmarks,
            'observations' : observations,
            'feat_to_lm'   : feat_to_lm,
        }

    def _find_common_features(
        self,
        kf_a: dict,
        kf_b: dict,
    ) -> List[Tuple]:
        map_a = {fid: pt for fid, pt in zip(kf_a['feat_ids'], kf_a['pts'])}
        map_b = {fid: pt for fid, pt in zip(kf_b['feat_ids'], kf_b['pts'])}

        common = []
        for feat_id, pt_a in map_a.items():
            if feat_id in map_b:
                pt_b = map_b[feat_id]
                common.append((feat_id,
                                pt_a[0], pt_a[1],
                                pt_b[0], pt_b[1]))
        return common

    @staticmethod
    def _triangulation_angle(
        X: np.ndarray,
        cam_a_center: np.ndarray,
        cam_b_center: np.ndarray,
    ) -> float:
        ray_a = X - cam_a_center
        ray_b = X - cam_b_center

        norm_a = np.linalg.norm(ray_a)
        norm_b = np.linalg.norm(ray_b)

        if norm_a < 1e-9 or norm_b < 1e-9:
            return 0.0

        cos_angle = np.clip(
            np.dot(ray_a, ray_b) / (norm_a * norm_b),
            -1.0, 1.0
        )
        return float(np.degrees(np.arccos(cos_angle)))