"""
stereo_triangulator.py
──────────────────────
Per-feature block-matching stereo triangulator that works on rectified
image pairs.

For each 2-D feature tracked on the LEFT frame by KLT, we search for its
correspondence on the RIGHT frame along the epipolar line (a horizontal
strip in a rectified pair) using normalised cross-correlation (NCC) block
matching.  Valid disparities are converted to metric 3-D points using

    Z = f * baseline / disparity
    X = (u_left - cx) * Z / f
    Y = (v_left - cy) * Z / f

These landmarks are immediately in metric scale — no visual-inertial
alignment is required.

Usage
─────
    tri = StereoTriangulator(K_left, baseline_m, block_size=11,
                             min_disparity=1.0, max_disparity=128.0,
                             ncc_threshold=0.7)

    result = tri.triangulate(
        rect_left,           # rectified left  gray frame  (H, W)
        rect_right,          # rectified right gray frame  (H, W)
        left_points,         # (N, 2) float32  pixel coords on left frame
        feat_ids,            # (N,)   int64    feature IDs
        frame_idx,           # int
    )
    # result: dict with keys 'landmarks', 'feat_to_lm', 'observations'
    #         or None if < min_landmarks valid points produced.
"""

import cv2
import numpy as np
from typing import Optional, Tuple


class StereoTriangulator:

    def __init__(
        self,
        K: np.ndarray,
        baseline: float,
        block_size:    int   = 11,
        min_disparity: float = 1.0,
        max_disparity: float = 128.0,
        ncc_threshold: float = 0.7,
        epipolar_band: int   = 2,
        min_landmarks: int   = 5,
    ):
        """
        Parameters
        ──────────
        K              : left camera intrinsics (3×3)
        baseline       : stereo baseline in metres  (positive, left-to-right)
        block_size     : NCC patch half-size; full window = (2k+1)×(2k+1).
                         Must be odd; if even, rounded up.
        min_disparity  : reject matches closer than this many pixels
        max_disparity  : reject matches farther than this many pixels
                         (sets the max search range along the epipolar row)
        ncc_threshold  : minimum NCC score to accept a match   [0, 1]
        epipolar_band  : ±N rows to search around the epipolar line
                         (should be 0-2 px for well-rectified pairs)
        min_landmarks  : triangulate() returns None if fewer valid 3-D
                         points are produced
        """
        self.K             = K.astype(np.float64)
        self.baseline      = float(baseline)
        self.min_disparity = float(min_disparity)
        self.max_disparity = float(max_disparity)
        self.ncc_threshold = float(ncc_threshold)
        self.epipolar_band = int(epipolar_band)
        self.min_landmarks = int(min_landmarks)

        # Ensure block_size is odd
        bk = int(block_size)
        if bk % 2 == 0:
            bk += 1
        self.block_size = bk
        self.half       = bk // 2

        self.fx = float(K[0, 0])
        self.fy = float(K[1, 1])
        self.cx = float(K[0, 2])
        self.cy = float(K[1, 2])

        self._next_lm_id = 0

    # ── Public API ────────────────────────────────────────────────────────

    def triangulate(
        self,
        rect_left:   np.ndarray,
        rect_right:  np.ndarray,
        left_points: np.ndarray,
        feat_ids:    np.ndarray,
        frame_idx:   int,
    ) -> Optional[dict]:
        """
        Parameters
        ──────────
        rect_left    : (H, W) uint8 rectified left  gray image
        rect_right   : (H, W) uint8 rectified right gray image
        left_points  : (N, 2) float32  [u, v] pixel coordinates on left
        feat_ids     : (N,)   int64    KLT feature IDs (parallel to left_points)
        frame_idx    : current frame index (stored in observations)

        Returns
        ───────
        dict with:
            'landmarks'    : { lm_id: np.ndarray (3,) }   metric XYZ
            'feat_to_lm'   : { feat_id: lm_id }
            'observations' : { lm_id: [(frame_idx, u, v)] }
        or None if fewer than min_landmarks valid points are produced.
        """
        if len(left_points) == 0:
            return None

        H, W = rect_left.shape[:2]

        landmarks    = {}
        feat_to_lm   = {}
        observations = {}

        for feat_id, pt in zip(feat_ids, left_points):
            u_l = float(pt[0])
            v_l = float(pt[1])

            disparity = self._match_feature(rect_left, rect_right, u_l, v_l, H, W)
            if disparity is None:
                continue

            X, Y, Z = self._disparity_to_3d(u_l, v_l, disparity)

            # Basic sanity: point must be in front of camera
            if Z <= 0.0:
                continue

            lm_id = self._next_lm_id
            self._next_lm_id += 1

            landmarks[lm_id]    = np.array([X, Y, Z], dtype=np.float64)
            feat_to_lm[int(feat_id)] = lm_id
            observations[lm_id] = [(frame_idx, u_l, v_l)]

        if len(landmarks) < self.min_landmarks:
            return None

        return {
            'landmarks'    : landmarks,
            'feat_to_lm'   : feat_to_lm,
            'observations' : observations,
        }

    # ── Core matching ─────────────────────────────────────────────────────

    def _match_feature(
        self,
        left:  np.ndarray,
        right: np.ndarray,
        u_l:   float,
        v_l:   float,
        H:     int,
        W:     int,
    ) -> Optional[float]:
        """
        Search the rectified right image for the best NCC match of the
        patch around (u_l, v_l) in the left image.

        The search is restricted to:
          - same row ± epipolar_band   (epipolar constraint)
          - columns [u_l - max_disparity, u_l - min_disparity]
            (right feature must be to the LEFT of left feature)

        Returns disparity in pixels, or None if no valid match found.
        """
        h = self.half
        ui = int(round(u_l))
        vi = int(round(v_l))

        # Patch must be fully inside the left image
        if ui - h < 0 or ui + h >= W or vi - h < 0 or vi + h >= H:
            return None

        left_patch = left[vi - h: vi + h + 1,
                          ui - h: ui + h + 1].astype(np.float32)

        # Normalise the left patch once
        lp_norm = self._normalise_patch(left_patch)
        if lp_norm is None:
            return None

        # Search bounds on the right image (horizontal only)
        u_r_max = ui - int(self.min_disparity)
        u_r_min = ui - int(self.max_disparity)

        # Clamp to image bounds (patch centre must allow full patch)
        u_r_min = max(u_r_min, h)
        u_r_max = min(u_r_max, W - h - 1)

        if u_r_min > u_r_max:
            return None

        # Epipolar band rows on the right image
        v_r_min = max(vi - self.epipolar_band, h)
        v_r_max = min(vi + self.epipolar_band, H - h - 1)

        best_ncc  = -1.0
        best_disp = None

        for v_r in range(v_r_min, v_r_max + 1):
            # Extract a wide strip from the right image for this row
            strip = right[v_r - h: v_r + h + 1,
                          u_r_min - h: u_r_max + h + 1].astype(np.float32)

            if strip.shape[1] < self.block_size:
                continue

            # Slide the patch across the strip using matchTemplate (NCC)
            result = cv2.matchTemplate(strip, lp_norm, cv2.TM_CCOEFF_NORMED)
            # result shape: (2*epipolar_band+1, u_r_max - u_r_min + 1) approx
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if max_val > best_ncc:
                best_ncc  = max_val
                # max_loc[0] is the x offset inside the strip
                best_u_r  = u_r_min + max_loc[0]
                best_disp = float(ui - best_u_r)

        if best_ncc < self.ncc_threshold:
            return None
        if best_disp is None:
            return None
        if best_disp < self.min_disparity or best_disp > self.max_disparity:
            return None

        return best_disp

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _normalise_patch(patch: np.ndarray) -> Optional[np.ndarray]:
        """Return zero-mean unit-variance patch, or None if std ≈ 0."""
        std = patch.std()
        if std < 1e-3:
            return None
        return ((patch - patch.mean()) / std).astype(np.float32)

    def _disparity_to_3d(
        self, u_l: float, v_l: float, disparity: float
    ) -> Tuple[float, float, float]:
        """Convert left pixel + disparity → metric 3-D point (camera frame)."""
        Z = self.fx * self.baseline / disparity
        X = (u_l - self.cx) * Z / self.fx
        Y = (v_l - self.cy) * Z / self.fy
        return X, Y, Z

    # ── Introspection ─────────────────────────────────────────────────────

    def reset_id_counter(self, start: int = 0) -> None:
        """Optionally reset landmark ID counter (useful for tests)."""
        self._next_lm_id = start
