"""
stereo_rectifier.py
───────────────────
Computes stereo rectification maps once at construction time from the
left and right camera calibration dicts (same YAML format as
left_camera.yaml / right_camera.yaml).

Usage
─────
    rectifier = StereoRectifier(calib_left, calib_right)
    rect_left, rect_right = rectifier.rectify(gray_left, gray_right)
"""

import cv2
import numpy as np


class StereoRectifier:

    def __init__(self, calib_left: dict, calib_right: dict):
        """
        Parameters
        ──────────
        calib_left  : dict loaded from left_camera.yaml
        calib_right : dict loaded from right_camera.yaml

        Both dicts must contain:
            intrinsics                : [fu, fv, cu, cv]
            distortion_coefficients   : list (4 or 5 values)
            resolution                : [width, height]
            T_BS                      : {'data': [...16 floats...]}
        """
        # ── Parse intrinsics ──────────────────────────────────────────────
        self.K_l, self.D_l = self._parse_calib(calib_left)
        self.K_r, self.D_r = self._parse_calib(calib_right)

        w = int(calib_left['resolution'][0])
        h = int(calib_left['resolution'][1])
        self.image_size = (w, h)

        # ── Recover R_lr, t_lr from body-frame extrinsics ────────────────
        # T_BS maps sensor → body.  To go from left → right camera frame:
        #   T_rl = T_BS_right @ inv(T_BS_left)
        T_BS_l = np.array(calib_left['T_BS']['data'],  dtype=np.float64).reshape(4, 4)
        T_BS_r = np.array(calib_right['T_BS']['data'], dtype=np.float64).reshape(4, 4)

        T_rl = T_BS_r @ np.linalg.inv(T_BS_l)   # right-cam w.r.t. left-cam

        R_lr = T_rl[:3, :3]
        t_lr = T_rl[:3,  3]                       # metres (body-frame units)

        # Baseline: magnitude of translation along x-axis (signed below)
        self.baseline = float(np.linalg.norm(t_lr))

        # ── Stereo rectification ──────────────────────────────────────────
        (self._R1, self._R2,
         self._P1, self._P2,
         self._Q,  _, _) = cv2.stereoRectify(
            self.K_l, self.D_l,
            self.K_r, self.D_r,
            self.image_size,
            R_lr, t_lr,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0,                  # no black borders
        )

        # Precompute undistort+rectify maps (fast fixed-point remapping)
        self._map1_l, self._map2_l = cv2.initUndistortRectifyMap(
            self.K_l, self.D_l, self._R1, self._P1,
            self.image_size, cv2.CV_16SC2,
        )
        self._map1_r, self._map2_r = cv2.initUndistortRectifyMap(
            self.K_r, self.D_r, self._R2, self._P2,
            self.image_size, cv2.CV_16SC2,
        )

        # Rectified left intrinsics (for downstream triangulation)
        self.K_rect = self._P1[:3, :3].copy()

        print(
            f"[StereoRectifier] baseline={self.baseline*100:.1f} cm  "
            f"image_size={self.image_size}  "
            f"fx_rect={self.K_rect[0,0]:.1f}"
        )

    # ── Public API ────────────────────────────────────────────────────────

    def rectify(
        self,
        gray_left:  np.ndarray,
        gray_right: np.ndarray,
    ):
        """
        Apply precomputed rectification maps to both frames.

        Parameters
        ──────────
        gray_left  : (H, W) uint8 raw left  grayscale image
        gray_right : (H, W) uint8 raw right grayscale image

        Returns
        ───────
        rect_left, rect_right : (H, W) uint8  rectified grayscale images
        """
        rect_left  = cv2.remap(gray_left,  self._map1_l, self._map2_l,
                               cv2.INTER_LINEAR)
        rect_right = cv2.remap(gray_right, self._map1_r, self._map2_r,
                               cv2.INTER_LINEAR)
        return rect_left, rect_right

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _parse_calib(calib: dict):
        intr = calib['intrinsics']          # [fu, fv, cu, cv]
        K = np.array([
            [intr[0], 0.0,    intr[2]],
            [0.0,    intr[1], intr[3]],
            [0.0,    0.0,    1.0],
        ], dtype=np.float64)
        D = np.array(calib['distortion_coefficients'], dtype=np.float64)
        return K, D
