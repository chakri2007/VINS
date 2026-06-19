import numpy as np
import cv2
from typing import Optional, Tuple


class PnPEstimator:
    def __init__(
        self,
        K: np.ndarray,
        dist_coeffs: np.ndarray,
        min_inliers: int = 12,
        reprojection_error_px: float = 4.0,
        confidence: float = 0.999,
        iterations: int = 200,
    ):
        self.K                     = K.astype(np.float64)
        self.dist_coeffs           = dist_coeffs.astype(np.float64)
        self.min_inliers           = min_inliers
        self.reprojection_error_px = reprojection_error_px
        self.confidence            = confidence
        self.iterations            = iterations

        self._last_R: Optional[np.ndarray] = None
        self._last_t: Optional[np.ndarray] = None

    def estimate(
        self,
        pts3d: np.ndarray,   # (N, 3)  3D landmark positions
        pts2d: np.ndarray,   # (N, 2)  corresponding 2D pixel observations
    ) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        if len(pts3d) < self.min_inliers:
            return None

        use_guess = self._last_R is not None
        if use_guess:
            rvec_init, _ = cv2.Rodrigues(self._last_R)
            tvec_init    = self._last_t.reshape(3, 1)
        else:
            rvec_init = np.zeros((3, 1), dtype=np.float64)
            tvec_init = np.zeros((3, 1), dtype=np.float64)

        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints      = pts3d.reshape(-1, 1, 3).astype(np.float64),
            imagePoints       = pts2d.reshape(-1, 1, 2).astype(np.float64),
            cameraMatrix      = self.K,
            distCoeffs        = self.dist_coeffs,
            rvec              = rvec_init if use_guess else None,
            tvec              = tvec_init if use_guess else None,
            useExtrinsicGuess = use_guess,
            iterationsCount   = self.iterations,
            reprojectionError = self.reprojection_error_px,
            confidence        = self.confidence,
            flags             = cv2.SOLVEPNP_ITERATIVE,
        )

        if not success or inliers is None:
            return None

        inlier_idx = inliers.flatten()
        if len(inlier_idx) < self.min_inliers:
            return None

        R, _ = cv2.Rodrigues(rvec)
        t    = tvec.reshape(3, 1)

        self._last_R = R.copy()
        self._last_t = t.copy()

        inlier_mask = np.zeros(len(pts3d), dtype=bool)
        inlier_mask[inlier_idx] = True

        return R, t, inlier_mask

    def reset(self):
        self._last_R = None
        self._last_t = None

    @property
    def has_prior(self) -> bool:
        return self._last_R is not None