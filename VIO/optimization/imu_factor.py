"""
imu_factor.py

IMU factor connecting two consecutive [pose, velocity, bias] states in
the Phase-3 windowed factor graph (GraphBuilder.build_windowed_vio).
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np

from imu.preintegration import PreintegratedIMU
from optimization.ceres_bundle_adjustment_motion import _regularize_information


@dataclass
class IMUFactor:
    """
    One IMU preintegration constraint between two consecutive
    keyframes' [pose, velocity, bias] states.

    Parameters
    ----------
    from_view : int
        Previous (older) camera view.

    to_view : int
        Current (newer) camera view.

    preintegration : PreintegratedIMU
        Preintegrated IMU measurements between the two views,
        including its own error-state covariance (preintegration.
        covariance) and bias Jacobians -- the windowed BA wrapper
        (CeresBundleAdjusterWindow, optimization/
        ceres_bundle_adjustment_window.py) derives the dense 15x15
        sqrt-information directly from that covariance, exactly like
        BA_motion's wrapper does, rather than from the field below.

    information : (15,15) ndarray or None
        Optional caller-supplied dense information matrix, used
        as-is instead of deriving one from preintegration.covariance
        if provided. None (the default) means "derive it from the
        preintegration", which is what build_windowed_vio does.
    """

    from_view: int

    to_view: int

    preintegration: PreintegratedIMU

    information: Optional[np.ndarray] = None

    @property
    def sqrt_information(self):
        """
        Square-root information matrix used for residual weighting,
        derived from `information` if explicitly supplied, otherwise
        from the preintegration's own error-state covariance (the
        normal path -- see the class docstring).
        """
        if self.information is not None:
            return np.linalg.cholesky(self.information)

        # Bug fix: this used to add a *fixed* eps=1e-9 ridge and invert
        # directly. Real preintegration covariances for short/degenerate
        # intervals go as low as ~1e-11 on the diagonal (or are outright
        # singular), so that fixed ridge silently became the dominant
        # term, producing information eigenvalues of order 1e9 -- a
        # single IMU factor able to outweigh every reprojection factor
        # in the window by many orders of magnitude and wreck the
        # windowed solver's conditioning (see the matching fix and
        # longer explanation in ceres_bundle_adjustment_motion.py's
        # _regularize_information, which this now shares).
        cov = np.asarray(self.preintegration.covariance, dtype=np.float64)
        information = _regularize_information(cov)
        return np.linalg.cholesky(information)
