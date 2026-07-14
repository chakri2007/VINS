"""
imu_factor.py

IMU factor connecting two consecutive [pose, velocity, bias] states in
the Phase-3 windowed factor graph (GraphBuilder.build_windowed_vio).
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np

from imu.preintegration import PreintegratedIMU


def robust_sqrt_information(covariance, rel_floor=1e-6, min_floor=1e-12):
    """
    Dense sqrt-information L (L @ L.T == inv(covariance_regularized))
    from a preintegration error-state covariance, regularized in a way
    that scales with the covariance itself rather than a fixed
    absolute ridge.

    Why not a fixed `eps * I` ridge (the previous approach here and in
    ceres_bundle_adjustment_motion.py)? The natural eigenvalues of this
    covariance span many orders of magnitude depending on which
    error-state block they belong to (e.g. bias random-walk blocks are
    tiny even for perfectly healthy, well-sampled intervals -- see the
    synthetic sweep in tests/test_a_imu_covariance.py). A fixed
    eps=1e-9 sits far above those natural eigenvalues almost always, so
    it doesn't just prevent outright singularity -- it silently
    *replaces* the true (tiny) eigenvalue with eps, and inverting
    turns that into an information eigenvalue pinned at ~1/eps = 1e9,
    regardless of how much real data actually supports that direction.
    That phantom ~1e9-weighted constraint is what was producing the
    1e6-1e15 initial costs and downstream NO_CONVERGENCE in both
    BA_motion and the windowed BA.

    Instead, floor each eigenvalue relative to the covariance's own
    largest eigenvalue (i.e. cap the condition number at 1/rel_floor),
    so a genuinely well-constrained interval keeps its real,
    proportionate weighting, and only directions that are degenerate
    *relative to the rest of this specific interval* get clamped.

    Parameters
    ----------
    covariance : (N,N) array
    rel_floor : float
        Minimum eigenvalue, as a fraction of the largest eigenvalue
        (i.e. caps information-matrix condition number at ~1/rel_floor).
    min_floor : float
        Absolute floor, only relevant if every eigenvalue is ~0.
    """
    cov = np.asarray(covariance, dtype=np.float64)
    cov = 0.5 * (cov + cov.T)

    eigvals, eigvecs = np.linalg.eigh(cov)

    max_eig = float(eigvals.max())
    floor = max(rel_floor * max_eig, min_floor)

    eigvals_floored = np.clip(eigvals, floor, None)

    cov_reg = (eigvecs * eigvals_floored) @ eigvecs.T
    information = np.linalg.inv(cov_reg)
    information = 0.5 * (information + information.T)

    return np.linalg.cholesky(information)


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

        return robust_sqrt_information(self.preintegration.covariance)