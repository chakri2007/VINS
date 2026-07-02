"""
imu_factor.py

IMU factor connecting two consecutive camera poses.
"""

from dataclasses import dataclass
import numpy as np

from imu.preintegration import PreintegratedIMU


@dataclass
class IMUFactor:
    """
    One IMU constraint between two camera poses.

    Parameters
    ----------
    from_view : int
        Previous camera view.

    to_view : int
        Current camera view.

    preintegration : PreintegratedIMU
        Preintegrated IMU measurements between the two views.

    information : (9,9) ndarray
        Information matrix of the IMU measurement.
        (Initially identity.)
    """

    from_view: int

    to_view: int

    preintegration: PreintegratedIMU

    information: np.ndarray

    @property
    def sqrt_information(self):
        """
        Square-root information matrix used for residual weighting.
        """
        return np.linalg.cholesky(self.information)