"""
imu_measurement.py

One synchronized IMU measurement.
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class IMUMeasurement:
    """
    One IMU sample.

    Attributes
    ----------
    timestamp : float
        Measurement time (seconds).

    accel : (3,)
        Linear acceleration (m/s²).

    gyro : (3,)
        Angular velocity (rad/s).
    """

    timestamp: float

    accel: np.ndarray

    gyro: np.ndarray