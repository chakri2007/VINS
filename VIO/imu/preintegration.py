"""
preintegration.py

IMU preintegration between two camera frames.

Version 1:
    - No bias correction
    - No covariance propagation
    - No Jacobians

Outputs:
    ΔR
    Δv
    Δp
    Δt
"""

from dataclasses import dataclass

import cv2
import numpy as np

from imu.imu_measurement import IMUMeasurement


@dataclass
class PreintegratedIMU:
    """
    Result of IMU preintegration.

    Attributes
    ----------
    delta_R : (3,3)
        Relative rotation.

    delta_v : (3,)
        Relative velocity.

    delta_p : (3,)
        Relative position.

    delta_t : float
        Total integration time.
    """

    delta_R: np.ndarray
    delta_v: np.ndarray
    delta_p: np.ndarray
    delta_t: float


class IMUPreintegrator:
    """
    Bias-free IMU preintegrator.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """
        Reset accumulated preintegration.
        """

        self.delta_R = np.eye(3)

        self.delta_v = np.zeros(3)

        self.delta_p = np.zeros(3)

        self.delta_t = 0.0

        self.previous_timestamp = None

    def integrate(
        self,
        measurement: IMUMeasurement,
    ):
        """
        Integrate one IMU sample.
        """

        #
        # First measurement only initializes time.
        #
        if self.previous_timestamp is None:

            self.previous_timestamp = measurement.timestamp

            return

        dt = measurement.timestamp - self.previous_timestamp

        self.previous_timestamp = measurement.timestamp

        if dt <= 0:
            return

        #
        # Angular velocity
        #
        omega = measurement.gyro

        #
        # Rodrigues rotation increment
        #
        rvec = omega * dt

        dR, _ = cv2.Rodrigues(rvec)

        #
        # Acceleration
        #
        accel_world = self.delta_R @ measurement.accel

        #
        # Position update
        #
        self.delta_p += (
            self.delta_v * dt
            + 0.5 * accel_world * dt * dt
        )

        #
        # Velocity update
        #
        self.delta_v += accel_world * dt

        #
        # Rotation update
        #
        self.delta_R = self.delta_R @ dR

        #
        # Time
        #
        self.delta_t += dt

    def integrate_measurements(
        self,
        measurements,
    ):
        """
        Integrate a sequence of IMU measurements.

        Parameters
        ----------
        measurements : list[IMUMeasurement]

        Returns
        -------
        PreintegratedIMU
        """

        self.reset()

        for measurement in measurements:

            self.integrate(measurement)

        return PreintegratedIMU(
            delta_R=self.delta_R.copy(),
            delta_v=self.delta_v.copy(),
            delta_p=self.delta_p.copy(),
            delta_t=self.delta_t,
        )