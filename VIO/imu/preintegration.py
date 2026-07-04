"""
preintegration.py

IMU preintegration following the manifold formulation used by
modern Visual-Inertial Odometry systems.

This implementation is inspired by

    • Forster et al.
      "IMU Preintegration on Manifold"

    • GTSAM

    • VINS-Mono

The class computes

    ΔR
    Δv
    Δp

together with

    covariance
    bias Jacobians

so that the same object can later be used by

    • VI Alignment
    • IMU Factors
    • Sliding Window Optimization
"""

from dataclasses import dataclass
from copy import deepcopy

import numpy as np

from imu.imu_measurement import IMUMeasurement

from imu.math_utils import (
    skew,
    exp_so3,
    normalize_rotation,
    right_jacobian,
)


# ============================================================
# Result container
# ============================================================

@dataclass
class PreintegratedIMU:
    """
    Result of IMU preintegration.

    Parameters
    ----------
    delta_R : (3,3)
        Relative rotation.

    delta_v : (3,)
        Relative velocity increment.

    delta_p : (3,)
        Relative position increment.

    delta_t : float
        Integrated time.

    covariance : (15,15)
        Error-state covariance.

    J_R_bg : (3,3)
        Jacobian of rotation wrt gyro bias.

    J_v_bg : (3,3)
        Jacobian of velocity wrt gyro bias.

    J_v_ba : (3,3)
        Jacobian of velocity wrt accel bias.

    J_p_bg : (3,3)
        Jacobian of position wrt gyro bias.

    J_p_ba : (3,3)
        Jacobian of position wrt accel bias.

    bias_g : (3,)
        Gyroscope bias used during integration.

    bias_a : (3,)
        Accelerometer bias used during integration.
    """

    delta_R: np.ndarray
    delta_v: np.ndarray
    delta_p: np.ndarray

    delta_t: float

    covariance: np.ndarray

    J_R_bg: np.ndarray
    J_v_bg: np.ndarray
    J_v_ba: np.ndarray
    J_p_bg: np.ndarray
    J_p_ba: np.ndarray

    bias_g: np.ndarray
    bias_a: np.ndarray

    # ------------------------------------------------------------
    # First-order bias correction
    # ------------------------------------------------------------

    def bias_corrected_delta(self, bias_g_new, bias_a_new):
        """
        First-order correction of (ΔR, Δv, Δp) for a bias update,
        without re-running the full integration.

        Valid as long as the new bias is close to the bias that was
        used during integration (the linearization point).

        Parameters
        ----------
        bias_g_new : (3,)
            Updated gyroscope bias.

        bias_a_new : (3,)
            Updated accelerometer bias.

        Returns
        -------
        delta_R_corrected : (3,3)
        delta_v_corrected : (3,)
        delta_p_corrected : (3,)
        """

        d_bg = bias_g_new - self.bias_g
        d_ba = bias_a_new - self.bias_a

        #
        # Rotation correction lives on the manifold:
        #
        #   ΔR(bg + δbg) ≈ ΔR(bg) · Exp(J_R_bg · δbg)
        #
        delta_R_corrected = normalize_rotation(
            self.delta_R @ exp_so3(self.J_R_bg @ d_bg)
        )

        #
        # Velocity and position corrections are linear in the
        # tangent space.
        #
        delta_v_corrected = (
            self.delta_v
            + self.J_v_bg @ d_bg
            + self.J_v_ba @ d_ba
        )

        delta_p_corrected = (
            self.delta_p
            + self.J_p_bg @ d_bg
            + self.J_p_ba @ d_ba
        )

        return delta_R_corrected, delta_v_corrected, delta_p_corrected


# ============================================================
# IMU Preintegrator
# ============================================================

class IMUPreintegrator:
    """
    Bias-aware IMU preintegrator.

    The class accumulates IMU measurements between two keyframes
    while maintaining

        ΔR
        Δv
        Δp

    together with covariance and bias Jacobians.
    """

    def __init__(
        self,
        gyro_noise=1.0e-3,
        accel_noise=1.0e-2,
        gyro_random_walk=1.0e-5,
        accel_random_walk=1.0e-4,
        bias_g=None,
        bias_a=None,
    ):
        """
        Parameters
        ----------
        gyro_noise : float
            Gyroscope white noise, given as a *continuous-time*
            spectral density (units: rad/s / sqrt(Hz)), e.g. the
            "noise density" figure from an IMU datasheet -- NOT the
            std you'd see on a single discrete sample. See the note
            on `self.Q` below for how this gets converted into a
            per-step, per-sample covariance during propagation.

        accel_noise : float
            Accelerometer white noise, continuous-time spectral
            density (units: m/s^2 / sqrt(Hz)). Same convention as
            `gyro_noise`.

        gyro_random_walk : float
            Gyroscope bias random walk, continuous-time spectral
            density (units: rad/s^2 / sqrt(Hz)).

        accel_random_walk : float
            Accelerometer bias random walk, continuous-time spectral
            density (units: m/s^3 / sqrt(Hz)).

        bias_g : (3,), optional
            Initial gyroscope bias.

        bias_a : (3,), optional
            Initial accelerometer bias.
        """

        self.gyro_noise = gyro_noise
        self.accel_noise = accel_noise

        self.gyro_random_walk = gyro_random_walk
        self.accel_random_walk = accel_random_walk

        if bias_g is None:
            bias_g = np.zeros(3)

        if bias_a is None:
            bias_a = np.zeros(3)

        self.bias_g = bias_g.astype(float).copy()
        self.bias_a = bias_a.astype(float).copy()

        #
        # Continuous-time process noise spectral density Qc.
        #
        # NOISE CONVENTION (read this before touching propagation):
        #
        #   * `self.Q` holds *continuous-time* spectral densities
        #     (Qc), i.e. the values you'd read off an IMU datasheet
        #     as "noise density". They do NOT depend on the sample
        #     rate.
        #
        #   * The noise-input matrix `B` built in
        #     `_propagate_covariance` already has the per-step `dt`
        #     scaling baked into it (e.g. `Jr * dt`, `R * dt`,
        #     `0.5 * R * dt**2` -- the same structure as the `A`
        #     matrix's off-diagonal terms), matching the discrete
        #     recursion in Forster et al., "IMU Preintegration on
        #     Manifold".
        #
        #   * Because `B` already carries one factor of `dt`, the
        #     noise covariance it gets multiplied by must be the
        #     *discrete-time sample* covariance, which for white
        #     noise with continuous-time density Qc over a step of
        #     length `dt` is:
        #
        #         Qd = Qc / dt
        #
        #     NOT `Qc * dt`. (Euler-Maruyama discretization of
        #     dx = F x dt + G dw with Cov(dw) = Qc*dt, combined with
        #     B == G*dt, requires B @ (Qc/dt) @ B.T == G @ Qc @ G.T
        #     * dt to recover the correct continuous-time noise
        #     contribution -- squaring `B`'s extra `dt` and then
        #     dividing it back out.)
        #
        #     This is verified directly in
        #     tests/test_preintegration.py::test_covariance_matches_monte_carlo,
        #     which compares this analytic `P` against the sample
        #     covariance of many randomly-perturbed IMU integrations.
        #     If you ever change this scaling, that test is the one
        #     to check.
        #
        self.Q = np.diag(
            [
                gyro_noise ** 2,
                gyro_noise ** 2,
                gyro_noise ** 2,

                accel_noise ** 2,
                accel_noise ** 2,
                accel_noise ** 2,

                gyro_random_walk ** 2,
                gyro_random_walk ** 2,
                gyro_random_walk ** 2,

                accel_random_walk ** 2,
                accel_random_walk ** 2,
                accel_random_walk ** 2,
            ]
        )

        self.reset()


    # ============================================================
    # Reset
    # ============================================================

    def reset(self):
        """
        Reset accumulated preintegration.
        """

        #
        # Nominal state
        #

        self.delta_R = np.eye(3)

        self.delta_v = np.zeros(3)

        self.delta_p = np.zeros(3)

        self.delta_t = 0.0

        #
        # Error-state covariance
        #
        # Ordering
        #
        #   θ
        #   v
        #   p
        #   bg
        #   ba
        #

        self.P = np.zeros((15, 15))

        #
        # Bias Jacobians
        #

        self.J_R_bg = np.zeros((3, 3))

        self.J_v_bg = np.zeros((3, 3))
        self.J_v_ba = np.zeros((3, 3))

        self.J_p_bg = np.zeros((3, 3))
        self.J_p_ba = np.zeros((3, 3))

        #
        # Timing
        #

        self.previous_timestamp = None


    # ============================================================
    # Internal propagation
    # ============================================================

    def _integrate_nominal(
        self,
        omega,
        accel,
        dt,
    ):
        """
        Propagate the nominal preintegrated state.

        Parameters
        ----------
        omega : (3,)
            Bias-corrected angular velocity.

        accel : (3,)
            Bias-corrected acceleration.

        dt : float
            Time step.

        Returns
        -------
        dR : (3,3)
            Incremental rotation applied this step (returned so the
            covariance / Jacobian propagation can reuse it without
            recomputation).
        """

        #
        # Rotation increment
        #
        dR = exp_so3(omega * dt)

        #
        # Specific force expressed in the current
        # preintegration frame.
        #
        accel_world = self.delta_R @ accel

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
        self.delta_R = normalize_rotation(
            self.delta_R @ dR
        )

        #
        # Time
        #
        self.delta_t += dt

        return dR


    def _propagate_covariance(
        self,
        R_k,
        omega,
        accel,
        dR,
        dt,
    ):
        """
        Propagate the 15x15 error-state covariance by one step.

        Error state ordering
            θ  (rotation error, 3)
            v  (velocity error, 3)
            p  (position error, 3)
            bg (gyro bias error, 3)
            ba (accel bias error, 3)

        Parameters
        ----------
        R_k : (3,3)
            ΔR *before* this step's update (linearization point).

        omega : (3,)
            Bias-corrected angular velocity for this step.

        accel : (3,)
            Bias-corrected acceleration for this step.

        dR : (3,3)
            Incremental rotation Exp(omega * dt) for this step.

        dt : float
            Time step.
        """

        acc_skew = skew(accel)

        Jr = right_jacobian(omega * dt)

        #
        # State transition matrix A (15x15), linearizing the
        # discrete-time error dynamics about the current estimate.
        #
        A = np.eye(15)

        # d(theta_k+1) / d(theta_k)
        A[0:3, 0:3] = dR.T

        # d(v_k+1) / d(theta_k)
        A[3:6, 0:3] = -R_k @ acc_skew * dt
        # d(v_k+1) / d(v_k)
        A[3:6, 3:6] = np.eye(3)

        # d(p_k+1) / d(theta_k)
        A[6:9, 0:3] = -0.5 * R_k @ acc_skew * dt * dt
        # d(p_k+1) / d(v_k)
        A[6:9, 3:6] = np.eye(3) * dt
        # d(p_k+1) / d(p_k)
        A[6:9, 6:9] = np.eye(3)

        # d(theta_k+1) / d(bg)
        A[0:3, 9:12] = -Jr * dt

        # d(v_k+1) / d(ba)
        #
        # accel = accel_raw - ba, so a perturbation δba enters the
        # velocity/position dynamics exactly like accelerometer
        # white noise does (same -R δba term) -- this is a direct,
        # same-step coupling, not one mediated by theta.
        A[3:6, 12:15] = -R_k * dt

        # d(p_k+1) / d(ba)
        A[6:9, 12:15] = -0.5 * R_k * dt * dt

        #
        # Note: there is deliberately no direct A[3:6, 9:12] or
        # A[6:9, 9:12] term for gyro bias. Unlike accel bias, δbg
        # only enters the dynamics through the rotation-rate error
        # (δθ̇ = ... - δbg); it has no direct term in the v̇ / ṗ
        # equations. Its effect on v and p is entirely mediated by
        # θ and is already captured by composing A[3:6,0:3]/
        # A[6:9,0:3] with A[0:3,9:12] across steps -- confirmed
        # numerically: d(v_new)/d(bg) and d(p_new)/d(bg) are exactly
        # zero within a single step (see validation).
        #
        # bias states are random-walk: identity block already set
        # by np.eye(15) above for [9:12,9:12] and [12:15,12:15].

        #
        # Noise Jacobian B (15x12), mapping the continuous-time
        # noise vector [n_g, n_a, n_bg, n_ba] onto the error state
        # over this discrete step.
        #
        B = np.zeros((15, 12))

        # gyro white noise -> rotation error
        B[0:3, 0:3] = Jr * dt

        # accel white noise -> velocity / position error
        B[3:6, 3:6] = R_k * dt
        B[6:9, 3:6] = 0.5 * R_k * dt * dt

        # bias random walk -> bias error
        B[9:12, 6:9] = np.eye(3) * dt
        B[12:15, 9:12] = np.eye(3) * dt

        #
        # Discrete-time process noise covariance: Qd = Qc / dt.
        #
        # See the "NOISE CONVENTION" note in __init__ where `self.Q`
        # is built, and
        # tests/test_preintegration.py::test_covariance_matches_monte_carlo
        # for the empirical check of this exact line. Do not change
        # this to `self.Q * dt` without re-running that test -- it's
        # a very common but incorrect "fix".
        #
        Q_discrete = self.Q / dt

        self.P = (
            A @ self.P @ A.T
            + B @ Q_discrete @ B.T
        )

        #
        # Keep the covariance numerically symmetric.
        #
        self.P = 0.5 * (self.P + self.P.T)


    def _propagate_jacobians(
        self,
        R_k,
        omega,
        accel,
        dR,
        dt,
    ):
        """
        Propagate the bias Jacobians by one step.

        These allow (ΔR, Δv, Δp) to be corrected to first order for
        a changed bias estimate, without re-integrating the raw IMU
        measurements (see `PreintegratedIMU.bias_corrected_delta`).

        Parameters
        ----------
        R_k : (3,3)
            ΔR *before* this step's update.

        omega : (3,)
            Bias-corrected angular velocity for this step.

        accel : (3,)
            Bias-corrected acceleration for this step.

        dR : (3,3)
            Incremental rotation Exp(omega * dt) for this step.

        dt : float
            Time step.
        """

        acc_skew = skew(accel)

        Jr = right_jacobian(omega * dt)

        #
        # Snapshot the previous Jacobians; several of the updates
        # below depend on each other's *old* values.
        #
        J_R_bg_old = self.J_R_bg.copy()
        J_v_bg_old = self.J_v_bg.copy()
        J_v_ba_old = self.J_v_ba.copy()
        J_p_bg_old = self.J_p_bg.copy()
        J_p_ba_old = self.J_p_ba.copy()

        #
        # d(ΔR) / d(bg)
        #
        self.J_R_bg = (
            dR.T @ J_R_bg_old
            - Jr * dt
        )

        #
        # d(Δv) / d(bg), d(Δv) / d(ba)
        #
        self.J_v_bg = (
            J_v_bg_old
            - R_k @ acc_skew @ J_R_bg_old * dt
        )

        self.J_v_ba = (
            J_v_ba_old
            - R_k * dt
        )

        #
        # d(Δp) / d(bg), d(Δp) / d(ba)
        #
        self.J_p_bg = (
            J_p_bg_old
            + J_v_bg_old * dt
            - 0.5 * R_k @ acc_skew @ J_R_bg_old * dt * dt
        )

        self.J_p_ba = (
            J_p_ba_old
            + J_v_ba_old * dt
            - 0.5 * R_k * dt * dt
        )


    # ============================================================
    # Public integration
    # ============================================================

    def integrate(
        self,
        measurement: IMUMeasurement,
    ):
        """
        Integrate a single IMU sample.

        Parameters
        ----------
        measurement : IMUMeasurement
        """

        #
        # First measurement initializes the clock.
        #
        if self.previous_timestamp is None:

            self.previous_timestamp = measurement.timestamp
            return

        dt = measurement.timestamp - self.previous_timestamp

        self.previous_timestamp = measurement.timestamp

        #
        # Ignore invalid measurements.
        #
        if dt <= 0.0:
            return

        #
        # Bias corrected measurements
        #
        omega = measurement.gyro - self.bias_g

        accel = measurement.accel - self.bias_a

        #
        # ΔR before this step's update; this is the linearization
        # point used by both the covariance and Jacobian updates.
        #
        R_k = self.delta_R.copy()

        #
        # Nominal propagation
        #
        dR = self._integrate_nominal(
            omega,
            accel,
            dt,
        )

        #
        # Error-state propagation.
        #
        self._propagate_covariance(
            R_k,
            omega,
            accel,
            dR,
            dt,
        )

        self._propagate_jacobians(
            R_k,
            omega,
            accel,
            dR,
            dt,
        )


    # ============================================================
    # Batch integration
    # ============================================================

    def integrate_measurements(
        self,
        measurements,
    ):
        """
        Integrate all IMU measurements between two keyframes.

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

            covariance=self.P.copy(),

            J_R_bg=self.J_R_bg.copy(),

            J_v_bg=self.J_v_bg.copy(),
            J_v_ba=self.J_v_ba.copy(),

            J_p_bg=self.J_p_bg.copy(),
            J_p_ba=self.J_p_ba.copy(),

            bias_g=self.bias_g.copy(),
            bias_a=self.bias_a.copy(),
        )


    # ============================================================
    # Utility
    # ============================================================

    def set_bias(self, bias_g, bias_a):
        """
        Update the linearization biases and reset accumulated
        preintegration.

        This does *not* attempt an incremental repropagation; call
        `integrate_measurements` again (or use
        `PreintegratedIMU.bias_corrected_delta` on an already
        finished result) if you only need a first-order correction.

        Parameters
        ----------
        bias_g : (3,)
        bias_a : (3,)
        """

        self.bias_g = np.asarray(bias_g, dtype=float).copy()
        self.bias_a = np.asarray(bias_a, dtype=float).copy()

        self.reset()


    def copy(self):
        """
        Deep copy of the preintegrator.
        """

        return deepcopy(self)