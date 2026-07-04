"""
tests/test_preintegration.py

Regression tests for imu/preintegration.py.

Run with either:

    python -m unittest tests.test_preintegration -v

or, if pytest is installed:

    pytest tests/test_preintegration.py -v

By default, the slow Monte Carlo covariance test is SKIPPED so the
rest of the suite stays fast for everyday use. To run the full
suite including it:

    RUN_SLOW_TESTS=1 python -m unittest tests.test_preintegration -v

The most important test here is
`test_covariance_matches_monte_carlo`: it validates the analytic
error-state covariance `P` produced by `IMUPreintegrator` against
the sample covariance of many randomly-perturbed IMU integrations.

If you ever touch `_propagate_covariance`, `_propagate_jacobians`,
the `A`/`B` matrix construction, or the `Qc -> Qd` conversion
(`Q_discrete = self.Q / dt` in `_propagate_covariance`), re-run it
with RUN_SLOW_TESTS=1. It is the one piece of ground truth we have
that isn't just "the algebra looks right" -- it checks the actual
statistics. Run it before merging any change that touches
propagation, even though it's skipped by default locally.
"""

import sys
import os
import unittest

import numpy as np

# Allow running this file directly (`python tests/test_preintegration.py`)
# as well as via `python -m unittest` / pytest from the repo root.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preintegration import IMUPreintegrator
from imu.imu_measurement import IMUMeasurement
from imu.math_utils import exp_so3, log_so3, normalize_rotation

RUN_SLOW_TESTS = bool(os.environ.get("RUN_SLOW_TESTS"))


# ============================================================
# Helpers
# ============================================================

def make_measurements(gyro, accel, dt, n_steps, noise_fn=None):
    """
    Build a list of IMUMeasurement with constant true angular
    velocity / acceleration, optionally perturbed by `noise_fn(i)`
    which should return (gyro_noise, accel_noise) for step `i`.
    """

    measurements = []
    t = 0.0

    for i in range(n_steps + 1):

        if noise_fn is not None:
            ng, na = noise_fn(i)
        else:
            ng = np.zeros(3)
            na = np.zeros(3)

        measurements.append(
            IMUMeasurement(
                timestamp=t,
                gyro=gyro + ng,
                accel=accel + na,
            )
        )

        t += dt

    return measurements


# ============================================================
# Basic sanity tests
# ============================================================

class TestNominalIntegration(unittest.TestCase):

    def test_rotation_stays_on_so3(self):
        """
        delta_R must always be a proper rotation matrix: orthogonal
        with determinant +1, even after many integration steps.
        """

        gyro = np.array([0.4, -0.2, 0.1])
        accel = np.array([0.3, -9.8, 0.2])

        measurements = make_measurements(gyro, accel, dt=0.005, n_steps=200)

        pre = IMUPreintegrator()
        result = pre.integrate_measurements(measurements)

        R = result.delta_R

        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-9)
        self.assertAlmostEqual(np.linalg.det(R), 1.0, places=9)

    def test_zero_noise_rotation_matches_closed_form(self):
        """
        For constant angular velocity and zero noise, ΔR after N
        steps of dt should match Exp(omega * N * dt) (constant
        angular velocity integrates exactly under the exponential
        map, up to the per-step composition/normalization used by
        the integrator).
        """

        gyro = np.array([0.1, 0.0, 0.0])
        accel = np.zeros(3)
        dt = 0.001
        n_steps = 500

        measurements = make_measurements(gyro, accel, dt=dt, n_steps=n_steps)

        pre = IMUPreintegrator()
        result = pre.integrate_measurements(measurements)

        expected_R = exp_so3(gyro * n_steps * dt)

        # Compare via the rotation vector between the two, rather
        # than the raw matrices, to get an interpretable error size.
        error_vec = log_so3(expected_R.T @ result.delta_R)

        self.assertLess(np.linalg.norm(error_vec), 1e-6)


# ============================================================
# Bias Jacobian tests
# ============================================================

class TestBiasJacobians(unittest.TestCase):
    """
    Validate J_R_bg, J_v_bg, J_v_ba, J_p_bg, J_p_ba by comparing the
    first-order `bias_corrected_delta` prediction against actually
    re-integrating the same raw measurements with a perturbed bias.
    """

    def setUp(self):
        self.gyro = np.array([0.3, -0.15, 0.05])
        self.accel = np.array([0.2, -9.7, 0.4])
        self.dt = 0.005
        self.n_steps = 100

        self.measurements = make_measurements(
            self.gyro, self.accel, dt=self.dt, n_steps=self.n_steps
        )

        self.pre = IMUPreintegrator(bias_g=np.zeros(3), bias_a=np.zeros(3))
        self.result = self.pre.integrate_measurements(self.measurements)

    def _reintegrate_with_bias(self, bias_g, bias_a):
        pre = IMUPreintegrator(bias_g=bias_g, bias_a=bias_a)
        return pre.integrate_measurements(self.measurements)

    def test_gyro_bias_jacobian_first_order(self):
        d_bg = np.array([1e-4, -2e-4, 5e-5])

        truth = self._reintegrate_with_bias(d_bg, np.zeros(3))
        dR_pred, dv_pred, dp_pred = self.result.bias_corrected_delta(
            d_bg, np.zeros(3)
        )

        rot_err = np.linalg.norm(log_so3(truth.delta_R.T @ dR_pred))
        v_err = np.linalg.norm(truth.delta_v - dv_pred)
        p_err = np.linalg.norm(truth.delta_p - dp_pred)

        self.assertLess(rot_err, 1e-6)
        self.assertLess(v_err, 1e-6)
        self.assertLess(p_err, 1e-6)

    def test_accel_bias_jacobian_first_order(self):
        d_ba = np.array([2e-4, -1e-4, 3e-4])

        truth = self._reintegrate_with_bias(np.zeros(3), d_ba)
        dR_pred, dv_pred, dp_pred = self.result.bias_corrected_delta(
            np.zeros(3), d_ba
        )

        v_err = np.linalg.norm(truth.delta_v - dv_pred)
        p_err = np.linalg.norm(truth.delta_p - dp_pred)

        self.assertLess(v_err, 1e-6)
        self.assertLess(p_err, 1e-6)


# ============================================================
# Covariance consistency (Monte Carlo)
# ============================================================

class TestCovarianceConsistency(unittest.TestCase):
    """
    The key regression test: does the analytic covariance `P`
    produced by `_propagate_covariance` match the empirical
    covariance of many independently-perturbed integrations?

    This is a statistical test. It uses a fixed seed and a large
    enough trial count that the sampling error on each variance
    estimate is small (relative std error ~ sqrt(2/n_trials)), but
    it is still a Monte Carlo test -- if you shrink n_trials a lot
    or tighten the tolerance a lot, expect occasional flakiness.
    """

    @classmethod
    def setUpClass(cls):
        cls.rng_seed = 12345

        cls.gyro_noise = 0.01
        cls.accel_noise = 0.05
        cls.gyro_random_walk = 0.002
        cls.accel_random_walk = 0.01

        cls.dt = 0.01
        cls.n_steps = 40
        cls.n_trials = 8000

        cls.true_gyro = np.array([0.2, -0.1, 0.05])
        cls.true_accel = np.array([0.3, -9.7, 0.2])

    def _run_trial(self, rng, add_noise):

        pre = IMUPreintegrator(
            gyro_noise=self.gyro_noise,
            accel_noise=self.accel_noise,
            gyro_random_walk=self.gyro_random_walk,
            accel_random_walk=self.accel_random_walk,
        )

        measurements = []
        t = 0.0

        # True (unknown-to-the-filter) bias random walk accumulated
        # over the trial, so that the accel/gyro bias states end up
        # with genuine uncertainty (otherwise the A-matrix blocks
        # that couple bias error into v/p, e.g. A[3:6, 12:15], never
        # get exercised).
        bg_walk = np.zeros(3)
        ba_walk = np.zeros(3)

        for _ in range(self.n_steps + 1):

            if add_noise:
                # Discrete-sample noise std = continuous-time
                # density / sqrt(dt) -- the standard IMU noise
                # convention, and the same one `self.Q` assumes.
                ng = rng.standard_normal(3) * (self.gyro_noise / np.sqrt(self.dt))
                na = rng.standard_normal(3) * (self.accel_noise / np.sqrt(self.dt))

                bg_walk = bg_walk + rng.standard_normal(3) * (
                    self.gyro_random_walk * np.sqrt(self.dt)
                )
                ba_walk = ba_walk + rng.standard_normal(3) * (
                    self.accel_random_walk * np.sqrt(self.dt)
                )
            else:
                ng = na = np.zeros(3)

            measurements.append(
                IMUMeasurement(
                    timestamp=t,
                    gyro=self.true_gyro + ng + bg_walk,
                    accel=self.true_accel + na + ba_walk,
                )
            )

            t += self.dt

        return pre.integrate_measurements(measurements)

    @unittest.skipUnless(
        RUN_SLOW_TESTS,
        "slow Monte Carlo test (~45s); set RUN_SLOW_TESTS=1 to run it",
    )
    def test_covariance_matches_monte_carlo(self):

        rng = np.random.default_rng(self.rng_seed)

        # Noise-free reference trajectory + the analytic covariance
        # the filter believes describes deviations around it.
        nominal = self._run_trial(rng, add_noise=False)
        P_analytic = nominal.covariance

        errors = np.zeros((self.n_trials, 9))

        for k in range(self.n_trials):
            trial = self._run_trial(rng, add_noise=True)

            d_theta = log_so3(nominal.delta_R.T @ trial.delta_R)
            d_v = trial.delta_v - nominal.delta_v
            d_p = trial.delta_p - nominal.delta_p

            errors[k, 0:3] = d_theta
            errors[k, 3:6] = d_v
            errors[k, 6:9] = d_p

        P_mc = np.cov(errors.T)

        analytic_diag = np.diag(P_analytic)[0:9]
        mc_diag = np.diag(P_mc)

        # Relative-error check on the variances (diagonal). With
        # n_trials = 8000, the relative sampling std error on each
        # variance estimate is ~sqrt(2/8000) ~= 1.6%, so a 20%
        # tolerance gives a very comfortable margin while still
        # catching real bugs (e.g. a stray factor of dt, which would
        # be off by orders of magnitude, or a missing coupling term,
        # which typically shows up as a >2x discrepancy).
        rel_error = np.abs(mc_diag - analytic_diag) / analytic_diag

        self.assertTrue(
            np.all(rel_error < 0.20),
            msg=(
                "Analytic covariance diagonal doesn't match Monte Carlo "
                "simulation.\nanalytic: {}\nmonte carlo: {}\n"
                "relative error: {}".format(analytic_diag, mc_diag, rel_error)
            ),
        )

    def test_accel_bias_correlates_with_velocity_and_position(self):
        """
        Regression test for the specific bug where A[3:6, 12:15] and
        A[6:9, 12:15] (direct coupling of accel-bias error into
        velocity/position error) were missing. Without them,
        P[v, ba] and P[p, ba] are structurally stuck at exactly
        zero, which is wrong: accel bias uncertainty must correlate
        with velocity/position error once the bias itself has
        nonzero variance (i.e. once accel_random_walk > 0).
        """

        rng = np.random.default_rng(self.rng_seed)
        nominal = self._run_trial(rng, add_noise=False)

        P = nominal.covariance

        P_v_ba = P[3:6, 12:15]
        P_p_ba = P[6:9, 12:15]

        self.assertGreater(np.max(np.abs(P_v_ba)), 1e-10)
        self.assertGreater(np.max(np.abs(P_p_ba)), 1e-10)


# ============================================================
# Edge cases
# ============================================================

class TestEdgeCases(unittest.TestCase):

    def test_empty_measurement_list_returns_identity(self):
        """
        No measurements at all -> nothing to integrate, everything
        should stay at its reset() defaults.
        """

        pre = IMUPreintegrator()
        result = pre.integrate_measurements([])

        np.testing.assert_allclose(result.delta_R, np.eye(3))
        np.testing.assert_allclose(result.delta_v, np.zeros(3))
        np.testing.assert_allclose(result.delta_p, np.zeros(3))
        self.assertEqual(result.delta_t, 0.0)

    def test_single_measurement_only_sets_clock(self):
        """
        The first sample only initializes `previous_timestamp`; with
        nothing to diff against yet, no integration should occur.
        """

        pre = IMUPreintegrator()
        result = pre.integrate_measurements(
            [IMUMeasurement(timestamp=1.0, gyro=np.ones(3), accel=np.ones(3))]
        )

        np.testing.assert_allclose(result.delta_R, np.eye(3))
        np.testing.assert_allclose(result.delta_v, np.zeros(3))
        np.testing.assert_allclose(result.delta_p, np.zeros(3))
        self.assertEqual(result.delta_t, 0.0)

    def test_non_increasing_timestamp_is_ignored(self):
        """
        A sample with dt <= 0 (duplicate or out-of-order timestamp)
        must be skipped rather than corrupting the integration.
        """

        gyro = np.array([0.2, 0.0, 0.0])
        accel = np.array([0.0, 0.0, 9.8])

        clean = make_measurements(gyro, accel, dt=0.01, n_steps=20)

        # Insert a duplicate-timestamp sample with garbage values in
        # the middle; it must have zero effect on the result. Its
        # timestamp must match the *preceding* sample's timestamp
        # (not the one it displaces) so that dt == 0 when it's
        # processed.
        corrupted = list(clean)
        bad_sample = IMUMeasurement(
            timestamp=corrupted[9].timestamp,  # same as clean[9] -> dt == 0
            gyro=np.array([999.0, 999.0, 999.0]),
            accel=np.array([999.0, 999.0, 999.0]),
        )
        corrupted.insert(10, bad_sample)

        result_clean = IMUPreintegrator().integrate_measurements(clean)
        result_corrupted = IMUPreintegrator().integrate_measurements(corrupted)

        np.testing.assert_allclose(
            result_clean.delta_R, result_corrupted.delta_R, atol=1e-12
        )
        np.testing.assert_allclose(
            result_clean.delta_v, result_corrupted.delta_v, atol=1e-12
        )
        np.testing.assert_allclose(
            result_clean.delta_p, result_corrupted.delta_p, atol=1e-12
        )
        self.assertEqual(result_clean.delta_t, result_corrupted.delta_t)


# ============================================================
# Utility methods
# ============================================================

class TestUtilityMethods(unittest.TestCase):

    def test_reset_clears_accumulated_state(self):
        gyro = np.array([0.3, -0.1, 0.2])
        accel = np.array([0.1, -9.8, 0.0])
        measurements = make_measurements(gyro, accel, dt=0.01, n_steps=50)

        pre = IMUPreintegrator()
        pre.integrate_measurements(measurements)  # leaves accumulated state

        pre.reset()

        np.testing.assert_allclose(pre.delta_R, np.eye(3))
        np.testing.assert_allclose(pre.delta_v, np.zeros(3))
        np.testing.assert_allclose(pre.delta_p, np.zeros(3))
        np.testing.assert_allclose(pre.P, np.zeros((15, 15)))
        np.testing.assert_allclose(pre.J_R_bg, np.zeros((3, 3)))
        self.assertEqual(pre.delta_t, 0.0)
        self.assertIsNone(pre.previous_timestamp)

    def test_copy_is_independent(self):
        gyro = np.array([0.2, 0.1, -0.1])
        accel = np.array([0.0, -9.8, 0.1])
        measurements = make_measurements(gyro, accel, dt=0.01, n_steps=10)

        pre = IMUPreintegrator()
        pre.integrate_measurements(measurements)

        pre_copy = pre.copy()
        pre_copy.integrate(
            IMUMeasurement(timestamp=1000.0, gyro=np.ones(3), accel=np.ones(3))
        )

        # Mutating the copy must not affect the original.
        self.assertNotEqual(pre.delta_t, pre_copy.delta_t)
        self.assertIsNot(pre.delta_R, pre_copy.delta_R)

    def test_set_bias_resets_and_updates_bias(self):
        pre = IMUPreintegrator(bias_g=np.zeros(3), bias_a=np.zeros(3))

        measurements = make_measurements(
            np.array([0.1, 0.0, 0.0]), np.array([0.0, 0.0, 9.8]),
            dt=0.01, n_steps=20,
        )
        pre.integrate_measurements(measurements)

        new_bg = np.array([0.01, -0.02, 0.03])
        new_ba = np.array([0.1, 0.2, -0.1])
        pre.set_bias(new_bg, new_ba)

        np.testing.assert_allclose(pre.bias_g, new_bg)
        np.testing.assert_allclose(pre.bias_a, new_ba)
        # set_bias documents that it resets accumulated preintegration.
        np.testing.assert_allclose(pre.delta_R, np.eye(3))
        self.assertEqual(pre.delta_t, 0.0)


# ============================================================
# Covariance structural invariants (cheap, always run)
# ============================================================

class TestCovarianceInvariants(unittest.TestCase):
    """
    Cheap (non-Monte-Carlo) structural checks on P that should hold
    after any integration: symmetric and positive semi-definite.
    These run every time (unlike the slow MC test) so a structurally
    broken covariance update gets caught immediately.
    """

    def test_covariance_symmetric_and_psd(self):
        gyro = np.array([0.15, -0.1, 0.05])
        accel = np.array([0.2, -9.75, 0.1])
        measurements = make_measurements(gyro, accel, dt=0.005, n_steps=300)

        pre = IMUPreintegrator(
            gyro_noise=0.01, accel_noise=0.05,
            gyro_random_walk=0.001, accel_random_walk=0.005,
        )
        result = pre.integrate_measurements(measurements)

        P = result.covariance

        np.testing.assert_allclose(P, P.T, atol=1e-10)

        eigvals = np.linalg.eigvalsh(P)
        self.assertGreaterEqual(eigvals.min(), -1e-9)


if __name__ == "__main__":
    unittest.main(verbosity=2)