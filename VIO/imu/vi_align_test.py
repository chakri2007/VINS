"""
test_vi_alignment.py

Synthetic ground-truth test for linear VI alignment.

Strategy
--------
1. Generate a known trajectory: rotations, metric positions,
   velocities, and a known gravity vector.
2. Divide the metric positions by a known scale factor to build the
   "visual" (monocular, scale-ambiguous) poses that a VO front-end
   would actually produce.
3. Synthesize *ideal* IMU preintegration deltas (delta_R, delta_v,
   delta_p) directly from the ground-truth metric trajectory and
   gravity, with zero accelerometer bias and zero bias Jacobians.
4. Run collect_alignment_pairs -> build_alignment_system ->
   solve_alignment -> refine_gravity and check that everything
   recovered matches the ground truth.

This isolates the linear-algebra correctness of vi_alignment.py from
the rest of the VIO pipeline (no real front-end / real IMU needed).
"""

from dataclasses import dataclass, field

import numpy as np

from vi_alignment import (
    collect_alignment_pairs,
    build_alignment_system,
    solve_alignment,
    refine_gravity,
    gravity_column,
)


# ============================================================
# Minimal fakes for ViewSet / PreintegratedIMU
# ============================================================

@dataclass
class FakePreintegratedIMU:
    delta_t: float
    delta_R: np.ndarray
    delta_v: np.ndarray
    delta_p: np.ndarray
    J_v_ba: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))
    J_p_ba: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))


class FakeViewSet:
    """
    Minimal stand-in matching the real ViewSet's public API:
        view_set.view_ids
        view_set.get_pose(view_id) -> (R, t)
    """

    def __init__(self, view_ids, poses):
        self.view_ids = list(view_ids)
        self._poses = dict(poses)  # view_id -> (R, t)

    def get_pose(self, view_id):
        return self._poses[view_id]


# ============================================================
# Small rotation helper (axis-angle -> R), no scipy dependency
# ============================================================

def rotation_from_axis_angle(axis, angle):
    axis = axis / np.linalg.norm(axis)
    K = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0],
    ])
    R = (
        np.eye(3)
        + np.sin(angle) * K
        + (1.0 - np.cos(angle)) * (K @ K)
    )
    return R


# ============================================================
# Synthetic trajectory generator
# ============================================================

def generate_synthetic_trajectory(
    num_views=6,
    dt=0.1,
    true_scale=2.5,
    g_true=np.array([0.0, 0.0, -9.81]),
    seed=0,
):
    """
    Returns
    -------
    view_ids : list[int]
    R_list, p_metric_list, v_metric_list : ground truth (world frame)
    p_visual_list : p_metric_list / true_scale  (what a monocular VO
        front-end would actually report)
    imu_preintegrations : dict[(i,j)] -> FakePreintegratedIMU
        built directly from ground truth, zero bias
    """

    rng = np.random.default_rng(seed)

    view_ids = list(range(num_views))

    # Random-ish but smooth-ish rotations and an accelerating path,
    # so the linear system isn't degenerate (e.g. pure straight line
    # would leave scale/gravity poorly observable, same as in real VIO).
    R_list = []
    p_metric_list = []
    v_metric_list = []

    R = np.eye(3)
    p = np.zeros(3)
    v = np.array([0.5, 0.2, 0.1])

    for k in range(num_views):

        R_list.append(R.copy())
        p_metric_list.append(p.copy())
        v_metric_list.append(v.copy())

        # constant-ish jerk to keep velocity/position varying
        accel_world = np.array(
            [0.3 * np.sin(0.7 * k), 0.2 * np.cos(0.5 * k), 0.1]
        )

        # propagate with gravity included (as a real IMU would sense
        # specific force; here we just propagate true kinematics)
        p = p + v * dt + 0.5 * accel_world * dt * dt
        v = v + accel_world * dt

        axis = rng.normal(size=3)
        angle = 0.05 * (k + 1)
        R = R @ rotation_from_axis_angle(axis, angle)

    p_visual_list = [p / true_scale for p in p_metric_list]

    imu_preintegrations = {}

    for k in range(num_views - 1):

        R_i = R_list[k]
        p_i = p_metric_list[k]
        p_j = p_metric_list[k + 1]
        v_i = v_metric_list[k]
        v_j = v_metric_list[k + 1]
        R_j = R_list[k + 1]

        delta_R = R_i.T @ R_j

        delta_v = R_i.T @ (v_j - v_i - g_true * dt)

        delta_p = R_i.T @ (
            p_j - p_i - v_i * dt - 0.5 * g_true * dt * dt
        )

        # Bias Jacobians must be non-zero for the accel-bias columns of
        # A to be observable at all (with true bias = 0 here, the
        # exact values don't affect delta_v/delta_p, only whether the
        # bias block of A has rank). Using the standard first-order
        # approximation -dt*I / -0.5*dt^2*I is enough for that.
        imu_preintegrations[(view_ids[k], view_ids[k + 1])] = (
            FakePreintegratedIMU(
                delta_t=dt,
                delta_R=delta_R,
                delta_v=delta_v,
                delta_p=delta_p,
                J_v_ba=-dt * np.eye(3),
                J_p_ba=-0.5 * dt * dt * np.eye(3),
            )
        )

    return (
        view_ids,
        R_list,
        p_metric_list,
        v_metric_list,
        p_visual_list,
        imu_preintegrations,
    )


# ============================================================
# Test
# ============================================================

def test_linear_vi_alignment_recovers_ground_truth():

    true_scale = 2.5
    g_true = np.array([0.0, 0.0, -9.81])

    (
        view_ids,
        R_list,
        p_metric_list,
        v_metric_list,
        p_visual_list,
        imu_preintegrations,
    ) = generate_synthetic_trajectory(
        num_views=6,
        dt=0.1,
        true_scale=true_scale,
        g_true=g_true,
    )

    poses = {
        view_id: (R_list[k], p_visual_list[k])
        for k, view_id in enumerate(view_ids)
    }

    view_set = FakeViewSet(view_ids, poses)

    pairs = collect_alignment_pairs(view_set, imu_preintegrations)

    assert len(pairs) == len(view_ids) - 1

    A, b = build_alignment_system(pairs, len(view_ids))

    result = solve_alignment(A, b, sorted(view_ids))

    assert result.success, "Linear alignment failed to reach full rank."

    # ---- scale ----
    scale_err = abs(result.scale - true_scale)
    assert scale_err < 1e-6, f"scale error too large: {scale_err}"

    # ---- gravity (direction/magnitude, pre-refinement) ----
    g_err = np.linalg.norm(result.gravity - g_true)
    assert g_err < 1e-4, f"raw gravity error too large: {g_err}"

    # ---- accel bias (should be ~0, no bias injected) ----
    ba_err = np.linalg.norm(result.accel_bias)
    assert ba_err < 1e-6, f"accel bias should be ~0, got {ba_err}"

    # ---- velocities ----
    for k, view_id in enumerate(view_ids):
        v_err = np.linalg.norm(result.velocities[view_id] - v_metric_list[k])
        assert v_err < 1e-6, (
            f"velocity error too large at view {view_id}: {v_err}"
        )

    # ---- gravity refinement should preserve/tighten the estimate ----
    g_refined = refine_gravity(
        A, b, len(view_ids), result.gravity, gravity_magnitude=9.81
    )

    assert abs(np.linalg.norm(g_refined) - 9.81) < 1e-9, (
        "refined gravity magnitude should be exactly 9.81"
    )

    g_refined_err = np.linalg.norm(g_refined - g_true)
    assert g_refined_err < 1e-4, (
        f"refined gravity error too large: {g_refined_err}"
    )

    print("scale error         :", scale_err)
    print("raw gravity error    :", g_err)
    print("refined gravity error:", g_refined_err)
    print("accel bias error     :", ba_err)
    print("ALL CHECKS PASSED")


def test_underdetermined_system_raises():

    (
        view_ids,
        R_list,
        p_metric_list,
        v_metric_list,
        p_visual_list,
        imu_preintegrations,
    ) = generate_synthetic_trajectory(num_views=2, dt=0.1)

    # Only 2 views -> 1 pair -> 6 equations, but 3*2+7=13 unknowns.
    # This must be caught as underdetermined rather than silently
    # returning garbage.

    poses = {
        view_id: (R_list[k], p_visual_list[k])
        for k, view_id in enumerate(view_ids)
    }
    view_set = FakeViewSet(view_ids, poses)

    pairs = collect_alignment_pairs(view_set, imu_preintegrations)
    A, b = build_alignment_system(pairs, len(view_ids))

    raised = False
    try:
        solve_alignment(A, b, sorted(view_ids))
    except RuntimeError:
        raised = True

    assert raised, "Expected RuntimeError for underdetermined system."

    print("underdetermined system correctly raised RuntimeError")


if __name__ == "__main__":
    test_linear_vi_alignment_recovers_ground_truth()
    test_underdetermined_system_raises()