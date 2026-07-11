"""
ceres_bundle_adjustment_motion.py

Ceres port of helperBundleAdjustmentMotion.m ("BA_motion"): a
motion-only, single-frame refinement of the newest view's
[pose, velocity, bias] given

    - the previous keyframe's pose/velocity/bias (held fixed)
    - one IMU preintegration factor between the two views
    - the current frame's 3D-2D correspondences (landmarks held fixed)

plus priors that pull the refined current velocity/bias back toward
their initial guesses -- exactly the factorVelocity3Prior /
factorIMUBiasPrior terms in the MATLAB helper.

This intentionally does NOT touch the sliding-window FactorGraph /
optimize(fg, ...) step -- that is the next, separate phase (full
window smoothing). BA_motion only ever sees two views: "previous" and
"current".

Camera <-> body convention (must match imu/vi_alignment.py exactly):

    p_body = R_bs @ p_camera + t_bs

    R_wb = R_wc @ R_bs.T
    t_wb = C   - R_wb @ t_bs        (body origin, world)
"""

import numpy as np

from optimization.ceres_ba import ceres_ba_motion


class BAMotionResult:
    """Minimal result container, analogous to CeresResult in
    ceres_bundle_adjustment.py."""

    def __init__(self, success, cost, iterations, message):
        self.success = success
        self.cost = cost
        self.iterations = iterations
        self.message = message


def _pose_to_vec(R, C):
    """(3,3) camera-to-world rotation + (3,) camera center -> [rvec, C] (6,)."""
    import cv2
    rvec, _ = cv2.Rodrigues(R)
    return np.concatenate([rvec.flatten(), np.asarray(C, dtype=np.float64).reshape(3)])


def _vec_to_pose(vec):
    """[rvec, C] (6,) -> (R (3,3), C (3,))."""
    import cv2
    vec = np.asarray(vec, dtype=np.float64)
    R, _ = cv2.Rodrigues(vec[0:3])
    C = vec[3:6].copy()
    return R, C


def _pack_bias(bias_g, bias_a):
    return np.concatenate([
        np.asarray(bias_g, dtype=np.float64).reshape(3),
        np.asarray(bias_a, dtype=np.float64).reshape(3),
    ])


def _imu_sqrt_information(covariance, eps=1e-9):
    """
    Dense 15x15 sqrt-information from the preintegration's error-state
    covariance: L such that L @ L.T == inv(covariance).

    A small ridge (eps * I) is added before inversion/Cholesky purely
    for numerical safety (e.g. very short/near-static intervals can
    leave the covariance close to singular in some sub-blocks) -- it
    does not change the physical noise model in any meaningful way.
    """
    cov = np.asarray(covariance, dtype=np.float64)
    cov = 0.5 * (cov + cov.T) + eps * np.eye(cov.shape[0])
    information = np.linalg.inv(cov)
    # np.linalg.cholesky returns lower-triangular Lc with Lc @ Lc.T == information.
    # We want row-major L (any square root works; Ceres just needs
    # L @ L.T == information for correct weighting), so Lc.T works too --
    # use Lc directly, it's already exactly that decomposition.
    Lc = np.linalg.cholesky(information)
    return Lc


def _reprojection_sqrt_info(information_2x2):
    L = np.linalg.cholesky(information_2x2)
    return L


def bundle_adjustment_motion(
    xyz_tracked_in_current_view,      # (N,3) world landmarks (FIXED)
    current_view_correspondences,     # (N,2) pixel observations
    intrinsics_K,                      # (3,3)
    image_size,                        # (H, W)
    current_view_pose_guess,           # (R (3,3), C (3,))
    current_view_velocity_guess,       # (3,)
    previous_view_pose,                # (R (3,3), C (3,))
    previous_view_velocity,            # (3,)
    previous_view_bias,                # (bias_g (3,), bias_a (3,))
    preintegrated_imu,                  # imu.preintegration.PreintegratedIMU
    R_bs,                               # (3,3) camera->body rotation
    t_bs,                               # (3,)  camera->body translation
    gravity=np.array([0.0, 0.0, -9.81]),
    observation_information=None,       # (2,2), default eye(2)
    velocity_prior_sigma=1.0,           # m/s -- loose prior, see VINS-Mono
    bias_gyro_prior_sigma=0.01,         # rad/s
    bias_accel_prior_sigma=0.05,        # m/s^2
    max_iterations=10,
    huber_delta=1.0,
    num_threads=4,
    max_solver_time_in_seconds=0.04,   # VINS-Mono steady-state BA_motion cap
    verbose=False,
):
    """
    Ceres port of helperBundleAdjustmentMotion.m.

    Returns
    -------
    current_pose_refined : (R (3,3), C (3,))
    velocity_refined      : (3,)
    bias_refined          : (bias_g (3,), bias_a (3,))
    valid                 : (N,) bool -- mirrors MATLAB's cheirality +
                             <5px-reprojection-error validity mask
    """

    if observation_information is None:
        observation_information = np.eye(2)

    N = len(current_view_correspondences)

    R_curr_guess, C_curr_guess = current_view_pose_guess
    R_prev, C_prev = previous_view_pose

    current_pose_guess_vec = _pose_to_vec(R_curr_guess, C_curr_guess)
    previous_pose_vec = _pose_to_vec(R_prev, C_prev)

    bias_g_prev, bias_a_prev = previous_view_bias
    previous_bias_vec = _pack_bias(bias_g_prev, bias_a_prev)
    # BA_motion only ever solves for ONE new bias node; the natural
    # starting guess for it is the previous keyframe's bias estimate
    # (constant-bias assumption over one inter-keyframe interval).
    current_bias_guess_vec = previous_bias_vec.copy()

    # ---- IMU preintegration measurement -----------------------------
    p = preintegrated_imu
    delta_R = np.asarray(p.delta_R, dtype=np.float64).reshape(9)
    delta_v = np.asarray(p.delta_v, dtype=np.float64).reshape(3)
    delta_p = np.asarray(p.delta_p, dtype=np.float64).reshape(3)
    delta_t = float(p.delta_t)
    bias_lin = _pack_bias(p.bias_g, p.bias_a)

    cov_diag = np.asarray(p.covariance, dtype=np.float64).diagonal()
    cov_cond = np.linalg.cond(np.asarray(p.covariance, dtype=np.float64))
    print(f"[BA_motion debug] N={N} delta_t={delta_t:.6f} "
          f"cov_diag_min={cov_diag.min():.3e} cov_diag_max={cov_diag.max():.3e} "
          f"cov_cond={cov_cond:.3e}")

    J_R_bg = np.asarray(p.J_R_bg, dtype=np.float64).reshape(9)
    J_v_bg = np.asarray(p.J_v_bg, dtype=np.float64).reshape(9)
    J_v_ba = np.asarray(p.J_v_ba, dtype=np.float64).reshape(9)
    J_p_bg = np.asarray(p.J_p_bg, dtype=np.float64).reshape(9)
    J_p_ba = np.asarray(p.J_p_ba, dtype=np.float64).reshape(9)

    imu_sqrt_info = _imu_sqrt_information(p.covariance).reshape(225)

    # ---- reprojection observations (landmarks fixed) ------------------
    L_obs = _reprojection_sqrt_info(observation_information)

    observations = []
    for i in range(N):
        obs = ceres_ba_motion.MotionObservation()
        obs.u = float(current_view_correspondences[i, 0])
        obs.v = float(current_view_correspondences[i, 1])
        obs.L00 = float(L_obs[0, 0])
        obs.L01 = float(L_obs[0, 1])
        obs.L10 = float(L_obs[1, 0])
        obs.L11 = float(L_obs[1, 1])
        obs.xyz = [float(x) for x in xyz_tracked_in_current_view[i]]
        observations.append(obs)

    K = np.asarray(intrinsics_K, dtype=np.float64)
    K_vec = [float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])]

    velocity_prior_sqrt_info = (1.0 / velocity_prior_sigma) * np.eye(3)
    bias_prior_sqrt_info = np.diag([
        1.0 / bias_gyro_prior_sigma,
        1.0 / bias_gyro_prior_sigma,
        1.0 / bias_gyro_prior_sigma,
        1.0 / bias_accel_prior_sigma,
        1.0 / bias_accel_prior_sigma,
        1.0 / bias_accel_prior_sigma,
    ])

    R_bs_arr = np.asarray(R_bs, dtype=np.float64).reshape(9)
    t_bs_arr = np.asarray(t_bs, dtype=np.float64).reshape(3)

    result = ceres_ba_motion.solve_bundle_adjustment_motion(
        current_pose_guess=list(current_pose_guess_vec),
        current_velocity_guess=list(np.asarray(current_view_velocity_guess, dtype=np.float64)),
        current_bias_guess=list(current_bias_guess_vec),
        previous_pose=list(previous_pose_vec),
        previous_velocity=list(np.asarray(previous_view_velocity, dtype=np.float64)),
        previous_bias=list(previous_bias_vec),
        R_bs=list(R_bs_arr),
        t_bs=list(t_bs_arr),
        delta_R=list(delta_R),
        delta_v=list(delta_v),
        delta_p=list(delta_p),
        delta_t=delta_t,
        bias_lin=list(bias_lin),
        J_R_bg=list(J_R_bg),
        J_v_bg=list(J_v_bg),
        J_v_ba=list(J_v_ba),
        J_p_bg=list(J_p_bg),
        J_p_ba=list(J_p_ba),
        gravity=list(np.asarray(gravity, dtype=np.float64)),
        imu_sqrt_information=list(imu_sqrt_info),
        observations=observations,
        K_vec=K_vec,
        velocity_prior_sqrt_info=list(velocity_prior_sqrt_info.reshape(9)),
        bias_prior_sqrt_info=list(bias_prior_sqrt_info.reshape(36)),
        max_iterations=max_iterations,
        verbose=verbose,
        huber_delta=huber_delta,
        num_threads=num_threads,
        max_solver_time_in_seconds=max_solver_time_in_seconds,
    )

    print("\n========== BUNDLE ADJUSTMENT MOTION (Ceres) ==========")
    print(f"Correspondences : {N}")
    print(f"Cost            : {result['initial_cost']:.4f} -> {result['final_cost']:.4f}")
    print(f"Iterations      : {result['iterations']}")
    print(f"Termination     : {result['termination']}")
    print("========================================================\n")

    if not result["success"] or result["termination"] not in ("CONVERGENCE", "USER_SUCCESS"):
        print(f"[BA_motion] Rejecting solve ({result['termination']}): {result['message']}")
        return None, None, None, None

    R_refined, C_refined = _vec_to_pose(result["pose"])
    velocity_refined = np.asarray(result["velocity"], dtype=np.float64)
    bias_refined_vec = np.asarray(result["bias"], dtype=np.float64)
    bias_refined = (bias_refined_vec[0:3].copy(), bias_refined_vec[3:6].copy())

    # ---- validity mask -------------------------------------------------
    # Mirrors the tail of helperBundleAdjustmentMotion.m: reproject with
    # the REFINED pose, keep points in front of the camera, inside the
    # image, and within 5px of the observed pixel.
    xyz = np.asarray(xyz_tracked_in_current_view, dtype=np.float64)
    uv_obs = np.asarray(current_view_correspondences, dtype=np.float64)

    pc = (xyz - C_refined) @ R_refined  # (N,3), == R_refined.T @ (xyz - C) row-wise
    z = pc[:, 2]
    in_front = z > 0

    z_safe = np.where(in_front, z, 1.0)
    uv_pred = np.column_stack([
        K[0, 0] * pc[:, 0] / z_safe + K[0, 2],
        K[1, 1] * pc[:, 1] / z_safe + K[1, 2],
    ])

    H, W = image_size[0], image_size[1]
    in_image = (
        (uv_pred[:, 0] >= 0) & (uv_pred[:, 0] <= W) &
        (uv_pred[:, 1] >= 0) & (uv_pred[:, 1] <= H)
    )

    reproj_err = np.linalg.norm(uv_obs - uv_pred, axis=1)
    valid = in_front & in_image & (reproj_err < 5.0)

    return (R_refined, C_refined), velocity_refined, bias_refined, valid
