import numpy as np
from typing import List, Optional

def solve_scale_gravity_velocity(
    visual_poses,
    alphas,
    betas,
    dts,
    p_bc,
    R_bc,
    J_alpha_bas: Optional[List[np.ndarray]] = None,
    J_beta_bas: Optional[List[np.ndarray]] = None,
):
    """
    Joint estimation of velocities, gravity, scale, and shared accelerometer bias b_a.
    """
    n_frames = len(visual_poses)
    n_pairs = n_frames - 1

    # State vector: [v0(3), v1(3), ..., vn(3), g(3), s(1), ba(3)]
    state_size = 3 * n_frames + 3 + 1 + 3
    ba_idx = 3 * n_frames + 3 + 1

    H = np.zeros((6 * n_pairs, state_size))
    z = np.zeros(6 * n_pairs)

    for k in range(n_pairs):
        p_k, R_k = visual_poses[k]
        p_k1, R_k1 = visual_poses[k + 1]
        alpha = alphas[k]
        beta = betas[k]
        dt = dts[k]

        R_bk_c0 = R_k.T
        R_bk1_c0 = R_k1.T

        z_alpha = alpha - p_bc + R_bk_c0 @ R_k1 @ p_bc
        z_beta = beta

        z[6*k:6*k+3] = z_alpha
        z[6*k+3:6*k+6] = z_beta

        v_k_idx = 3 * k
        v_k1_idx = 3 * (k + 1)
        g_idx = 3 * n_frames
        s_idx = g_idx + 3

        # Alpha equations
        H[6*k:6*k+3, v_k_idx:v_k_idx+3] = -np.eye(3) * dt
        H[6*k:6*k+3, g_idx:g_idx+3] = 0.5 * R_bk_c0 * dt**2
        H[6*k:6*k+3, s_idx] = R_bk_c0 @ (p_k1 - p_k)

        # Beta equations
        H[6*k+3:6*k+6, v_k_idx:v_k_idx+3] = -np.eye(3)
        H[6*k+3:6*k+6, v_k1_idx:v_k1_idx+3] = R_bk_c0 @ R_k1
        H[6*k+3:6*k+6, g_idx:g_idx+3] = R_bk_c0 * dt

        # Accelerometer bias contribution
        if J_alpha_bas is not None and J_beta_bas is not None:
            H[6*k:6*k+3, ba_idx:ba_idx+3] = -J_alpha_bas[k]
            H[6*k+3:6*k+6, ba_idx:ba_idx+3] = -J_beta_bas[k]

    # Least squares solution
    X, _, _, _ = np.linalg.lstsq(H, z, rcond=None)

    velocities = [X[3*i:3*i+3] for i in range(n_frames)]
    g_c0 = X[g_idx:g_idx+3]
    s = float(X[s_idx])
    delta_ba = X[ba_idx:ba_idx+3]

    return s, g_c0, velocities, delta_ba