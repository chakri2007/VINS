import numpy as np
from typing import List, Tuple

def solve_scale_gravity_velocity(
    visual_poses,          # list of (p_bar_c0_bk, R_c0_bk)
    alphas, betas, dts,
    p_bc, R_bc,
    J_alpha_bas=None,      # List of J_a_ba (d alpha / d b_a) per pair
    J_beta_bas=None,       # List of J_v_ba (d beta / d b_a) per pair
):
    """
    Extended solver: joint estimation of velocities, g_c0, scale s, and shared b_a (3 DoF).
    """
    n_frames = len(visual_poses)
    n_pairs  = n_frames - 1
    
    # State: [v0(3), v1(3), ..., vn(3), g(3), s(1), ba(3)]
    state_size = 3 * n_frames + 3 + 1 + 3
    ba_idx = 3 * n_frames + 3 + 1  # start of ba block
    
    H_full = np.zeros((6 * n_pairs, state_size))
    z_full = np.zeros(6 * n_pairs)
    
    for k in range(n_pairs):
        p_k,  R_k  = visual_poses[k]
        p_k1, R_k1 = visual_poses[k+1]
        alpha = alphas[k]
        beta  = betas[k]
        dt    = dts[k]
        
        R_bk_c0  = R_k.T
        R_bk1_c0 = R_k1.T
        
        z_alpha = alpha - p_bc + R_bk_c0 @ R_k1 @ p_bc
        z_beta  = beta
        
        z_full[6*k:6*k+3] = z_alpha
        z_full[6*k+3:6*k+6] = z_beta
        
        v_k_idx  = 3 * k
        v_k1_idx = 3 * (k + 1)
        g_idx    = 3 * n_frames
        s_idx    = g_idx + 3
        
        # Alpha block (rows 0:3)
        H_full[6*k:6*k+3, v_k_idx:v_k_idx+3]   = -np.eye(3) * dt
        H_full[6*k:6*k+3, v_k1_idx:v_k1_idx+3] = np.zeros((3,3))
        H_full[6*k:6*k+3, g_idx:g_idx+3]        = 0.5 * R_bk_c0 * dt**2
        H_full[6*k:6*k+3, s_idx]                = R_bk_c0 @ (p_k1 - p_k)
        
        # Beta block (rows 3:6)
        H_full[6*k+3:6*k+6, v_k_idx:v_k_idx+3]   = -np.eye(3)
        H_full[6*k+3:6*k+6, v_k1_idx:v_k1_idx+3] = R_bk_c0 @ R_k1
        H_full[6*k+3:6*k+6, g_idx:g_idx+3]        = R_bk_c0 * dt
        H_full[6*k+3:6*k+6, s_idx]                = np.zeros(3)
        
        # === NEW: b_a contributions using Jacobians ===
        if J_alpha_bas is not None and J_beta_bas is not None:
            J_a_ba = J_alpha_bas[k]
            J_v_ba = J_beta_bas[k]
            H_full[6*k:6*k+3, ba_idx:ba_idx+3]     = -J_a_ba
            H_full[6*k+3:6*k+6, ba_idx:ba_idx+3]   = -J_v_ba
    
    # Solve
    X_I, _, _, _ = np.linalg.lstsq(H_full, z_full, rcond=None)
    
    velocities = [X_I[3*k:3*k+3] for k in range(n_frames)]
    g_c0       = X_I[g_idx:g_idx+3]
    s          = X_I[s_idx]
    delta_ba   = X_I[ba_idx:ba_idx+3]
    
    return s, g_c0, velocities, delta_ba