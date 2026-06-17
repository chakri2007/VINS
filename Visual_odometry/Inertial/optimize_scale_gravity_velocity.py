import numpy as np
from typing import List, Tuple

def solve_scale_gravity_velocity(
    visual_poses,    # list of (p_bar_c0_bk, R_c0_bk) — up-to-scale
    alphas,          # preintegrated alpha for each pair
    betas,           # preintegrated beta for each pair
    dts,             # time intervals
    p_bc, R_bc       # camera-IMU extrinsic
):
    """
    Solves for X_I = [v_b0, v_b1, ..., v_bn, g_c0, s]
    State size: 3*(n+1) + 3 + 1 = 3n+7
    
    For n keyframe pairs, we get 6*(n) equations (alpha+beta each 3D)
    Need at least 4 pairs for the system to be determined.
    """
    n_frames = len(visual_poses)
    n_pairs  = n_frames - 1
    
    state_size = 3 * n_frames + 3 + 1  # velocities + gravity + scale
    
    # Build full system: H * X_I = z
    H_full = np.zeros((6 * n_pairs, state_size))
    z_full = np.zeros( 6 * n_pairs)
    
    for k in range(n_pairs):
        p_k,  R_k  = visual_poses[k]      # up-to-scale position and rotation
        p_k1, R_k1 = visual_poses[k+1]
        alpha = alphas[k]
        beta  = betas[k]
        dt    = dts[k]
        
        # R^{bk}_{c0} = (R^{c0}_{bk})^T
        R_bk_c0  = R_k.T
        R_bk1_c0 = R_k1.T
        
        # Measurement vector z (from VINS-Mono eq 18):
        # z = [alpha - p_bc + R_bk_c0 @ R_c0_bk1 @ p_bc,
        #      beta                                      ]
        z_alpha = alpha - p_bc + R_bk_c0 @ R_k1 @ p_bc
        z_beta  = beta
        
        z_full[6*k:6*k+3] = z_alpha
        z_full[6*k+3:6*k+6] = z_beta
        
        # ── Build H block for this pair ──────────────────────────
        # State layout: [v_0(3), v_1(3), ..., v_n(3), g(3), s(1)]
        # Indices:
        v_k_idx  = 3 * k          # start of v_k in state
        v_k1_idx = 3 * (k+1)      # start of v_{k+1} in state
        g_idx    = 3 * n_frames   # start of g in state
        s_idx    = 3 * n_frames + 3  # index of s
        
        # From VINS-Mono eq (19):
        # H * X = z
        #
        # alpha equation (row 0:3):
        # -I*dt * v_k  +  0 * v_k1  +  0.5*R_bk_c0*dt² * g  +  R_bk_c0*(p_k1-p_k) * s  = z_alpha
        H_full[6*k:6*k+3, v_k_idx:v_k_idx+3]   = -np.eye(3) * dt
        H_full[6*k:6*k+3, v_k1_idx:v_k1_idx+3] =  np.zeros((3,3))
        H_full[6*k:6*k+3, g_idx:g_idx+3]        =  0.5 * R_bk_c0 * dt**2
        H_full[6*k:6*k+3, s_idx]                =  R_bk_c0 @ (p_k1 - p_k)
        
        # beta equation (row 3:6):
        # -I * v_k  +  R_bk_c0*R_c0_bk1 * v_k1  +  R_bk_c0*dt * g  +  0 * s  = z_beta
        H_full[6*k+3:6*k+6, v_k_idx:v_k_idx+3]   = -np.eye(3)
        H_full[6*k+3:6*k+6, v_k1_idx:v_k1_idx+3] =  R_bk_c0 @ R_k1
        H_full[6*k+3:6*k+6, g_idx:g_idx+3]        =  R_bk_c0 * dt
        H_full[6*k+3:6*k+6, s_idx]                =  np.zeros(3)
    
    # Solve normal equations
    X_I, _, _, _ = np.linalg.lstsq(H_full, z_full, rcond=None)
    
    # Extract results
    velocities = [X_I[3*k:3*k+3] for k in range(n_frames)]
    g_c0       = X_I[g_idx:g_idx+3]
    s          = X_I[s_idx]
    
    return s, g_c0, velocities