import numpy as np
from typing import List, Tuple

def solve_scale_gravity_velocity(
    visual_poses,    # list of (p_bar_c0_bk, R_c0_bk) — up-to-scale
    alphas,          # preintegrated alpha for each pair (at b_a=0)
    betas,           # preintegrated beta for each pair  (at b_a=0)
    dts,             # time intervals
    p_bc, R_bc,      # camera-IMU extrinsic
    J_a_bas=None,    # [BUGFIX] d(alpha)/d(b_a) per pair, from preintegrate()
    J_v_bas=None,    # [BUGFIX] d(beta)/d(b_a)  per pair, from preintegrate()
):
    """
    Solves for X_I = [v_b0, v_b1, ..., v_bn, g_c0, s, b_a]
    State size: 3*(n+1) + 3 + 1 + 3 = 3n+10

    [BUGFIX] b_a (accelerometer bias) was previously fixed at zero forever
    (never estimated, unlike b_w which IS corrected each VIA run). Any real
    accel bias leaked directly into alpha/beta, and that leak grows with
    window duration -- this is what produced the slow, monotonic gravity-
    magnitude drift seen in the field log (7.10 -> 1.61 m/s² over a run
    with growing keyframe-pair durations).

    preintegrate() already computes alpha(0), beta(0) (evaluated at b_a=0)
    plus the Jacobians J_a_ba = d(alpha)/d(b_a) and J_v_ba = d(beta)/d(b_a).
    First-order correction for nonzero b_a:
        alpha(b_a) ≈ alpha(0) + J_a_ba @ b_a
        beta(b_a)  ≈ beta(0)  + J_v_ba @ b_a
    Moving the b_a term to the LHS (it's now an unknown) gives one extra
    3-column block per pair: column = -J_a_ba (alpha rows), -J_v_ba (beta
    rows). b_a is shared across all pairs in the window (one global bias),
    matching the VINS-Mono initialization convention.

    If J_a_bas/J_v_bas are not supplied, falls back to the old b_a=0,
    not-estimated behaviour (state size 3n+7) for backward compatibility.

    For n keyframe pairs, we get 6*(n) equations (alpha+beta each 3D).
    Need at least 4 pairs for the system to be determined (more once b_a
    is added).
    """
    n_frames = len(visual_poses)
    n_pairs  = n_frames - 1

    estimate_ba = J_a_bas is not None and J_v_bas is not None

    if estimate_ba:
        state_size = 3 * n_frames + 3 + 1 + 3  # velocities + gravity + scale + b_a
    else:
        state_size = 3 * n_frames + 3 + 1      # velocities + gravity + scale

    # Build full system: H * X_I = z
    H_full = np.zeros((6 * n_pairs, state_size))
    z_full = np.zeros( 6 * n_pairs)

    g_idx  = 3 * n_frames
    s_idx  = 3 * n_frames + 3
    ba_idx = 3 * n_frames + 4   # only valid if estimate_ba

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
        # State layout: [v_0(3), v_1(3), ..., v_n(3), g(3), s(1), b_a(3)?]
        # Indices:
        v_k_idx  = 3 * k          # start of v_k in state
        v_k1_idx = 3 * (k+1)      # start of v_{k+1} in state

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

        # [BUGFIX] b_a equation terms — moves J_*_ba @ b_a from z to H
        if estimate_ba:
            J_a_ba = J_a_bas[k]
            J_v_ba = J_v_bas[k]
            H_full[6*k:6*k+3,   ba_idx:ba_idx+3] = -J_a_ba
            H_full[6*k+3:6*k+6, ba_idx:ba_idx+3] = -J_v_ba

    # Solve normal equations
    X_I, _, _, _ = np.linalg.lstsq(H_full, z_full, rcond=None)

    # Extract results
    velocities = [X_I[3*k:3*k+3] for k in range(n_frames)]
    g_c0       = X_I[g_idx:g_idx+3]
    s          = X_I[s_idx]
    delta_b_a  = X_I[ba_idx:ba_idx+3] if estimate_ba else np.zeros(3)

    return s, g_c0, velocities, delta_b_a