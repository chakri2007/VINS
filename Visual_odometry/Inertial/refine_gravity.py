import numpy as np
from typing import List, Tuple


def refine_gravity(g_c0_init, visual_poses, alphas, betas, 
                   dts, p_bc, velocities, n_iter=4):
    G_MAGNITUDE = 9.81
    
    g_hat = g_c0_init / np.linalg.norm(g_c0_init)
    
    for iteration in range(n_iter):
        # Build tangent basis b1, b2 at g_hat
        b1, b2 = tangent_basis(g_hat)  # two orthonormal vectors
        
        # Rebuild linear system but now g = G_MAGNITUDE*g_hat + w1*b1 + w2*b2
        # unknowns are now [velocities..., w1, w2]
        # This is the same H matrix but gravity columns replaced
        # g_column * [w1, w2] replaces g_column * g
        
        n_frames = len(visual_poses)
        n_pairs  = n_frames - 1
        state_size = 3 * n_frames + 2  # velocities + w1 + w2
        
        H = np.zeros((6*n_pairs, state_size))
        z = np.zeros( 6*n_pairs)
        
        for k in range(n_pairs):
            p_k,  R_k  = visual_poses[k]
            p_k1, R_k1 = visual_poses[k+1]
            dt = dts[k]
            R_bk_c0 = R_k.T
            
            v_k_idx  = 3 * k
            v_k1_idx = 3 * (k+1)
            w_idx    = 3 * n_frames  # w1, w2 start here
            
            # gravity contribution to measurement (known part):
            g_known = G_MAGNITUDE * g_hat
            
            # alpha row
            z_alpha = alphas[k] - p_bc + R_bk_c0 @ R_k1 @ p_bc
            z_alpha -= 0.5 * R_bk_c0 @ g_known * dt**2   # subtract known g part
            # add back visual scale (use current s estimate)
            # ... (same structure as before)
            
            H[6*k:6*k+3, v_k_idx:v_k_idx+3] = -np.eye(3) * dt
            H[6*k:6*k+3, w_idx]   = 0.5 * R_bk_c0 @ b1 * dt**2
            H[6*k:6*k+3, w_idx+1] = 0.5 * R_bk_c0 @ b2 * dt**2
            
            # beta row
            z_beta = betas[k]
            z_beta -= R_bk_c0 @ g_known * dt
            
            H[6*k+3:6*k+6, v_k_idx:v_k_idx+3]   = -np.eye(3)
            H[6*k+3:6*k+6, v_k1_idx:v_k1_idx+3] =  R_bk_c0 @ R_k1
            H[6*k+3:6*k+6, w_idx]   = R_bk_c0 @ b1 * dt
            H[6*k+3:6*k+6, w_idx+1] = R_bk_c0 @ b2 * dt
            
            z[6*k:6*k+3]   = z_alpha
            z[6*k+3:6*k+6] = z_beta
        
        X, _, _, _ = np.linalg.lstsq(H, z, rcond=None)
        w1, w2 = X[-2], X[-1]
        
        # Update g_hat
        g_c0 = G_MAGNITUDE * g_hat + w1 * b1 + w2 * b2
        g_hat = g_c0 / np.linalg.norm(g_c0)
    
    return g_hat * G_MAGNITUDE

def tangent_basis(g_hat):
    """Two orthonormal vectors perpendicular to g_hat"""
    if abs(g_hat[0]) < 0.9:
        tmp = np.array([1., 0., 0.])
    else:
        tmp = np.array([0., 0., 1.])
    b1 = np.cross(g_hat, tmp)
    b1 /= np.linalg.norm(b1)
    b2 = np.cross(g_hat, b1)
    return b1, b2