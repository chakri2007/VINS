def calibrate_gyro_bias(visual_quats, preint_gammas, J_gamma_bws):
    A = np.zeros((3*len(preint_gammas), 3))
    b = np.zeros( 3*len(preint_gammas))
    
    for k, (gamma, J_bw) in enumerate(zip(preint_gammas, J_gamma_bws)):
        q_k   = visual_quats[k]
        q_k1  = visual_quats[k+1]
        
        # residual quaternion: should be identity if no bias
        q_res = quat_inv(q_k1) * quat_mult(q_k, gamma)
        
        # extract rotation vector (small angle)
        # q_res ≈ [1, 0.5*theta] for small theta
        theta_res = 2.0 * q_res[1:4]   # vector part
        
        # linearize: theta_res ≈ J_bw * delta_bw
        A[3*k:3*k+3, :] = J_bw
        b[3*k:3*k+3]    = -theta_res
    
    # Solve least squares
    delta_bw, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    return delta_bw