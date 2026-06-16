def preintegrate(imu_data, b_a, b_w):
    alpha = np.zeros(3)
    beta  = np.zeros(3)
    gamma = np.array([1., 0., 0., 0.])  # quaternion [w,x,y,z]
    
    # Jacobians wrt bias (for bias correction later)
    J_alpha_ba = np.zeros((3,3))
    J_alpha_bw = np.zeros((3,3))
    J_beta_ba  = np.zeros((3,3))
    J_beta_bw  = np.zeros((3,3))
    J_gamma_bw = np.zeros((3,3))
    
    # Covariance (9x9: alpha, beta, gamma_error)
    P = np.zeros((9,9))
    
    for (dt, accel, gyro) in imu_data:
        a = accel - b_a          # remove bias
        w = gyro  - b_w          # remove bias
        
        R = quat_to_rot(gamma)   # current R^{b_k}_t
        
        # Midpoint integration
        alpha += beta * dt + 0.5 * R @ a * dt**2
        beta  += R @ a * dt
        
        # Quaternion update: integrate angular velocity
        omega_mat = 0.5 * np.array([
            [0,   -w[0], -w[1], -w[2]],
            [w[0],  0,    w[2], -w[1]],
            [w[1], -w[2],  0,    w[0]],
            [w[2],  w[1], -w[0],  0  ]
        ])
        gamma = gamma + omega_mat @ gamma * dt
        gamma = gamma / np.linalg.norm(gamma)  # renormalize
        
        # Update Jacobians (F matrix propagation)
        # F is the 9x9 continuous-time error dynamics
        F = np.zeros((9,9))
        F[0:3, 3:6] = np.eye(3)                    # d_alpha/d_beta
        F[3:6, 6:9] = -R @ skew(a)                 # d_beta/d_theta
        F[6:9, 6:9] = -skew(w)                     # d_theta/d_theta
        
        G = np.zeros((9,6))
        G[3:6, 0:3] = -R                           # accel noise
        G[6:9, 3:6] = -np.eye(3)                   # gyro noise
        
        # Discrete update
        Phi = np.eye(9) + F * dt
        P   = Phi @ P @ Phi.T + (G * dt) @ Q @ (G * dt).T
        
        # Jacobian propagation (simplified)
        J_alpha_ba += J_beta_ba * dt - 0.5 * R * dt**2
        J_alpha_bw += J_beta_bw * dt - 0.5 * R @ skew(a) @ J_gamma_bw * dt**2
        J_beta_ba  += -R * dt
        J_beta_bw  += -R @ skew(a) @ J_gamma_bw * dt
        J_gamma_bw += -skew(w) @ J_gamma_bw * dt + (-np.eye(3)) * dt
    
    return alpha, beta, gamma, \
           J_alpha_ba, J_alpha_bw, \
           J_beta_ba,  J_beta_bw, \
           J_gamma_bw, P