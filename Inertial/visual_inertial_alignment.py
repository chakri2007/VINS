class VisualInertialAlignment:
    def __init__(self, T_bc, imu_noise_params):
        """
        T_bc: 4x4 camera-to-IMU extrinsic transform
        imu_noise_params: dict with sigma_a, sigma_w, sigma_ba, sigma_bw
        """
        self.R_bc = T_bc[:3,:3]
        self.p_bc = T_bc[:3, 3]
        self.noise = imu_noise_params
        
        self.b_w = np.zeros(3)
        self.b_a = np.zeros(3)
    
    def run(self, keyframe_poses, imu_segments):
        """
        keyframe_poses: list of (p_bar, R) — up-to-scale from your VO
                        p_bar in visual units, R exact
        imu_segments:   list of IMU data between consecutive keyframes
                        each = list of (dt, accel(3,), gyro(3,))
        
        Returns:
            s          — metric scale
            g_world    — gravity vector
            velocities — per-keyframe velocity
        """
        assert len(imu_segments) == len(keyframe_poses) - 1, \
            "Need one IMU segment per consecutive keyframe pair"
        assert len(keyframe_poses) >= 5, \
            "Need at least 5 keyframes for reliable initialization"
        
        # ── Step 1: Preintegrate all IMU segments ──────────────
        preint_results = []
        for segment in imu_segments:
            result = preintegrate(segment, self.b_a, self.b_w)
            preint_results.append(result)
        
        alphas      = [r[0] for r in preint_results]
        betas       = [r[1] for r in preint_results]
        gammas      = [r[2] for r in preint_results]
        J_gamma_bws = [r[7] for r in preint_results]
        dts         = [sum(d for d,a,g in seg) for seg in imu_segments]
        
        # ── Step 2: Calibrate gyroscope bias ───────────────────
        visual_quats = [pose_to_quat(R) for _, R in keyframe_poses]
        delta_bw = calibrate_gyro_bias(visual_quats, gammas, J_gamma_bws)
        self.b_w += delta_bw
        
        # Re-preintegrate with corrected bias
        preint_results = [preintegrate(seg, self.b_a, self.b_w) 
                         for seg in imu_segments]
        alphas = [r[0] for r in preint_results]
        betas  = [r[1] for r in preint_results]
        
        # ── Step 3: Solve linear system ────────────────────────
        s, g_c0, velocities = solve_scale_gravity_velocity(
            keyframe_poses, alphas, betas, dts,
            self.p_bc, self.R_bc
        )
        
        # Sanity checks
        assert s > 0,          "Scale must be positive"
        assert 9.0 < np.linalg.norm(g_c0) < 10.5, \
            f"Gravity magnitude {np.linalg.norm(g_c0):.2f} unreasonable"
        
        # ── Step 4: Gravity refinement ─────────────────────────
        g_c0 = refine_gravity(g_c0, keyframe_poses, 
                              alphas, betas, dts, self.p_bc, velocities)
        
        return s, g_c0, velocities
    
    def scale_visual_map(self, s, keyframe_poses, map_points):
        """
        After alignment, scale everything for your BA
        """
        scaled_poses = [(s * p, R) for p, R in keyframe_poses]
        scaled_points = [s * X for X in map_points]
        return scaled_poses, scaled_points