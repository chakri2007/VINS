// ceres_ba_motion.cpp
//
// Motion-only bundle adjustment (single-frame BA), the Ceres port of
// helperBundleAdjustmentMotion.m.
//
// Given:
//   - the previous keyframe's pose/velocity/bias (FIXED — baked in as
//     constants, never registered as Ceres parameter blocks, exactly
//     like the MATLAB helper's f.fixNode() on the previous-view nodes)
//   - the current frame's pose/velocity/bias guesses (FREE — the only
//     variables actually being solved for)
//   - one IMU preintegration factor between the two views
//   - the current frame's 3D-2D correspondences (landmarks FIXED)
//   - priors pulling the refined current velocity/bias back toward
//     their initial guesses, matching factorVelocity3Prior /
//     factorIMUBiasPrior in the MATLAB helper
//
// this refines ONLY the current view's [rvec, C, vel, bias].
//
// ─────────────────────────────────────────────────────────────────────
// Conventions (must match the rest of this repo exactly):
//
//   Camera pose:  p_cam = R_wc^T @ (p_world - C),  i.e. (R_wc, C) is
//                 camera-to-world, C is the camera center in world
//                 coordinates (see camera_factor.py / ceres_ba.cpp).
//
//   Camera->body extrinsic (T_BS), matching imu/vi_alignment.py:
//                 p_body = R_bs @ p_camera + t_bs
//
//                 R_wb = R_wc @ R_bs^T
//                 t_wb = C   - R_wb @ t_bs        (body origin, world)
//
//   SO(3) exp/log follow the exact Rodrigues formulas used in
//   imu/math_utils.py (exp_so3 / log_so3 / skew) — reimplemented here
//   by hand (rather than relying on Ceres' AngleAxis* helpers, whose
//   internal storage-order assumptions are easy to get backwards) so
//   the two implementations are provably in lock-step.
// ─────────────────────────────────────────────────────────────────────

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <ceres/ceres.h>

#include <vector>
#include <array>
#include <cmath>

namespace py = pybind11;


// ─────────────────────────────────────────────────────────────────────
// Small templated SO(3) / mat3 helpers (row-major 3x3: M[3*r+c])
// ─────────────────────────────────────────────────────────────────────

template <typename T>
void mat3_identity(T M[9]) {
    for (int i = 0; i < 9; ++i) M[i] = T(0);
    M[0] = M[4] = M[8] = T(1);
}

template <typename T>
void mat3_mul(const T A[9], const T B[9], T C[9]) {
    // C = A @ B, row-major
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            T s = T(0);
            for (int k = 0; k < 3; ++k) s += A[3 * r + k] * B[3 * k + c];
            C[3 * r + c] = s;
        }
    }
}

template <typename T>
void mat3_transpose(const T A[9], T At[9]) {
    for (int r = 0; r < 3; ++r)
        for (int c = 0; c < 3; ++c)
            At[3 * c + r] = A[3 * r + c];
}

template <typename T>
void mat3_vec(const T A[9], const T v[3], T out[3]) {
    for (int r = 0; r < 3; ++r) {
        out[r] = A[3 * r + 0] * v[0] + A[3 * r + 1] * v[1] + A[3 * r + 2] * v[2];
    }
}

template <typename T>
void skew3(const T v[3], T S[9]) {
    S[0] = T(0);   S[1] = -v[2];  S[2] =  v[1];
    S[3] =  v[2];  S[4] = T(0);   S[5] = -v[0];
    S[6] = -v[1];  S[7] =  v[0];  S[8] = T(0);
}

// Rodrigues exponential map: angle-axis (3,) -> rotation matrix (row-major 9,)
// Mirrors imu/math_utils.py::exp_so3 exactly.
template <typename T>
void exp_so3(const T phi[3], T R[9]) {
    T theta2 = phi[0] * phi[0] + phi[1] * phi[1] + phi[2] * phi[2];
    T theta = sqrt(theta2);

    T W[9];
    skew3(phi, W);

    T WW[9];
    mat3_mul(W, W, WW);

    T A, B;
    // Use a smooth (auto-diff safe) small-angle fallback. 1e-12 threshold
    // matches EPS in math_utils.py; below that, use the Taylor series
    // (I + W + 0.5 W^2), consistent with exp_so3's theta<EPS branch.
    if (theta < T(1e-8)) {
        A = T(1.0);
        B = T(0.5);
    } else {
        A = sin(theta) / theta;
        B = (T(1.0) - cos(theta)) / theta2;
    }

    mat3_identity(R);
    for (int i = 0; i < 9; ++i) R[i] += A * W[i] + B * WW[i];
}

// SO(3) logarithm: rotation matrix (row-major 9,) -> angle-axis (3,)
// Mirrors imu/math_utils.py::log_so3 exactly (including the
// near-identity vee(R - R^T)/2 fallback).
template <typename T>
void log_so3(const T R[9], T phi[3]) {
    T trace = R[0] + R[4] + R[8];
    T cos_theta = (trace - T(1.0)) * T(0.5);

    // clamp to [-1, 1]
    cos_theta = ceres::fmin(T(1.0), ceres::fmax(T(-1.0), cos_theta));

    T theta = acos(cos_theta);

    // vee(R - R^T) = [R32-R23, R13-R31, R21-R12]
    T vee_x = R[7] - R[5];
    T vee_y = R[2] - R[6];
    T vee_z = R[3] - R[1];

    if (theta < T(1e-8)) {
        phi[0] = T(0.5) * vee_x;
        phi[1] = T(0.5) * vee_y;
        phi[2] = T(0.5) * vee_z;
        return;
    }

    T scale = theta / (T(2.0) * sin(theta));
    phi[0] = scale * vee_x;
    phi[1] = scale * vee_y;
    phi[2] = scale * vee_z;
}


// ─────────────────────────────────────────────────────────────────────
// IMU preintegration factor between a FIXED previous state and the
// FREE current state [rvec_curr(3), C_curr(3), vel_curr(3), bias_curr(6)].
//
// 15-dim residual, ordered [rotation(3), velocity(3), position(3),
// gyro-bias-walk(3), accel-bias-walk(3)], weighted by the dense
// sqrt-information (15x15, row-major) derived in Python from the
// preintegration covariance — i.e. the same structure as MATLAB's
// factorIMU / GTSAM's CombinedImuFactor.
// ─────────────────────────────────────────────────────────────────────
struct IMUPreintegrationError {

    IMUPreintegrationError(
        // fixed previous state (already baked to doubles)
        const double R_wb_prev[9], const double t_wb_prev[3],
        const double vel_prev[3],
        const double bias_prev[6],           // [bg(3), ba(3)]
        // camera->body extrinsic (fixed)
        const double R_bs[9], const double t_bs[3],
        // preintegration measurement (linearized at bias_lin)
        const double delta_R[9], const double delta_v[3], const double delta_p[3],
        double delta_t,
        const double bias_lin[6],             // linearization point [bg,ba]
        const double J_R_bg[9], const double J_v_bg[9], const double J_v_ba[9],
        const double J_p_bg[9], const double J_p_ba[9],
        const double gravity[3],
        const double sqrt_information[225]    // row-major 15x15
    ) : delta_t_(delta_t) {
        for (int i = 0; i < 9; ++i) {
            R_wb_prev_[i] = R_wb_prev[i];
            R_bs_[i] = R_bs[i];
            delta_R_[i] = delta_R[i];
            J_R_bg_[i] = J_R_bg[i];
            J_v_bg_[i] = J_v_bg[i];
            J_v_ba_[i] = J_v_ba[i];
            J_p_bg_[i] = J_p_bg[i];
            J_p_ba_[i] = J_p_ba[i];
        }
        for (int i = 0; i < 3; ++i) {
            t_wb_prev_[i] = t_wb_prev[i];
            vel_prev_[i] = vel_prev[i];
            t_bs_[i] = t_bs[i];
            delta_v_[i] = delta_v[i];
            delta_p_[i] = delta_p[i];
            gravity_[i] = gravity[i];
        }
        for (int i = 0; i < 6; ++i) {
            bias_prev_[i] = bias_prev[i];
            bias_lin_[i] = bias_lin[i];
        }
        for (int i = 0; i < 225; ++i) sqrt_information_[i] = sqrt_information[i];
    }

    template <typename T>
    bool operator()(
        const T* const rvec_curr,
        const T* const C_curr,
        const T* const vel_curr,
        const T* const bias_curr,     // [bg(3), ba(3)]
        T* residuals
    ) const {

        // ---- cast all fixed double[] constants to T[] once, up front ----
        T R_bs_T_cast[9], t_bs_cast[3];
        {
            T R_bs_cast[9];
            for (int i = 0; i < 9; ++i) R_bs_cast[i] = T(R_bs_[i]);
            mat3_transpose(R_bs_cast, R_bs_T_cast);
            for (int i = 0; i < 3; ++i) t_bs_cast[i] = T(t_bs_[i]);
        }
        T J_R_bg_cast[9], J_v_bg_cast[9], J_v_ba_cast[9], J_p_bg_cast[9], J_p_ba_cast[9];
        T delta_R_cast[9];
        T R_wb_prev_cast[9];
        for (int i = 0; i < 9; ++i) {
            J_R_bg_cast[i] = T(J_R_bg_[i]);
            J_v_bg_cast[i] = T(J_v_bg_[i]);
            J_v_ba_cast[i] = T(J_v_ba_[i]);
            J_p_bg_cast[i] = T(J_p_bg_[i]);
            J_p_ba_cast[i] = T(J_p_ba_[i]);
            delta_R_cast[i] = T(delta_R_[i]);
            R_wb_prev_cast[i] = T(R_wb_prev_[i]);
        }

        // ---- current body pose from current camera pose -------------
        T R_wc_curr[9];
        exp_so3(rvec_curr, R_wc_curr);

        T R_wb_curr[9];
        mat3_mul(R_wc_curr, R_bs_T_cast, R_wb_curr);

        T R_wb_curr_tbs[3];
        mat3_vec(R_wb_curr, t_bs_cast, R_wb_curr_tbs);

        T t_wb_curr[3];
        for (int i = 0; i < 3; ++i) t_wb_curr[i] = C_curr[i] - R_wb_curr_tbs[i];

        // ---- bias correction (first-order, about bias_lin) -----------
        T d_bg[3], d_ba[3];
        for (int i = 0; i < 3; ++i) {
            d_bg[i] = bias_curr[i]     - T(bias_lin_[i]);
            d_ba[i] = bias_curr[3 + i] - T(bias_lin_[3 + i]);
        }

        T J_R_bg_dbg[3];
        mat3_vec(J_R_bg_cast, d_bg, J_R_bg_dbg);

        T dR_corr_small[9];
        exp_so3(J_R_bg_dbg, dR_corr_small);

        T delta_R_corrected[9];
        mat3_mul(delta_R_cast, dR_corr_small, delta_R_corrected);

        T J_v_bg_dbg[3], J_v_ba_dba[3];
        mat3_vec(J_v_bg_cast, d_bg, J_v_bg_dbg);
        mat3_vec(J_v_ba_cast, d_ba, J_v_ba_dba);

        T delta_v_corrected[3];
        for (int i = 0; i < 3; ++i)
            delta_v_corrected[i] = T(delta_v_[i]) + J_v_bg_dbg[i] + J_v_ba_dba[i];

        T J_p_bg_dbg[3], J_p_ba_dba[3];
        mat3_vec(J_p_bg_cast, d_bg, J_p_bg_dbg);
        mat3_vec(J_p_ba_cast, d_ba, J_p_ba_dba);

        T delta_p_corrected[3];
        for (int i = 0; i < 3; ++i)
            delta_p_corrected[i] = T(delta_p_[i]) + J_p_bg_dbg[i] + J_p_ba_dba[i];

        // ---- rotation residual ---------------------------------------
        // r_R = Log( dR_corrected^T @ R_wb_prev^T @ R_wb_curr )
        T R_wb_prev_T[9];
        mat3_transpose(R_wb_prev_cast, R_wb_prev_T);

        T Rpt_Rc[9];
        mat3_mul(R_wb_prev_T, R_wb_curr, Rpt_Rc);

        T dR_corr_T[9];
        mat3_transpose(delta_R_corrected, dR_corr_T);

        T R_err[9];
        mat3_mul(dR_corr_T, Rpt_Rc, R_err);

        T r_R[3];
        log_so3(R_err, r_R);

        // ---- velocity residual ----------------------------------------
        // r_v = R_wb_prev^T @ (vel_curr - vel_prev - g*dt) - Δv_corrected
        T dv_raw[3];
        for (int i = 0; i < 3; ++i)
            dv_raw[i] = vel_curr[i] - T(vel_prev_[i]) - T(gravity_[i]) * T(delta_t_);

        T r_v[3];
        mat3_vec(R_wb_prev_T, dv_raw, r_v);
        for (int i = 0; i < 3; ++i) r_v[i] -= delta_v_corrected[i];

        // ---- position residual -----------------------------------------
        // r_p = R_wb_prev^T @ (t_curr - t_prev - vel_prev*dt - 0.5*g*dt^2) - Δp_corrected
        T dp_raw[3];
        for (int i = 0; i < 3; ++i) {
            dp_raw[i] = t_wb_curr[i] - T(t_wb_prev_[i])
                      - T(vel_prev_[i]) * T(delta_t_)
                      - T(0.5) * T(gravity_[i]) * T(delta_t_) * T(delta_t_);
        }

        T r_p[3];
        mat3_vec(R_wb_prev_T, dp_raw, r_p);
        for (int i = 0; i < 3; ++i) r_p[i] -= delta_p_corrected[i];

        // ---- bias-walk residual ------------------------------------------
        T r_bg[3], r_ba[3];
        for (int i = 0; i < 3; ++i) {
            r_bg[i] = bias_curr[i]     - T(bias_prev_[i]);
            r_ba[i] = bias_curr[3 + i] - T(bias_prev_[3 + i]);
        }

        // ---- stack + weight by dense sqrt-information (15x15) -------------
        T r[15];
        for (int i = 0; i < 3; ++i) r[i]      = r_R[i];
        for (int i = 0; i < 3; ++i) r[3 + i]  = r_v[i];
        for (int i = 0; i < 3; ++i) r[6 + i]  = r_p[i];
        for (int i = 0; i < 3; ++i) r[9 + i]  = r_bg[i];
        for (int i = 0; i < 3; ++i) r[12 + i] = r_ba[i];

        for (int row = 0; row < 15; ++row) {
            T s = T(0);
            for (int col = 0; col < 15; ++col) {
                s += T(sqrt_information_[15 * row + col]) * r[col];
            }
            residuals[row] = s;
        }

        return true;
    }

    double R_wb_prev_[9], t_wb_prev_[3];
    double vel_prev_[3];
    double bias_prev_[6];
    double R_bs_[9], t_bs_[3];
    double delta_R_[9], delta_v_[3], delta_p_[3];
    double delta_t_;
    double bias_lin_[6];
    double J_R_bg_[9], J_v_bg_[9], J_v_ba_[9], J_p_bg_[9], J_p_ba_[9];
    double gravity_[3];
    double sqrt_information_[225];
};


// ─────────────────────────────────────────────────────────────────────
// Reprojection error for the current (free) view only — landmark is
// baked in as a constant (matches MATLAB's f.fixNode(wPids)).
// Same pc = R^T(X - C) / K convention as ceres_ba.cpp.
// ─────────────────────────────────────────────────────────────────────
struct ReprojectionErrorFixedPoint {

    ReprojectionErrorFixedPoint(
        double u_obs, double v_obs,
        double fx, double fy, double cx, double cy,
        double L00, double L01, double L10, double L11,
        const double point[3]
    ) : u_obs_(u_obs), v_obs_(v_obs),
        fx_(fx), fy_(fy), cx_(cx), cy_(cy),
        L00_(L00), L01_(L01), L10_(L10), L11_(L11) {
        for (int i = 0; i < 3; ++i) point_[i] = point[i];
    }

    template <typename T>
    bool operator()(
        const T* const rvec,
        const T* const C,
        T* residuals
    ) const {

        T diff[3] = {
            T(point_[0]) - C[0],
            T(point_[1]) - C[1],
            T(point_[2]) - C[2],
        };

        T R[9];
        exp_so3(rvec, R);
        T Rt[9];
        mat3_transpose(R, Rt);

        T p_cam[3];
        mat3_vec(Rt, diff, p_cam);

        T z = p_cam[2];
        if (z < T(1e-6)) z = T(1e-6);

        T xp = p_cam[0] / z;
        T yp = p_cam[1] / z;

        T u_pred = T(fx_) * xp + T(cx_);
        T v_pred = T(fy_) * yp + T(cy_);

        T r0 = T(u_obs_) - u_pred;
        T r1 = T(v_obs_) - v_pred;

        residuals[0] = T(L00_) * r0 + T(L01_) * r1;
        residuals[1] = T(L10_) * r0 + T(L11_) * r1;

        return true;
    }

    double u_obs_, v_obs_;
    double fx_, fy_, cx_, cy_;
    double L00_, L01_, L10_, L11_;
    double point_[3];
};


// ─────────────────────────────────────────────────────────────────────
// Simple L2 priors: residual = sqrt_info @ (x - measurement)
// ─────────────────────────────────────────────────────────────────────
struct Vec3Prior {
    Vec3Prior(const double measurement[3], const double sqrt_info[9]) {
        for (int i = 0; i < 3; ++i) measurement_[i] = measurement[i];
        for (int i = 0; i < 9; ++i) sqrt_info_[i] = sqrt_info[i];
    }
    template <typename T>
    bool operator()(const T* const x, T* residuals) const {
        T d[3];
        for (int i = 0; i < 3; ++i) d[i] = x[i] - T(measurement_[i]);
        for (int row = 0; row < 3; ++row) {
            T s = T(0);
            for (int col = 0; col < 3; ++col) s += T(sqrt_info_[3 * row + col]) * d[col];
            residuals[row] = s;
        }
        return true;
    }
    double measurement_[3];
    double sqrt_info_[9];
};

struct Vec6Prior {
    Vec6Prior(const double measurement[6], const double sqrt_info[36]) {
        for (int i = 0; i < 6; ++i) measurement_[i] = measurement[i];
        for (int i = 0; i < 36; ++i) sqrt_info_[i] = sqrt_info[i];
    }
    template <typename T>
    bool operator()(const T* const x, T* residuals) const {
        T d[6];
        for (int i = 0; i < 6; ++i) d[i] = x[i] - T(measurement_[i]);
        for (int row = 0; row < 6; ++row) {
            T s = T(0);
            for (int col = 0; col < 6; ++col) s += T(sqrt_info_[6 * row + col]) * d[col];
            residuals[row] = s;
        }
        return true;
    }
    double measurement_[6];
    double sqrt_info_[36];
};


// ─────────────────────────────────────────────────────────────────────
// Observation struct reused from the vision-only module's layout.
// ─────────────────────────────────────────────────────────────────────
struct MotionObservation {
    double u, v;
    double L00, L01, L10, L11;
    std::array<double, 3> xyz;
};


py::dict solve_bundle_adjustment_motion(
    // current view guesses (FREE)
    std::array<double, 6> current_pose_guess,      // [rx,ry,rz,cx,cy,cz]
    std::array<double, 3> current_velocity_guess,
    std::array<double, 6> current_bias_guess,       // [bg(3), ba(3)]

    // previous view state (FIXED)
    std::array<double, 6> previous_pose,
    std::array<double, 3> previous_velocity,
    std::array<double, 6> previous_bias,

    // camera->body extrinsic (FIXED), row-major R_bs(9), t_bs(3)
    std::array<double, 9> R_bs,
    std::array<double, 3> t_bs,

    // IMU preintegration measurement between previous and current view
    std::array<double, 9> delta_R,
    std::array<double, 3> delta_v,
    std::array<double, 3> delta_p,
    double delta_t,
    std::array<double, 6> bias_lin,                 // preint.bias_g/bias_a
    std::array<double, 9> J_R_bg,
    std::array<double, 9> J_v_bg,
    std::array<double, 9> J_v_ba,
    std::array<double, 9> J_p_bg,
    std::array<double, 9> J_p_ba,
    std::array<double, 3> gravity,
    std::array<double, 225> imu_sqrt_information,    // row-major 15x15

    // current-view 3D-2D correspondences (landmarks FIXED)
    std::vector<MotionObservation> observations,
    std::array<double, 4> K_vec,                     // [fx, fy, cx, cy]

    // priors on the refined current velocity/bias
    std::array<double, 9> velocity_prior_sqrt_info,
    std::array<double, 36> bias_prior_sqrt_info,

    int max_iterations,
    bool verbose,
    double huber_delta,
    int num_threads
) {
    double fx = K_vec[0], fy = K_vec[1], cx = K_vec[2], cy = K_vec[3];

    // ---- precompute the FIXED previous body pose (R_wb_prev, t_wb_prev) ---
    double R_wc_prev[9];
    exp_so3(previous_pose.data(), R_wc_prev);        // previous_pose[0:3] = rvec_prev
    double R_bs_T[9];
    mat3_transpose(R_bs.data(), R_bs_T);
    double R_wb_prev[9];
    mat3_mul(R_wc_prev, R_bs_T, R_wb_prev);
    double R_wb_prev_tbs[3];
    mat3_vec(R_wb_prev, t_bs.data(), R_wb_prev_tbs);
    double t_wb_prev[3];
    for (int i = 0; i < 3; ++i) t_wb_prev[i] = previous_pose[3 + i] - R_wb_prev_tbs[i];

    // ---- solver state (the only free variables) ----------------------------
    std::array<double, 6> pose_curr = current_pose_guess;
    std::array<double, 3> vel_curr  = current_velocity_guess;
    std::array<double, 6> bias_curr = current_bias_guess;

    ceres::Problem::Options problem_options;
    problem_options.enable_fast_removal = false;
    ceres::Problem problem(problem_options);

    double* rvec_ptr  = pose_curr.data();
    double* C_ptr     = pose_curr.data() + 3;
    double* vel_ptr   = vel_curr.data();
    double* bias_ptr  = bias_curr.data();

    // IMU factor
    {
        ceres::CostFunction* cost =
            new ceres::AutoDiffCostFunction<IMUPreintegrationError, 15, 3, 3, 3, 6>(
                new IMUPreintegrationError(
                    R_wb_prev, t_wb_prev,
                    previous_velocity.data(),
                    previous_bias.data(),
                    R_bs.data(), t_bs.data(),
                    delta_R.data(), delta_v.data(), delta_p.data(), delta_t,
                    bias_lin.data(),
                    J_R_bg.data(), J_v_bg.data(), J_v_ba.data(),
                    J_p_bg.data(), J_p_ba.data(),
                    gravity.data(),
                    imu_sqrt_information.data()
                )
            );
        problem.AddResidualBlock(cost, nullptr, rvec_ptr, C_ptr, vel_ptr, bias_ptr);
    }

    // Reprojection factors (landmarks fixed / baked in as constants)
    for (auto& obs : observations) {
        ceres::CostFunction* cost =
            new ceres::AutoDiffCostFunction<ReprojectionErrorFixedPoint, 2, 3, 3>(
                new ReprojectionErrorFixedPoint(
                    obs.u, obs.v, fx, fy, cx, cy,
                    obs.L00, obs.L01, obs.L10, obs.L11,
                    obs.xyz.data()
                )
            );
        ceres::LossFunction* loss = new ceres::HuberLoss(huber_delta);
        problem.AddResidualBlock(cost, loss, rvec_ptr, C_ptr);
    }

    // Priors on refined velocity / bias (pull back toward the guesses)
    {
        ceres::CostFunction* vp =
            new ceres::AutoDiffCostFunction<Vec3Prior, 3, 3>(
                new Vec3Prior(current_velocity_guess.data(), velocity_prior_sqrt_info.data())
            );
        problem.AddResidualBlock(vp, nullptr, vel_ptr);

        ceres::CostFunction* bp =
            new ceres::AutoDiffCostFunction<Vec6Prior, 6, 6>(
                new Vec6Prior(previous_bias.data(), bias_prior_sqrt_info.data())
            );
        problem.AddResidualBlock(bp, nullptr, bias_ptr);
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = verbose;
    options.max_num_iterations = max_iterations;
    options.num_threads = num_threads;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    if (verbose) {
        py::print(summary.BriefReport());
    }

    py::dict result;
    result["success"]      = summary.IsSolutionUsable();
    result["initial_cost"] = summary.initial_cost;
    result["final_cost"]   = summary.final_cost;
    result["iterations"]   = static_cast<int>(summary.iterations.size());
    result["termination"]  = ceres::TerminationTypeToString(summary.termination_type);
    result["message"]      = summary.message;

    py::list pose_out;
    for (double v : pose_curr) pose_out.append(v);
    result["pose"] = pose_out;

    py::list vel_out;
    for (double v : vel_curr) vel_out.append(v);
    result["velocity"] = vel_out;

    py::list bias_out;
    for (double v : bias_curr) bias_out.append(v);
    result["bias"] = bias_out;

    return result;
}


PYBIND11_MODULE(ceres_ba_motion, m) {
    m.doc() = "Ceres-based motion-only (single-frame) bundle adjustment for VIO";

    py::class_<MotionObservation>(m, "MotionObservation")
        .def(py::init<>())
        .def_readwrite("u", &MotionObservation::u)
        .def_readwrite("v", &MotionObservation::v)
        .def_readwrite("L00", &MotionObservation::L00)
        .def_readwrite("L01", &MotionObservation::L01)
        .def_readwrite("L10", &MotionObservation::L10)
        .def_readwrite("L11", &MotionObservation::L11)
        .def_readwrite("xyz", &MotionObservation::xyz);

    m.def(
        "solve_bundle_adjustment_motion",
        &solve_bundle_adjustment_motion,
        py::arg("current_pose_guess"),
        py::arg("current_velocity_guess"),
        py::arg("current_bias_guess"),
        py::arg("previous_pose"),
        py::arg("previous_velocity"),
        py::arg("previous_bias"),
        py::arg("R_bs"),
        py::arg("t_bs"),
        py::arg("delta_R"),
        py::arg("delta_v"),
        py::arg("delta_p"),
        py::arg("delta_t"),
        py::arg("bias_lin"),
        py::arg("J_R_bg"),
        py::arg("J_v_bg"),
        py::arg("J_v_ba"),
        py::arg("J_p_bg"),
        py::arg("J_p_ba"),
        py::arg("gravity"),
        py::arg("imu_sqrt_information"),
        py::arg("observations"),
        py::arg("K_vec"),
        py::arg("velocity_prior_sqrt_info"),
        py::arg("bias_prior_sqrt_info"),
        py::arg("max_iterations") = 10,
        py::arg("verbose") = false,
        py::arg("huber_delta") = 1.0,
        py::arg("num_threads") = 4
    );
}
