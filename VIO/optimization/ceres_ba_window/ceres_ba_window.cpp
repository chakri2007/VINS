// ceres_ba_window.cpp
//
// Phase 3 windowed bundle adjustment: the full sliding-window
// factor-graph optimizer that GraphBuilder.build_windowed_vio /
// optimization/ceres_bundle_adjustment_window.py drive.
//
// Given, over the current sliding window:
//   - N keyframe [pose, velocity, bias] states (ALL free, unlike
//     BA_motion where the previous state is baked in as a constant --
//     here every window keyframe is a live optimization variable,
//     with gauge-fixing handled explicitly via fixed_pose_ids /
//     fixed_translation_only_pose_ids / fixed_velocity_ids /
//     fixed_bias_ids, same mechanism as the vision-only ceres_ba.cpp)
//   - M landmark positions (free -- same as vision-only ceres_ba.cpp)
//   - one IMU preintegration factor per consecutive keyframe pair
//   - reprojection factors connecting keyframes to landmarks
//
// jointly refines every pose/velocity/bias/landmark in the window.
//
// ─────────────────────────────────────────────────────────────────────
// Conventions (must match the rest of this repo exactly -- see
// ceres_ba_motion.cpp's header comment, which this mirrors):
//
//   Camera pose:  p_cam = R_wc^T @ (p_world - C),  i.e. (R_wc, C) is
//                 camera-to-world, C is the camera center in world
//                 coordinates.
//
//   Camera->body extrinsic (T_BS), matching imu/vi_alignment.py:
//                 p_body = R_bs @ p_camera + t_bs
//                 R_wb = R_wc @ R_bs^T
//                 t_wb = C   - R_wb @ t_bs        (body origin, world)
//
//   SO(3) exp/log follow the exact Rodrigues formulas used in
//   imu/math_utils.py (exp_so3 / log_so3 / skew), hand-rolled here
//   (not ceres::AngleAxis*) for the same reason ceres_ba_motion.cpp
//   does: provably in lock-step with the Python-side math, including
//   for the reprojection factor, since this file needs both
//   reprojection AND IMU residuals to share one rotation convention.
// ─────────────────────────────────────────────────────────────────────

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <ceres/ceres.h>

#include <vector>
#include <map>
#include <array>
#include <cmath>

namespace py = pybind11;


// ─────────────────────────────────────────────────────────────────────
// Small templated SO(3) / mat3 helpers (row-major 3x3: M[3*r+c])
// Identical to ceres_ba_motion.cpp's copies -- duplicated rather than
// shared across translation units to keep each pybind11 module a
// single self-contained .cpp/.so, matching this repo's existing
// ceres_ba / ceres_ba_motion split.
// ─────────────────────────────────────────────────────────────────────

template <typename T>
void mat3_identity(T M[9]) {
    for (int i = 0; i < 9; ++i) M[i] = T(0);
    M[0] = M[4] = M[8] = T(1);
}

template <typename T>
void mat3_mul(const T A[9], const T B[9], T C[9]) {
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

template <typename T>
void exp_so3(const T phi[3], T R[9]) {
    T theta2 = phi[0] * phi[0] + phi[1] * phi[1] + phi[2] * phi[2];
    T theta = sqrt(theta2);

    T W[9];
    skew3(phi, W);

    T WW[9];
    mat3_mul(W, W, WW);

    T A, B;
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

template <typename T>
void log_so3(const T R[9], T phi[3]) {
    T trace = R[0] + R[4] + R[8];
    T cos_theta = (trace - T(1.0)) * T(0.5);

    cos_theta = ceres::fmin(T(1.0), ceres::fmax(T(-1.0), cos_theta));

    T theta = acos(cos_theta);

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
// Reprojection error, landmark FREE (same math as ceres_ba.cpp's
// ReprojectionError, hand-rolled exp_so3 instead of AngleAxisRotate-
// Point so it shares the exact same rotation convention as the IMU
// factor below within this one problem).
// ─────────────────────────────────────────────────────────────────────
struct ReprojectionErrorFree {

    ReprojectionErrorFree(
        double u_obs, double v_obs,
        double fx, double fy, double cx, double cy,
        double L00, double L01, double L10, double L11
    ) : u_obs_(u_obs), v_obs_(v_obs),
        fx_(fx), fy_(fy), cx_(cx), cy_(cy),
        L00_(L00), L01_(L01), L10_(L10), L11_(L11) {}

    template <typename T>
    bool operator()(
        const T* const rvec,
        const T* const C,
        const T* const point,
        T* residuals
    ) const {

        T diff[3] = { point[0] - C[0], point[1] - C[1], point[2] - C[2] };

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
};


// ─────────────────────────────────────────────────────────────────────
// IMU preintegration factor between two FREE states:
//   prev: [rvec_prev(3), C_prev(3), vel_prev(3), bias_prev(6)]
//   curr: [rvec_curr(3), C_curr(3), vel_curr(3), bias_curr(6)]
//
// Same 15-dim residual ordering/math as ceres_ba_motion.cpp's
// IMUPreintegrationError, generalized so the previous state is a
// template parameter (a Ceres parameter block) instead of a baked-in
// constant -- every keyframe in the window is a live variable here,
// not just the newest one.
// ─────────────────────────────────────────────────────────────────────
struct IMUPreintegrationErrorFree {

    IMUPreintegrationErrorFree(
        const double R_bs[9], const double t_bs[3],
        const double delta_R[9], const double delta_v[3], const double delta_p[3],
        double delta_t,
        const double bias_lin[6],
        const double J_R_bg[9], const double J_v_bg[9], const double J_v_ba[9],
        const double J_p_bg[9], const double J_p_ba[9],
        const double gravity[3],
        const double sqrt_information[225]
    ) : delta_t_(delta_t) {
        for (int i = 0; i < 9; ++i) {
            R_bs_[i] = R_bs[i];
            delta_R_[i] = delta_R[i];
            J_R_bg_[i] = J_R_bg[i];
            J_v_bg_[i] = J_v_bg[i];
            J_v_ba_[i] = J_v_ba[i];
            J_p_bg_[i] = J_p_bg[i];
            J_p_ba_[i] = J_p_ba[i];
        }
        for (int i = 0; i < 3; ++i) {
            t_bs_[i] = t_bs[i];
            delta_v_[i] = delta_v[i];
            delta_p_[i] = delta_p[i];
            gravity_[i] = gravity[i];
        }
        for (int i = 0; i < 6; ++i) bias_lin_[i] = bias_lin[i];
        for (int i = 0; i < 225; ++i) sqrt_information_[i] = sqrt_information[i];
    }

    template <typename T>
    void body_pose(
        const T* const rvec, const T* const C,
        const T R_bs_T[9], const T t_bs[9 /*unused size, 3*/],
        T R_wb[9], T t_wb[3]
    ) const {
        T R_wc[9];
        exp_so3(rvec, R_wc);
        mat3_mul(R_wc, R_bs_T, R_wb);
        T R_wb_tbs[3];
        mat3_vec(R_wb, t_bs, R_wb_tbs);
        for (int i = 0; i < 3; ++i) t_wb[i] = C[i] - R_wb_tbs[i];
    }

    template <typename T>
    bool operator()(
        const T* const rvec_prev, const T* const C_prev,
        const T* const vel_prev, const T* const bias_prev,
        const T* const rvec_curr, const T* const C_curr,
        const T* const vel_curr, const T* const bias_curr,
        T* residuals
    ) const {

        T R_bs_cast[9], R_bs_T_cast[9], t_bs_cast[3];
        for (int i = 0; i < 9; ++i) R_bs_cast[i] = T(R_bs_[i]);
        mat3_transpose(R_bs_cast, R_bs_T_cast);
        for (int i = 0; i < 3; ++i) t_bs_cast[i] = T(t_bs_[i]);

        T J_R_bg_cast[9], J_v_bg_cast[9], J_v_ba_cast[9], J_p_bg_cast[9], J_p_ba_cast[9];
        T delta_R_cast[9];
        for (int i = 0; i < 9; ++i) {
            J_R_bg_cast[i] = T(J_R_bg_[i]);
            J_v_bg_cast[i] = T(J_v_bg_[i]);
            J_v_ba_cast[i] = T(J_v_ba_[i]);
            J_p_bg_cast[i] = T(J_p_bg_[i]);
            J_p_ba_cast[i] = T(J_p_ba_[i]);
            delta_R_cast[i] = T(delta_R_[i]);
        }

        // ---- both endpoints' body poses, now both computed from FREE
        // parameter blocks (this is the only structural difference
        // from ceres_ba_motion.cpp's IMUPreintegrationError) ----------
        T R_wb_prev[9], t_wb_prev[3];
        body_pose(rvec_prev, C_prev, R_bs_T_cast, t_bs_cast, R_wb_prev, t_wb_prev);

        T R_wb_curr[9], t_wb_curr[3];
        body_pose(rvec_curr, C_curr, R_bs_T_cast, t_bs_cast, R_wb_curr, t_wb_curr);

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
        T R_wb_prev_T[9];
        mat3_transpose(R_wb_prev, R_wb_prev_T);

        T Rpt_Rc[9];
        mat3_mul(R_wb_prev_T, R_wb_curr, Rpt_Rc);

        T dR_corr_T[9];
        mat3_transpose(delta_R_corrected, dR_corr_T);

        T R_err[9];
        mat3_mul(dR_corr_T, Rpt_Rc, R_err);

        T r_R[3];
        log_so3(R_err, r_R);

        // ---- velocity residual ----------------------------------------
        T dv_raw[3];
        for (int i = 0; i < 3; ++i)
            dv_raw[i] = vel_curr[i] - vel_prev[i] - T(gravity_[i]) * T(delta_t_);

        T r_v[3];
        mat3_vec(R_wb_prev_T, dv_raw, r_v);
        for (int i = 0; i < 3; ++i) r_v[i] -= delta_v_corrected[i];

        // ---- position residual -----------------------------------------
        T dp_raw[3];
        for (int i = 0; i < 3; ++i) {
            dp_raw[i] = t_wb_curr[i] - t_wb_prev[i]
                      - vel_prev[i] * T(delta_t_)
                      - T(0.5) * T(gravity_[i]) * T(delta_t_) * T(delta_t_);
        }

        T r_p[3];
        mat3_vec(R_wb_prev_T, dp_raw, r_p);
        for (int i = 0; i < 3; ++i) r_p[i] -= delta_p_corrected[i];

        // ---- bias-walk residual ------------------------------------------
        T r_bg[3], r_ba[3];
        for (int i = 0; i < 3; ++i) {
            r_bg[i] = bias_curr[i]     - bias_prev[i];
            r_ba[i] = bias_curr[3 + i] - bias_prev[3 + i];
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

    double R_bs_[9], t_bs_[3];
    double delta_R_[9], delta_v_[3], delta_p_[3];
    double delta_t_;
    double bias_lin_[6];
    double J_R_bg_[9], J_v_bg_[9], J_v_ba_[9], J_p_bg_[9], J_p_ba_[9];
    double gravity_[3];
    double sqrt_information_[225];
};


struct WindowObservation {
    int view_id;
    int point_id;
    double u, v;
    double L00, L01, L10, L11;
};


struct IMUFactorData {
    int from_view;
    int to_view;
    std::array<double, 9> delta_R;
    std::array<double, 3> delta_v;
    std::array<double, 3> delta_p;
    double delta_t;
    std::array<double, 6> bias_lin;
    std::array<double, 9> J_R_bg;
    std::array<double, 9> J_v_bg;
    std::array<double, 9> J_v_ba;
    std::array<double, 9> J_p_bg;
    std::array<double, 9> J_p_ba;
    std::array<double, 225> sqrt_information;   // row-major 15x15
};


py::dict solve_bundle_adjustment_window(
    std::map<int, std::array<double, 6>> poses,       // view_id -> [rx,ry,rz,cx,cy,cz]
    std::map<int, std::array<double, 3>> velocities,  // view_id -> [vx,vy,vz]
    std::map<int, std::array<double, 6>> biases,       // view_id -> [bg(3), ba(3)]
    std::map<int, std::array<double, 3>> points,       // point_id -> [x,y,z]

    std::vector<WindowObservation> observations,             // camera factors
    std::vector<IMUFactorData> imu_factors,             // IMU factors

    std::array<double, 9> R_bs,                         // camera->body extrinsic (fixed, shared)
    std::array<double, 3> t_bs,
    std::array<double, 3> gravity,
    std::array<double, 4> K_vec,                        // [fx, fy, cx, cy]

    std::vector<int> fixed_pose_ids,
    std::vector<int> fixed_translation_only_pose_ids,
    std::vector<int> fixed_velocity_ids,
    std::vector<int> fixed_bias_ids,

    int max_iterations,
    bool verbose,
    double huber_delta,
    int num_threads,
    double max_solver_time_in_seconds = -1.0
) {
    double fx = K_vec[0], fy = K_vec[1], cx = K_vec[2], cy = K_vec[3];

    ceres::Problem::Options problem_options;
    problem_options.enable_fast_removal = false;
    ceres::Problem problem(problem_options);

    // ---- reprojection (camera) factors -------------------------------
    for (auto &obs : observations) {

        auto pose_it  = poses.find(obs.view_id);
        auto point_it = points.find(obs.point_id);

        if (pose_it == poses.end() || point_it == points.end()) {
            continue;
        }

        ceres::CostFunction* cost =
            new ceres::AutoDiffCostFunction<ReprojectionErrorFree, 2, 3, 3, 3>(
                new ReprojectionErrorFree(
                    obs.u, obs.v, fx, fy, cx, cy,
                    obs.L00, obs.L01, obs.L10, obs.L11
                )
            );

        ceres::LossFunction* loss = new ceres::HuberLoss(huber_delta);

        double* rvec_ptr  = pose_it->second.data();
        double* C_ptr     = pose_it->second.data() + 3;
        double* point_ptr = point_it->second.data();

        problem.AddResidualBlock(cost, loss, rvec_ptr, C_ptr, point_ptr);
    }

    // ---- IMU factors --------------------------------------------------
    for (auto &f : imu_factors) {

        auto prev_it = poses.find(f.from_view);
        auto curr_it = poses.find(f.to_view);
        auto vel_prev_it = velocities.find(f.from_view);
        auto vel_curr_it = velocities.find(f.to_view);
        auto bias_prev_it = biases.find(f.from_view);
        auto bias_curr_it = biases.find(f.to_view);

        if (prev_it == poses.end() || curr_it == poses.end() ||
            vel_prev_it == velocities.end() || vel_curr_it == velocities.end() ||
            bias_prev_it == biases.end() || bias_curr_it == biases.end()) {
            continue;
        }

        ceres::CostFunction* cost =
            new ceres::AutoDiffCostFunction<IMUPreintegrationErrorFree, 15,
                3, 3, 3, 6, 3, 3, 3, 6>(
                new IMUPreintegrationErrorFree(
                    R_bs.data(), t_bs.data(),
                    f.delta_R.data(), f.delta_v.data(), f.delta_p.data(), f.delta_t,
                    f.bias_lin.data(),
                    f.J_R_bg.data(), f.J_v_bg.data(), f.J_v_ba.data(),
                    f.J_p_bg.data(), f.J_p_ba.data(),
                    gravity.data(),
                    f.sqrt_information.data()
                )
            );

        problem.AddResidualBlock(
            cost, nullptr,
            prev_it->second.data(), prev_it->second.data() + 3,
            vel_prev_it->second.data(), bias_prev_it->second.data(),
            curr_it->second.data(), curr_it->second.data() + 3,
            vel_curr_it->second.data(), bias_curr_it->second.data()
        );
    }

    // ---- gauge fixing (arbitrary node fixing) --------------------------
    for (int vid : fixed_pose_ids) {
        auto it = poses.find(vid);
        if (it == poses.end()) continue;
        if (problem.HasParameterBlock(it->second.data())) {
            problem.SetParameterBlockConstant(it->second.data());
        }
        if (problem.HasParameterBlock(it->second.data() + 3)) {
            problem.SetParameterBlockConstant(it->second.data() + 3);
        }
    }

    for (int vid : fixed_translation_only_pose_ids) {
        auto it = poses.find(vid);
        if (it == poses.end()) continue;
        if (problem.HasParameterBlock(it->second.data() + 3)) {
            problem.SetParameterBlockConstant(it->second.data() + 3);
        }
    }

    for (int vid : fixed_velocity_ids) {
        auto it = velocities.find(vid);
        if (it == velocities.end()) continue;
        if (problem.HasParameterBlock(it->second.data())) {
            problem.SetParameterBlockConstant(it->second.data());
        }
    }

    for (int vid : fixed_bias_ids) {
        auto it = biases.find(vid);
        if (it == biases.end()) continue;
        if (problem.HasParameterBlock(it->second.data())) {
            problem.SetParameterBlockConstant(it->second.data());
        }
    }

    ceres::Solver::Options options;
    if (ceres::IsSparseLinearAlgebraLibraryTypeAvailable(ceres::SUITE_SPARSE) ||
        ceres::IsSparseLinearAlgebraLibraryTypeAvailable(ceres::EIGEN_SPARSE)) {
        options.linear_solver_type = ceres::SPARSE_SCHUR;
    } else {
        options.linear_solver_type = ceres::DENSE_SCHUR;
    }
    options.minimizer_progress_to_stdout = verbose;
    options.max_num_iterations           = max_iterations;
    options.num_threads                  = num_threads;
    if (max_solver_time_in_seconds > 0.0) {
        options.max_solver_time_in_seconds = max_solver_time_in_seconds;
    }

    ceres::Solver::Summary summary;
    {
        // Scoped tightly around just the solve, same reasoning as
        // ceres_ba.cpp / ceres_ba_motion.cpp.
        py::gil_scoped_release release;
        ceres::Solve(options, &problem, &summary);
    }

    if (verbose) {
        py::print(summary.BriefReport());
    }

    py::dict poses_out;
    for (auto &kv : poses) {
        py::list p;
        for (double v : kv.second) p.append(v);
        poses_out[py::int_(kv.first)] = p;
    }

    py::dict velocities_out;
    for (auto &kv : velocities) {
        py::list p;
        for (double v : kv.second) p.append(v);
        velocities_out[py::int_(kv.first)] = p;
    }

    py::dict biases_out;
    for (auto &kv : biases) {
        py::list p;
        for (double v : kv.second) p.append(v);
        biases_out[py::int_(kv.first)] = p;
    }

    py::dict points_out;
    for (auto &kv : points) {
        py::list p;
        for (double v : kv.second) p.append(v);
        points_out[py::int_(kv.first)] = p;
    }

    py::dict result;
    result["success"]      = summary.IsSolutionUsable();
    result["initial_cost"] = summary.initial_cost;
    result["final_cost"]   = summary.final_cost;
    result["iterations"]   = static_cast<int>(summary.iterations.size());
    result["termination"]  = ceres::TerminationTypeToString(summary.termination_type);
    result["message"]      = summary.message;
    result["poses"]        = poses_out;
    result["velocities"]   = velocities_out;
    result["biases"]       = biases_out;
    result["points"]       = points_out;
    result["num_residuals"] = summary.num_residuals;

    return result;
}


PYBIND11_MODULE(ceres_ba_window, m) {
    m.doc() = "Ceres-based windowed (multi-frame) bundle adjustment for VIO Phase 3";

    py::class_<WindowObservation>(m, "Observation")
        .def(py::init<>())
        .def_readwrite("view_id", &WindowObservation::view_id)
        .def_readwrite("point_id", &WindowObservation::point_id)
        .def_readwrite("u", &WindowObservation::u)
        .def_readwrite("v", &WindowObservation::v)
        .def_readwrite("L00", &WindowObservation::L00)
        .def_readwrite("L01", &WindowObservation::L01)
        .def_readwrite("L10", &WindowObservation::L10)
        .def_readwrite("L11", &WindowObservation::L11);

    py::class_<IMUFactorData>(m, "IMUFactorData")
        .def(py::init<>())
        .def_readwrite("from_view", &IMUFactorData::from_view)
        .def_readwrite("to_view", &IMUFactorData::to_view)
        .def_readwrite("delta_R", &IMUFactorData::delta_R)
        .def_readwrite("delta_v", &IMUFactorData::delta_v)
        .def_readwrite("delta_p", &IMUFactorData::delta_p)
        .def_readwrite("delta_t", &IMUFactorData::delta_t)
        .def_readwrite("bias_lin", &IMUFactorData::bias_lin)
        .def_readwrite("J_R_bg", &IMUFactorData::J_R_bg)
        .def_readwrite("J_v_bg", &IMUFactorData::J_v_bg)
        .def_readwrite("J_v_ba", &IMUFactorData::J_v_ba)
        .def_readwrite("J_p_bg", &IMUFactorData::J_p_bg)
        .def_readwrite("J_p_ba", &IMUFactorData::J_p_ba)
        .def_readwrite("sqrt_information", &IMUFactorData::sqrt_information);

    m.def(
        "solve_bundle_adjustment_window",
        &solve_bundle_adjustment_window,
        py::arg("poses"),
        py::arg("velocities"),
        py::arg("biases"),
        py::arg("points"),
        py::arg("observations"),
        py::arg("imu_factors"),
        py::arg("R_bs"),
        py::arg("t_bs"),
        py::arg("gravity"),
        py::arg("K_vec"),
        py::arg("fixed_pose_ids"),
        py::arg("fixed_translation_only_pose_ids"),
        py::arg("fixed_velocity_ids"),
        py::arg("fixed_bias_ids"),
        py::arg("max_iterations") = 100,
        py::arg("verbose") = false,
        py::arg("huber_delta") = 1.0,
        py::arg("num_threads") = 4,
        py::arg("max_solver_time_in_seconds") = -1.0
    );
}
