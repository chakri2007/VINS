// ba_solver.cpp
// Ceres Bundle Adjustment solver exposed to Python via pybind11.
//
// What this file does:
//   1. Defines the ReprojectionError cost functor
//   2. Builds a ceres::Problem from Python-supplied poses/points/observations
//   3. Runs the solver
//   4. Returns refined poses and points back to Python

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <vector>
#include <map>
#include <string>

namespace py = pybind11;


// ─────────────────────────────────────────────────────────────────────────────
// ReprojectionError
//
// This is the cost functor Ceres minimizes.
// For each observation (u_obs, v_obs) of a 3D point in a camera:
//
//   residual[0] = u_obs - pi(R, t, X)[0]
//   residual[1] = v_obs - pi(R, t, X)[1]
//
// Ceres uses automatic differentiation on this functor to get Jacobians.
// The template<typename T> is what enables this — when T=double it evaluates
// normally; when T=ceres::Jet it computes derivatives simultaneously.
// ─────────────────────────────────────────────────────────────────────────────

struct ReprojectionError {

    // Observed pixel and camera intrinsics stored at construction time
    ReprojectionError(double u_obs, double v_obs,
                      double fx, double fy, double cx, double cy)
        : u_obs_(u_obs), v_obs_(v_obs),
          fx_(fx), fy_(fy), cx_(cx), cy_(cy) {}

    template <typename T>
    bool operator()(
        const T* const rvec,    // rotation vector (3,) — angle-axis
        const T* const tvec,    // translation (3,)
        const T* const point,   // 3D point in world frame (3,)
        T* residuals
    ) const {

        // ── Step 1: Rotate point from world frame to camera frame ──
        // Ceres provides AngleAxisRotatePoint which handles the
        // Rodrigues formula with correct derivatives for autodiff.
        T p_cam[3];
        ceres::AngleAxisRotatePoint(rvec, point, p_cam);

        // ── Step 2: Add translation ────────────────────────────────
        p_cam[0] += tvec[0];
        p_cam[1] += tvec[1];
        p_cam[2] += tvec[2];

        // ── Step 3: Perspective projection ────────────────────────
        // x = X/Z, y = Y/Z
        T xp = p_cam[0] / p_cam[2];
        T yp = p_cam[1] / p_cam[2];

        // ── Step 4: Apply intrinsics ───────────────────────────────
        T u_pred = T(fx_) * xp + T(cx_);
        T v_pred = T(fy_) * yp + T(cy_);

        // ── Step 5: Residuals ──────────────────────────────────────
        residuals[0] = T(u_obs_) - u_pred;
        residuals[1] = T(v_obs_) - v_pred;

        return true;
    }

    // Factory: Ceres uses this to create a cost function with autodiff.
    // <ReprojectionError, 2, 3, 3, 3> means:
    //   - cost functor type: ReprojectionError
    //   - residual dimension: 2  (u,v)
    //   - param block 1 size: 3  (rvec)
    //   - param block 2 size: 3  (tvec)
    //   - param block 3 size: 3  (3D point)
    static ceres::CostFunction* Create(
        double u_obs, double v_obs,
        double fx, double fy, double cx, double cy
    ) {
        return new ceres::AutoDiffCostFunction<ReprojectionError, 2, 3, 3, 3>(
            new ReprojectionError(u_obs, v_obs, fx, fy, cx, cy)
        );
    }

private:
    const double u_obs_, v_obs_;
    const double fx_, fy_, cx_, cy_;
};


// ─────────────────────────────────────────────────────────────────────────────
// solve_bundle_adjustment
//
// Python calls this function with:
//   poses       : list of [rx, ry, rz, tx, ty, tz]  (one per keyframe)
//   points      : list of [X, Y, Z]                  (one per landmark)
//   observations: list of (cam_idx, pt_idx, u, v)    (one per observation)
//   K_vec       : [fx, fy, cx, cy]
//   fix_first   : whether to hold camera 0 fixed (breaks gauge freedom)
//   max_iters   : Ceres solver iterations
//
// Returns:
//   dict with 'poses' and 'points' (same layout, refined values)
// ─────────────────────────────────────────────────────────────────────────────

py::dict solve_bundle_adjustment(
    std::vector<std::array<double, 6>> poses,       // [rvec(3) | tvec(3)]
    std::vector<std::array<double, 3>> points,      // [X, Y, Z]
    std::vector<std::tuple<int,int,double,double>> observations,  // (cam,pt,u,v)
    std::array<double, 4> K_vec,                    // [fx, fy, cx, cy]
    bool fix_first,
    int max_iters
) {
    const double fx = K_vec[0], fy = K_vec[1];
    const double cx = K_vec[2], cy = K_vec[3];

    // ── Build Ceres Problem ────────────────────────────────────────────────
    ceres::Problem problem;

    // Add each observation as a residual block.
    // Each block connects one pose and one point.
    for (const auto& [cam_idx, pt_idx, u_obs, v_obs] : observations) {

        if (cam_idx < 0 || cam_idx >= (int)poses.size())  continue;
        if (pt_idx  < 0 || pt_idx  >= (int)points.size()) continue;

        ceres::CostFunction* cost_fn = ReprojectionError::Create(
            u_obs, v_obs, fx, fy, cx, cy
        );

        // Huber loss: downweights large residuals (outlier robustness).
        // threshold = 1.0 pixel — residuals larger than this are treated
        // as potential outliers and given less weight.
        // Set to nullptr for pure least squares (no robustification).
        ceres::LossFunction* loss_fn = new ceres::HuberLoss(1.0);

        problem.AddResidualBlock(
            cost_fn,
            loss_fn,
            poses[cam_idx].data(),    // rvec (3 params)
            poses[cam_idx].data() + 3,// tvec (3 params)
            points[pt_idx].data()     // 3D point (3 params)
        );
    }

    // ── Fix first camera pose to break gauge freedom ───────────────────────
    // Without this, the problem has a 6-DOF ambiguity (any rigid body
    // transform applied to everything gives the same cost).
    // Fixing the first pose anchors the coordinate frame.
    if (fix_first && !poses.empty()) {
        if (problem.HasParameterBlock(poses[0].data())) {
            problem.SetParameterBlockConstant(poses[0].data());
            problem.SetParameterBlockConstant(poses[0].data() + 3);
        } else {
            // First camera has no observations — fix the next one that does
            for (int i = 0; i < (int)poses.size(); ++i) {
                if (problem.HasParameterBlock(poses[i].data())) {
                    problem.SetParameterBlockConstant(poses[i].data());
                    problem.SetParameterBlockConstant(poses[i].data() + 3);
                    break;
                }
            }
        }
    }
    

    // ── Configure and run solver ───────────────────────────────────────────
    ceres::Solver::Options options;

    // SPARSE_SCHUR: exploits the bipartite structure of BA.
    // The Jacobian has a special block structure (cameras vs points)
    // that allows the Schur complement trick — eliminates point variables
    // first, solves for cameras, then back-substitutes for points.
    // This is what makes BA tractable at scale.
    options.linear_solver_type = ceres::SPARSE_SCHUR;

    // SPARSE_SCHUR needs a sparse matrix library.
    // SuiteSparse is faster; Eigen is more portable.
    options.sparse_linear_algebra_library_type = ceres::EIGEN_SPARSE;

    options.minimizer_progress_to_stdout = false;
    options.max_num_iterations           = max_iters;
    options.num_threads                  = 2;   // safe on Pi CM5

    // Convergence tolerances
    options.function_tolerance  = 1e-4;
    options.gradient_tolerance  = 1e-6;
    options.parameter_tolerance = 1e-4;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // ── Pack results ───────────────────────────────────────────────────────
    py::list refined_poses;
    for (const auto& p : poses) {
        py::list pose_vec;
        for (double v : p) pose_vec.append(v);
        refined_poses.append(pose_vec);
    }

    py::list refined_points;
    for (const auto& pt : points) {
        py::list pt_vec;
        for (double v : pt) pt_vec.append(v);
        refined_points.append(pt_vec);
    }

    py::dict result;
    result["poses"]         = refined_poses;
    result["points"]        = refined_points;
    result["final_cost"]    = summary.final_cost;
    result["initial_cost"]  = summary.initial_cost;
    result["iterations"]    = summary.iterations.size();
    result["success"]       = (summary.termination_type == ceres::CONVERGENCE ||
                               summary.termination_type == ceres::NO_CONVERGENCE);
    result["message"]       = summary.BriefReport();

    return result;
}


// ─────────────────────────────────────────────────────────────────────────────
// pybind11 module definition
// ─────────────────────────────────────────────────────────────────────────────

PYBIND11_MODULE(ba_solver, m) {
    m.doc() = "Ceres Bundle Adjustment solver";
    m.def(
        "solve_bundle_adjustment",
        &solve_bundle_adjustment,
        py::arg("poses"),
        py::arg("points"),
        py::arg("observations"),
        py::arg("K_vec"),
        py::arg("fix_first")   = true,
        py::arg("max_iters")   = 50,
        "Run sparse Bundle Adjustment using Ceres. "
        "Returns refined poses and 3D points."
    );
}