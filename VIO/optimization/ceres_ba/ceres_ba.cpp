// ceres_ba.cpp
//
// Drop-in Ceres replacement for optimization/bundle_adjustment.py's
// scipy.optimize.least_squares call.
//
// IMPORTANT — this project's convention (see camera_factor.py):
//
//     pc = R.T @ (xyz - C)
//
// i.e. R is camera-to-world rotation, and "t" stored on each pose node is
// actually the CAMERA CENTER in world coordinates (NOT the standard
// world-to-camera translation vector). This is different from the classic
// Ceres BA tutorial convention (p_cam = R * X + t), so the cost functor
// below is written specifically for this repo's parameterization.
//
// R^T is applied via AngleAxisRotatePoint with the NEGATED angle-axis
// vector, since Rodrigues(-rvec) == Rodrigues(rvec)^T.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <vector>
#include <map>
#include <array>
#include <thread>

namespace py = pybind11;


// ─────────────────────────────────────────────────────────────────────────
// ReprojectionError
//
// residual = sqrt_information @ (measurement - project(R, C, X))
//
// Parameter blocks:
//   rvec  (3,) angle-axis, camera-to-world rotation
//   C     (3,) camera center, world coordinates
//   point (3,) landmark, world coordinates
// ─────────────────────────────────────────────────────────────────────────
struct ReprojectionError {

    ReprojectionError(double u_obs, double v_obs,
                       double fx, double fy, double cx, double cy,
                       double L00, double L01, double L10, double L11)
        : u_obs_(u_obs), v_obs_(v_obs),
          fx_(fx), fy_(fy), cx_(cx), cy_(cy),
          L00_(L00), L01_(L01), L10_(L10), L11_(L11) {}

    template <typename T>
    bool operator()(
        const T* const rvec,
        const T* const C,
        const T* const point,
        T* residuals
    ) const {

        // world point minus camera center
        T diff[3] = {
            point[0] - C[0],
            point[1] - C[1],
            point[2] - C[2],
        };

        // apply R^T == rotation by -rvec
        T neg_rvec[3] = { -rvec[0], -rvec[1], -rvec[2] };

        T p_cam[3];
        ceres::AngleAxisRotatePoint(neg_rvec, diff, p_cam);

        // behind camera -> Ceres has no branchless "reject", so we clamp
        // depth away from zero to keep the residual finite/well-defined.
        // Bad-depth points should be filtered out in Python before
        // building the problem (same as the scipy path effectively does
        // via triangulation-time cheirality checks).
        T z = p_cam[2];
        if (z < T(1e-6)) {
            z = T(1e-6);
        }

        T xp = p_cam[0] / z;
        T yp = p_cam[1] / z;

        T u_pred = T(fx_) * xp + T(cx_);
        T v_pred = T(fy_) * yp + T(cy_);

        T r0 = T(u_obs_) - u_pred;
        T r1 = T(v_obs_) - v_pred;

        // weighted = L @ r   (L = sqrt_information, row-major 2x2)
        residuals[0] = T(L00_) * r0 + T(L01_) * r1;
        residuals[1] = T(L10_) * r0 + T(L11_) * r1;

        return true;
    }

    double u_obs_, v_obs_;
    double fx_, fy_, cx_, cy_;
    double L00_, L01_, L10_, L11_;
};


struct Observation {
    int view_id;
    int point_id;
    double u, v;
    double L00, L01, L10, L11;
};


py::dict solve_bundle_adjustment(
    std::map<int, std::array<double, 6>> poses,      // view_id -> [rx,ry,rz,cx,cy,cz]
    std::map<int, std::array<double, 3>> points,      // point_id -> [x,y,z]
    std::vector<Observation> observations,
    std::vector<double> K_vec,                         // [fx, fy, cx, cy]
    std::vector<int> fixed_pose_ids,
    int max_iterations,
    bool verbose,
    double huber_delta,
    int num_threads
) {
    double fx = K_vec[0], fy = K_vec[1], cx = K_vec[2], cy = K_vec[3];

    ceres::Problem::Options problem_options;
    problem_options.enable_fast_removal = false;
    ceres::Problem problem(problem_options);

    // Ceres needs stable storage for parameter blocks — the maps of
    // std::array already provide contiguous, stable double[N] storage
    // per key, so we can point directly into them.

    for (auto &obs : observations) {

        auto pose_it  = poses.find(obs.view_id);
        auto point_it = points.find(obs.point_id);

        if (pose_it == poses.end() || point_it == points.end()) {
            continue;
        }

        ceres::CostFunction* cost_function =
            new ceres::AutoDiffCostFunction<ReprojectionError, 2, 3, 3, 3>(
                new ReprojectionError(
                    obs.u, obs.v,
                    fx, fy, cx, cy,
                    obs.L00, obs.L01, obs.L10, obs.L11
                )
            );

        ceres::LossFunction* loss = new ceres::HuberLoss(huber_delta);

        double* rvec_ptr  = pose_it->second.data();      // first 3
        double* C_ptr     = pose_it->second.data() + 3;  // last 3
        double* point_ptr = point_it->second.data();

        problem.AddResidualBlock(
            cost_function, loss,
            rvec_ptr, C_ptr, point_ptr
        );
    }

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

    ceres::Solver::Options options;

    // SPARSE_SCHUR is the standard BA linear solver: eliminate landmarks
    // via the Schur complement, solve the much smaller pose-only system.
    if (ceres::IsSparseLinearAlgebraLibraryTypeAvailable(ceres::SUITE_SPARSE) ||
        ceres::IsSparseLinearAlgebraLibraryTypeAvailable(ceres::EIGEN_SPARSE)) {
        options.linear_solver_type = ceres::SPARSE_SCHUR;
    } else {
        options.linear_solver_type = ceres::DENSE_SCHUR;
    }

    options.minimizer_progress_to_stdout = verbose;
    options.max_num_iterations           = max_iterations;
    options.num_threads                  = num_threads;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    if (verbose) {
        py::print(summary.BriefReport());
    }

    py::dict poses_out;
    for (auto &kv : poses) {
        py::list p;
        for (double v : kv.second) p.append(v);
        poses_out[py::int_(kv.first)] = p;
    }

    py::dict points_out;
    for (auto &kv : points) {
        py::list p;
        for (double v : kv.second) p.append(v);
        points_out[py::int_(kv.first)] = p;
    }

    py::dict result;
    result["success"]        = summary.IsSolutionUsable();
    result["initial_cost"]   = summary.initial_cost;
    result["final_cost"]     = summary.final_cost;
    result["iterations"]     = static_cast<int>(summary.iterations.size());
    result["termination"]    = ceres::TerminationTypeToString(summary.termination_type);
    result["message"]        = summary.message;
    result["poses"]          = poses_out;
    result["points"]         = points_out;
    result["num_residuals"]  = summary.num_residuals;

    return result;
}


PYBIND11_MODULE(ceres_ba, m) {
    m.doc() = "Ceres-based bundle adjustment for VIO FactorGraph";

    py::class_<Observation>(m, "Observation")
        .def(py::init<>())
        .def_readwrite("view_id", &Observation::view_id)
        .def_readwrite("point_id", &Observation::point_id)
        .def_readwrite("u", &Observation::u)
        .def_readwrite("v", &Observation::v)
        .def_readwrite("L00", &Observation::L00)
        .def_readwrite("L01", &Observation::L01)
        .def_readwrite("L10", &Observation::L10)
        .def_readwrite("L11", &Observation::L11);

    m.def(
        "solve_bundle_adjustment",
        &solve_bundle_adjustment,
        py::arg("poses"),
        py::arg("points"),
        py::arg("observations"),
        py::arg("K_vec"),
        py::arg("fixed_pose_ids"),
        py::arg("max_iterations") = 100,
        py::arg("verbose") = false,
        py::arg("huber_delta") = 1.0,
        py::arg("num_threads") = 4
    );
}
