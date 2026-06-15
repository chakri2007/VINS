cmake_minimum_required(VERSION 3.15)
project(ba_solver)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)

# ── Find dependencies ──────────────────────────────────────────────────────
find_package(Ceres REQUIRED)
find_package(pybind11 REQUIRED)

# ── Build the Python extension module ─────────────────────────────────────
pybind11_add_module(ba_solver ba_solver.cpp)

target_link_libraries(ba_solver PRIVATE
    Ceres::ceres
)

# Ceres header includes are pulled in automatically via Ceres::ceres target.
# pybind11 headers are pulled in automatically via pybind11_add_module.