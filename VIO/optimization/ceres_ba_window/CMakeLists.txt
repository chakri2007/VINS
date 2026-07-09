cmake_minimum_required(VERSION 3.16)
project(ceres_ba_window)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Ceres REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(pybind11 REQUIRED)

# Windowed VIO bundle adjustment -- N poses/velocities/biases, M IMU
# factors, full camera factors, arbitrary node fixing. Separate
# extension module from ceres_ba / ceres_ba_motion (see the module
# docstring at the top of ceres_ba_window.cpp for how the three relate).
pybind11_add_module(ceres_ba_window ceres_ba_window.cpp)
target_link_libraries(ceres_ba_window PRIVATE Ceres::ceres Eigen3::Eigen)
