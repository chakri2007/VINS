cmake_minimum_required(VERSION 3.16)
project(ceres_ba)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Ceres REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(pybind11 REQUIRED)

pybind11_add_module(ceres_ba ceres_ba.cpp)
target_link_libraries(ceres_ba PRIVATE Ceres::ceres Eigen3::Eigen)
