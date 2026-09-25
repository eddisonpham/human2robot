cmake_minimum_required(VERSION 3.24)
project(h2r_traj VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

option(H2R_BUILD_TESTS "Build C++ tests" ON)
option(H2R_BUILD_BENCHMARKS "Build benchmark executables" ON)
option(H2R_BUILD_BINDINGS "Build pybind11 module" ON)
option(H2R_BUILD_MUJOCO "Build MuJoCo-backed robot model" OFF)
option(H2R_BUILD_EXAMPLES "Build example executables" ON)

include(FetchContent)

FetchContent_Declare(
  eigen
  URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
)
FetchContent_GetProperties(eigen)
if(NOT eigen_POPULATED)
  FetchContent_Populate(eigen)
endif()
if(NOT TARGET Eigen3::Eigen)
  add_library(Eigen3::Eigen INTERFACE IMPORTED)
  set_target_properties(Eigen3::Eigen PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${eigen_SOURCE_DIR}")
endif()

if(H2R_BUILD_TESTS)
  FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
  )
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  FetchContent_MakeAvailable(googletest)
endif()

if(H2R_BUILD_BINDINGS)
  FetchContent_Declare(
    pybind11
    URL https://github.com/pybind/pybind11/archive/refs/tags/v2.13.6.tar.gz
  )
  FetchContent_MakeAvailable(pybind11)
endif()

add_library(h2r_traj
  src/trajectory/interpolation.cpp
  src/trajectory/derivatives.cpp
  src/trajectory/smoothing.cpp
  src/kinematics/analytic_chain.cpp
  src/constraints/limits.cpp
  src/constraints/collision_checker.cpp
  src/metrics/metrics.cpp
  src/optimization/cost.cpp
  src/optimization/optimizer.cpp
  src/io/csv.cpp
)
target_include_directories(h2r_traj PUBLIC include)
target_link_libraries(h2r_traj PUBLIC Eigen3::Eigen)

if(H2R_BUILD_TESTS)
  enable_testing()
  add_executable(h2r_tests
    tests/test_trajectory.cpp
    tests/test_interpolation.cpp
    tests/test_derivatives.cpp
    tests/test_smoothing_resample.cpp
    tests/test_kinematics.cpp
    tests/test_constraints.cpp
    tests/test_metrics.cpp
    tests/test_cost.cpp
    tests/test_optimizer.cpp
    tests/test_csv_io.cpp
  )
  target_link_libraries(h2r_tests PRIVATE h2r_traj GTest::gtest_main)
  include(GoogleTest)
  gtest_discover_tests(h2r_tests)
endif()

if(H2R_BUILD_EXAMPLES)
  add_executable(h2r_optimize_example examples/optimize_demo.cpp)
  target_link_libraries(h2r_optimize_example PRIVATE h2r_traj)
endif()

if(H2R_BUILD_BINDINGS)
  pybind11_add_module(h2r_cpp src/bindings/bindings.cpp)
  target_link_libraries(h2r_cpp PRIVATE h2r_traj)
endif()
