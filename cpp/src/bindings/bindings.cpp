#include <h2r/analytic_chain.hpp>
#include <h2r/collision_checker.hpp>
#include <h2r/constraints.hpp>
#include <h2r/csv.hpp>
#include <h2r/derivatives.hpp>
#include <h2r/interpolation.hpp>
#include <h2r/metrics.hpp>
#include <h2r/optimizer.hpp>
#include <h2r/smoothing.hpp>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>

namespace py = pybind11;
using namespace h2r;

namespace {

Trajectory from_numpy(double dt, py::array_t<double, py::array::c_style | py::array::forcecast> arr) {
    auto info = arr.request();
    if (info.ndim != 2) {
        throw std::runtime_error("trajectory array must be 2D (T, dof)");
    }
    Trajectory traj;
    traj.dt = dt;
    const int rows = static_cast<int>(info.shape[0]);
    const int cols = static_cast<int>(info.shape[1]);
    const double* data = static_cast<double*>(info.ptr);
    for (int i = 0; i < rows; ++i) {
        Eigen::VectorXd q(cols);
        for (int j = 0; j < cols; ++j) {
            q(j) = data[i * cols + j];
        }
        traj.positions.push_back(q);
    }
    return traj;
}

py::array_t<double> to_numpy(const Trajectory& traj) {
    if (traj.positions.empty()) {
        return py::array_t<double>({0, 0});
    }
    const int rows = traj.num_timesteps();
    const int cols = traj.dof();
    py::array_t<double> arr({rows, cols});
    auto info = arr.request();
    double* data = static_cast<double*>(info.ptr);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            data[i * cols + j] = traj.positions[i](j);
        }
    }
    return arr;
}

Trajectory as_trajectory(const py::object& obj, double dt) {
    return from_numpy(dt, obj.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>());
}

}  // namespace

PYBIND11_MODULE(h2r_cpp, m) {
    m.doc() = "Human2Robot C++ trajectory processing and optimization";

    py::class_<JointLimits>(m, "JointLimits")
        .def(py::init<>())
        .def_readwrite("lower", &JointLimits::lower)
        .def_readwrite("upper", &JointLimits::upper);

    py::class_<CostWeights>(m, "CostWeights")
        .def(py::init<>())
        .def_readwrite("tracking", &CostWeights::tracking)
        .def_readwrite("velocity", &CostWeights::velocity)
        .def_readwrite("acceleration", &CostWeights::acceleration)
        .def_readwrite("jerk", &CostWeights::jerk)
        .def_readwrite("collision", &CostWeights::collision)
        .def_readwrite("limits", &CostWeights::limits);

    py::class_<OptimizerConfig>(m, "OptimizerConfig")
        .def(py::init<>())
        .def_readwrite("weights", &OptimizerConfig::weights)
        .def_readwrite("limits", &OptimizerConfig::limits)
        .def_readwrite("max_velocity", &OptimizerConfig::max_velocity)
        .def_readwrite("max_acceleration", &OptimizerConfig::max_acceleration)
        .def_readwrite("max_iterations", &OptimizerConfig::max_iterations)
        .def_readwrite("convergence_tolerance", &OptimizerConfig::convergence_tolerance)
        .def_readwrite("step_size", &OptimizerConfig::step_size)
        .def_readwrite("noise_scale", &OptimizerConfig::noise_scale)
        .def_readwrite("seed", &OptimizerConfig::seed);

    py::class_<OptimizerResult>(m, "OptimizerResult")
        .def_readonly("trajectory", &OptimizerResult::trajectory)
        .def_readonly("initial_cost", &OptimizerResult::initial_cost)
        .def_readonly("projected_initial_cost", &OptimizerResult::projected_initial_cost)
        .def_readonly("improvement_pct", &OptimizerResult::improvement_pct)
        .def_readonly("final_cost", &OptimizerResult::final_cost)
        .def_readonly("iterations", &OptimizerResult::iterations)
        .def_readonly("converged", &OptimizerResult::converged);

    py::class_<TrajectoryOptimizer>(m, "TrajectoryOptimizer")
        .def(py::init<OptimizerConfig>())
        .def("optimize", &TrajectoryOptimizer::optimize, py::arg("initial"),
             py::arg("reference"));

    py::class_<Trajectory>(m, "Trajectory")
        .def(py::init<>())
        .def_readwrite("dt", &Trajectory::dt)
        .def_property_readonly("positions",
            [](const Trajectory& t) { return to_numpy(t); })
        .def("num_timesteps", &Trajectory::num_timesteps)
        .def("dof", &Trajectory::dof)
        .def("duration", &Trajectory::duration);

    py::class_<RobotModel>(m, "RobotModel");

    m.def("trajectory_from_numpy", &from_numpy, py::arg("dt"), py::arg("array"));
    m.def("trajectory_to_numpy",
          [](const Trajectory& t) { return to_numpy(t); }, py::arg("trajectory"));

    m.def("linear_resample", [](const Trajectory& t, double dt) { return linear_resample(t, dt); });
    m.def("cubic_resample", [](const Trajectory& t, double dt) { return cubic_resample(t, dt); });
    m.def("moving_average_smooth", &moving_average_smooth, py::arg("trajectory"),
          py::arg("window"));
    m.def("velocity", &velocity);
    m.def("acceleration", &acceleration);
    m.def("jerk", &jerk);

    m.def("project_all", [](const Trajectory& t, const JointLimits& limits,
                            const Eigen::VectorXd& max_vel,
                            const Eigen::VectorXd& max_acc) {
        return project_all(t, limits, max_vel, max_acc);
    });

    m.def("compute_cost", [](const Trajectory& traj, const Trajectory& reference,
                             const CostWeights& weights) {
        return compute_cost(traj, reference, weights, nullptr, nullptr, nullptr,
                            nullptr, nullptr, LimitPenaltyScales{});
    });

    m.def("compute_metrics", [](const Trajectory& traj, py::object reference) {
        TrajectoryMetrics metrics;
        if (reference.is_none()) {
            metrics = compute_metrics(traj);
        } else {
            Trajectory ref = as_trajectory(reference, traj.dt);
            metrics = compute_metrics(traj, &ref, nullptr, nullptr, nullptr,
                                      nullptr, nullptr);
        }
        py::dict out;
        out["joint_limit_violations"] = metrics.joint_limit_violations;
        out["velocity_limit_violations"] = metrics.velocity_limit_violations;
        out["acceleration_limit_violations"] = metrics.acceleration_limit_violations;
        out["invalid_configurations"] = metrics.invalid_configurations;
        out["max_velocity"] = metrics.max_velocity;
        out["max_acceleration"] = metrics.max_acceleration;
        out["max_jerk"] = metrics.max_jerk;
        out["smoothness"] = metrics.smoothness;
        out["tracking_error"] = metrics.tracking_error;
        out["ee_tracking_error"] = metrics.ee_tracking_error;
        return out;
    }, py::arg("trajectory"), py::arg("reference") = py::none());

    py::class_<AnalyticChainModel, RobotModel>(m, "AnalyticChainModel")
        .def(py::init<std::vector<double>>())
        .def("forward_kinematics", &AnalyticChainModel::forward_kinematics)
        .def("jacobian", &AnalyticChainModel::jacobian)
        .def("dof", &AnalyticChainModel::dof)
        .def("joint_limits", &AnalyticChainModel::joint_limits);

    m.def("load_csv", [](const std::string& path, double dt) { return load_csv(path, dt); },
          py::arg("path"), py::arg("dt"));
    m.def("save_csv", [](const Trajectory& t, const std::string& path) { save_csv(t, path); },
          py::arg("trajectory"), py::arg("path"));
}
