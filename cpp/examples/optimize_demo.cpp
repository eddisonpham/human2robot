#include <h2r/csv.hpp>
#include <h2r/metrics.hpp>
#include <h2r/optimizer.hpp>

#include <cstdlib>
#include <iostream>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: h2r_optimize_example <demo.csv> <out.csv> [dt]"
                  << std::endl;
        return 2;
    }
    const double dt = argc > 3 ? std::atof(argv[3]) : 0.01;
    const h2r::Trajectory demo = h2r::load_csv(argv[1], dt);
    if (demo.positions.empty()) {
        std::cerr << "empty trajectory" << std::endl;
        return 1;
    }

    const int dof = demo.dof();
    h2r::OptimizerConfig config;
    config.weights.tracking = 1.0;
    config.weights.velocity = 0.1;
    config.weights.acceleration = 0.1;
    config.weights.jerk = 0.05;
    config.weights.limits = 10.0;
    config.limits.lower = Eigen::VectorXd::Constant(dof, -2.9);
    config.limits.upper = Eigen::VectorXd::Constant(dof, 2.9);
    config.max_velocity = Eigen::VectorXd::Constant(dof, 2.0);
    config.max_acceleration = Eigen::VectorXd::Constant(dof, 20.0);
    config.max_iterations = 300;
    config.convergence_tolerance = 1e-4;
    config.step_size = 0.05;
    config.seed = 0;

    const h2r::TrajectoryOptimizer optimizer(config);
    const h2r::OptimizerResult result = optimizer.optimize(demo, demo);
    h2r::save_csv(result.trajectory, argv[2]);

    const h2r::TrajectoryMetrics before = h2r::compute_metrics(demo);
    const h2r::TrajectoryMetrics after = h2r::compute_metrics(result.trajectory);
    std::cout << "cost " << result.initial_cost << " -> " << result.final_cost
              << " in " << result.iterations << " iterations (converged="
              << result.converged << ")" << std::endl;
    std::cout << "max_velocity " << before.max_velocity << " -> "
              << after.max_velocity << std::endl;
    std::cout << "max_acceleration " << before.max_acceleration << " -> "
              << after.max_acceleration << std::endl;
    std::cout << "max_jerk " << before.max_jerk << " -> " << after.max_jerk
              << std::endl;
    std::cout << "joint_limit_violations " << before.joint_limit_violations
              << " -> " << after.joint_limit_violations << std::endl;
    return 0;
}
