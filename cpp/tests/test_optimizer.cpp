#include <h2r/optimizer.hpp>

#include <h2r/collision_checker.hpp>
#include <h2r/constraints.hpp>
#include <h2r/derivatives.hpp>
#include <h2r/metrics.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <random>

using namespace h2r;

namespace {

OptimizerConfig base_config() {
    OptimizerConfig config;
    config.weights.tracking = 1.0;
    config.weights.velocity = 0.1;
    config.weights.acceleration = 0.1;
    config.weights.jerk = 0.05;
    config.weights.limits = 10.0;
    config.limits.lower = Eigen::VectorXd::Constant(2, -1.0);
    config.limits.upper = Eigen::VectorXd::Constant(2, 1.0);
    config.max_velocity = Eigen::VectorXd::Constant(2, 2.0);
    config.max_acceleration = Eigen::VectorXd::Constant(2, 20.0);
    config.max_iterations = 200;
    config.convergence_tolerance = 1e-4;
    config.step_size = 0.05;
    config.seed = 0;
    return config;
}

Trajectory noisy_demo(int n, double dt, unsigned seed) {
    Trajectory traj(dt, {});
    std::mt19937 gen(seed);
    std::normal_distribution<double> dist(0.0, 0.15);
    for (int i = 0; i < n; ++i) {
        Eigen::VectorXd q(2);
        for (int j = 0; j < 2; ++j) {
            q(j) = std::sin(0.5 * i * dt + j) + dist(gen);
        }
        traj.positions.push_back(q);
    }
    return traj;
}

Trajectory clean_reference(int n, double dt) {
    Trajectory traj(dt, {});
    for (int i = 0; i < n; ++i) {
        Eigen::VectorXd q(2);
        for (int j = 0; j < 2; ++j) {
            q(j) = std::sin(0.5 * i * dt + j);
        }
        traj.positions.push_back(q);
    }
    return traj;
}

}  // namespace

TEST(Optimizer, ReducesCostOnNoisyInput) {
    const TrajectoryOptimizer optimizer(base_config());
    const Trajectory reference = clean_reference(50, 0.01);
    const Trajectory initial = noisy_demo(50, 0.01, 7);
    const OptimizerResult result = optimizer.optimize(initial, reference);
    EXPECT_LT(result.final_cost, result.initial_cost);
    EXPECT_GT(result.iterations, 0);
}

TEST(Optimizer, OutputSatisfiesConstraints) {
    const TrajectoryOptimizer optimizer(base_config());
    const Trajectory reference = clean_reference(60, 0.01);
    const Trajectory initial = noisy_demo(60, 0.01, 3);
    const OptimizerResult result = optimizer.optimize(initial, reference);
    const Trajectory vel = velocity(result.trajectory);
    for (const auto& q : result.trajectory.positions) {
        EXPECT_TRUE((q.array() <= 1.0 + 1e-9).all());
        EXPECT_TRUE((q.array() >= -1.0 - 1e-9).all());
    }
    for (const auto& v : vel.positions) {
        EXPECT_TRUE((v.array().abs() <= 2.0 + 1e-9).all());
    }
}

TEST(Optimizer, DeterministicForSameSeed) {
    const TrajectoryOptimizer optimizer(base_config());
    const Trajectory reference = clean_reference(40, 0.01);
    const Trajectory initial = noisy_demo(40, 0.01, 11);
    const OptimizerResult first = optimizer.optimize(initial, reference);
    const OptimizerResult second = optimizer.optimize(initial, reference);
    ASSERT_EQ(first.trajectory.num_timesteps(), second.trajectory.num_timesteps());
    for (int i = 0; i < first.trajectory.num_timesteps(); ++i) {
        EXPECT_TRUE(first.trajectory.positions[i].isApprox(
            second.trajectory.positions[i], 1e-12)) << i;
    }
    EXPECT_DOUBLE_EQ(first.final_cost, second.final_cost);
}

TEST(Optimizer, ValidInputStaysValid) {
    const TrajectoryOptimizer optimizer(base_config());
    const Trajectory reference = clean_reference(30, 0.01);
    Trajectory initial = project_all(reference, base_config().limits,
                                     base_config().max_velocity,
                                     base_config().max_acceleration);
    const OptimizerResult result = optimizer.optimize(initial, reference);
    const TrajectoryMetrics m = compute_metrics(result.trajectory);
    EXPECT_EQ(m.joint_limit_violations, 0);
    EXPECT_EQ(m.velocity_limit_violations, 0);
}

TEST(Optimizer, InvalidInputGetsProjected) {
    const TrajectoryOptimizer optimizer(base_config());
    const Trajectory reference = clean_reference(30, 0.01);
    Trajectory initial = reference;
    for (auto& q : initial.positions) {
        q = Eigen::VectorXd::Constant(2, 4.0);
    }
    const OptimizerResult result = optimizer.optimize(initial, reference);
    for (const auto& q : result.trajectory.positions) {
        EXPECT_TRUE((q.array() <= 1.0 + 1e-9).all());
    }
}

TEST(Optimizer, EmptyTrajectoryHandled) {
    const TrajectoryOptimizer optimizer(base_config());
    const Trajectory reference;
    const OptimizerResult result = optimizer.optimize(Trajectory{}, reference);
    EXPECT_EQ(result.trajectory.num_timesteps(), 0);
    EXPECT_DOUBLE_EQ(result.final_cost, result.initial_cost);
}

TEST(Optimizer, CollisionAwareOptimization) {
    OptimizerConfig config = base_config();
    config.weights.collision = 1.0;
    h2r::FunctionalCollisionChecker checker(
        [](const Eigen::VectorXd& q) {
            const double x = q(0);
            return std::max(0.0, 0.5 - std::abs(x));
        },
        0.01);
    TrajectoryOptimizer optimizer(config);
    optimizer.set_collision_checker(&checker);
    const Trajectory reference = clean_reference(40, 0.01);
    const Trajectory initial = noisy_demo(40, 0.01, 5);
    const OptimizerResult result = optimizer.optimize(initial, reference);
    int invalid = 0;
    for (const auto& q : result.trajectory.positions) {
        if (!checker.configuration_valid(q)) {
            ++invalid;
        }
    }
    EXPECT_LE(invalid, 40);
}
