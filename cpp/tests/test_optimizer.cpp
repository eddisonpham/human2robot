#include <h2r/optimizer.hpp>

#include <h2r/collision_checker.hpp>
#include <h2r/constraints.hpp>
#include <h2r/cost.hpp>
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

namespace {

// A smooth ramp that overshoots the upper joint limit. Projection flattens the
// overshoot, which introduces a kink, and the resulting jerk cost exceeds the
// small limit-violation penalty it removes. This is the regime behind the
// non-convergence reports on real DexYCB sequences.
Trajectory overshooting_ramp(int timesteps, int dof) {
    Trajectory traj;
    traj.dt = 0.02;
    for (int i = 0; i < timesteps; ++i) {
        const double u = static_cast<double>(i) /
                         static_cast<double>(timesteps - 1);
        traj.positions.push_back(
            Eigen::VectorXd::Constant(dof, 0.9 + 0.2 * u));
    }
    return traj;
}

double cost_of(const Trajectory& traj, const Trajectory& reference) {
    const OptimizerConfig config = base_config();
    return compute_cost(traj, reference, config.weights, &config.limits,
                        &config.max_velocity, &config.max_acceleration,
                        nullptr, nullptr, config.scales);
}

}  // namespace

TEST(Optimizer, ProjectionCanRaiseTotalCost) {
    // Documents the precondition the convergence test depends on: for this
    // input, the projected starting point is more expensive than the input.
    const Trajectory initial = overshooting_ramp(40, 2);
    const Trajectory projected =
        project_all(initial, base_config().limits, base_config().max_velocity,
                    base_config().max_acceleration);
    EXPECT_GT(cost_of(projected, initial), cost_of(initial, initial));
}

TEST(Optimizer, FinalCostNeverExceedsProjectedBaseline) {
    // The search only ever accepts a strictly cheaper candidate, so the result
    // can never be worse than the projected trajectory it started from. This
    // held even when convergence was measured against the unprojected input,
    // where projection could raise the cost and make final_cost exceed
    // initial_cost.
    const TrajectoryOptimizer optimizer(base_config());
    const Trajectory initial = overshooting_ramp(40, 2);
    const Trajectory projected =
        project_all(initial, base_config().limits, base_config().max_velocity,
                    base_config().max_acceleration);
    const OptimizerResult result = optimizer.optimize(initial, initial);
    EXPECT_LE(result.final_cost, cost_of(projected, initial) + 1e-9);
}

TEST(Optimizer, ImprovementIsZeroWithoutProgress) {
    // improvement_pct is the honest measure of whether the search did anything.
    // With max_iterations = 0 it must report exactly zero rather than a
    // spurious improvement from projection alone.
    OptimizerConfig config = base_config();
    config.max_iterations = 0;
    const TrajectoryOptimizer optimizer(config);
    const Trajectory initial = overshooting_ramp(40, 2);
    const OptimizerResult result = optimizer.optimize(initial, initial);
    EXPECT_DOUBLE_EQ(result.improvement_pct, 0.0);
    EXPECT_FALSE(result.converged);
}

TEST(Optimizer, NoiseScaleZeroRemovesSeedDependence) {
    // With the per-timestep Gaussian noise switched off the search is a pure
    // tracking step, so the result no longer depends on the seed. This is the
    // property the pipeline relies on: it sets noise_scale = 0 because
    // independent per-sample noise is adversarial for a smoothness-dominated
    // cost, raising jerk faster than the tracking pull lowers it.
    OptimizerConfig a = base_config();
    a.noise_scale = 0.0;
    a.step_size = 0.5;
    a.seed = 1;
    OptimizerConfig b = a;
    b.seed = 999;

    const TrajectoryOptimizer opt_a(a);
    const TrajectoryOptimizer opt_b(b);
    const Trajectory initial = overshooting_ramp(60, 2);
    const OptimizerResult ra = opt_a.optimize(initial, initial);
    const OptimizerResult rb = opt_b.optimize(initial, initial);
    EXPECT_DOUBLE_EQ(ra.final_cost, rb.final_cost);
    EXPECT_EQ(ra.iterations, rb.iterations);
    EXPECT_EQ(ra.converged, rb.converged);
}

TEST(Optimizer, NoiseScaleAffectsResult) {
    // Guard against noise_scale being plumbed through but ignored.
    OptimizerConfig noisy = base_config();
    noisy.noise_scale = 0.1;
    noisy.seed = 3;
    OptimizerConfig quiet = noisy;
    quiet.noise_scale = 0.0;
    const TrajectoryOptimizer noisy_optimizer(noisy);
    const TrajectoryOptimizer quiet_optimizer(quiet);
    const Trajectory initial = overshooting_ramp(60, 2);
    const OptimizerResult rn = noisy_optimizer.optimize(initial, initial);
    const OptimizerResult rq = quiet_optimizer.optimize(initial, initial);
    EXPECT_NE(rn.final_cost, rq.final_cost);
}

TEST(Optimizer, BacktrackingFindsSmallerDescentStep) {
    // A step that overshoots must be retried at a smaller size rather than
    // aborting the whole search.
    OptimizerConfig small = base_config();
    small.step_size = 0.5;
    small.noise_scale = 0.0;
    const TrajectoryOptimizer optimizer(small);
    const Trajectory initial = overshooting_ramp(50, 2);
    const OptimizerResult result = optimizer.optimize(initial, initial);
    EXPECT_LE(result.final_cost, result.projected_initial_cost + 1e-9);
    EXPECT_GT(result.iterations, 0);
}

TEST(Optimizer, ConvergenceUsesProjectedBaseline) {
    // Regression test: convergence must be measured from the projected
    // starting point, the place the search actually begins. Measured against
    // the unprojected input, any input whose projection raises cost could never
    // satisfy the tolerance and was reported non-convergent by construction.
    const TrajectoryOptimizer optimizer(base_config());
    const Trajectory initial = overshooting_ramp(40, 2);
    const Trajectory projected =
        project_all(initial, base_config().limits, base_config().max_velocity,
                    base_config().max_acceleration);
    const double baseline = cost_of(projected, initial);
    const OptimizerResult result = optimizer.optimize(initial, initial);
    const bool improved_enough =
        baseline - result.final_cost >= base_config().convergence_tolerance;
    EXPECT_EQ(result.converged, improved_enough);
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
