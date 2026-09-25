#include <h2r/cost.hpp>

#include <h2r/collision_checker.hpp>

#include <gtest/gtest.h>

using namespace h2r;

namespace {

Trajectory straight(int n, double dt, double step) {
    Trajectory traj(dt, {});
    for (int i = 0; i < n; ++i) {
        traj.positions.push_back(Eigen::VectorXd::Constant(2, step * i));
    }
    return traj;
}

}  // namespace

TEST(Cost, ZeroForIdenticalTrajectories) {
    const Trajectory a = straight(10, 0.01, 0.01);
    const CostWeights weights;
    const double cost = compute_cost(a, a, weights, nullptr, nullptr, nullptr,
                                     nullptr, nullptr, LimitPenaltyScales{});
    EXPECT_NEAR(cost, 0.0, 1e-12);
}

TEST(Cost, TrackingWeightScalesCost) {
    const Trajectory a = straight(10, 0.01, 0.01);
    const Trajectory b = straight(10, 0.01, 0.02);
    CostWeights weights;
    weights.tracking = 2.0;
    const double cost2 = compute_cost(a, b, weights, nullptr, nullptr, nullptr,
                                      nullptr, nullptr, LimitPenaltyScales{});
    weights.tracking = 1.0;
    const double cost1 = compute_cost(a, b, weights, nullptr, nullptr, nullptr,
                                      nullptr, nullptr, LimitPenaltyScales{});
    EXPECT_NEAR(cost2, 2.0 * cost1, 1e-9);
}

TEST(Cost, JaggedTrajectoryPaysJerkPenalty) {
    Trajectory jagged(0.01, {});
    for (int i = 0; i < 10; ++i) {
        const double v = (i % 2 == 0) ? 0.0 : 1.0;
        jagged.positions.push_back(Eigen::VectorXd::Constant(1, v));
    }
    const Trajectory smooth = straight(10, 0.01, 0.0);
    CostWeights weights;
    weights.tracking = 0.0;
    weights.jerk = 1.0;
    const double jagged_cost =
        compute_cost(jagged, smooth, weights, nullptr, nullptr, nullptr, nullptr,
                     nullptr, LimitPenaltyScales{});
    EXPECT_GT(jagged_cost, 0.0);
}

TEST(Cost, LimitViolationPenalized) {
    Trajectory traj(0.01, {});
    for (int i = 0; i < 5; ++i) {
        traj.positions.push_back(Eigen::VectorXd::Constant(1, 2.0));
    }
    JointLimits limits;
    limits.lower = Eigen::VectorXd::Constant(1, -1.0);
    limits.upper = Eigen::VectorXd::Constant(1, 1.0);
    CostWeights weights;
    weights.tracking = 0.0;
    weights.limits = 1.0;
    const double cost =
        compute_cost(traj, traj, weights, &limits, nullptr, nullptr, nullptr,
                     nullptr, LimitPenaltyScales{});
    EXPECT_GT(cost, 0.0);
}

TEST(Cost, CollisionPenalized) {
    const Trajectory traj = straight(5, 0.01, 0.1);
    h2r::FunctionalCollisionChecker checker(
        [](const Eigen::VectorXd&) { return 3.0; }, 0.5);
    CostWeights weights;
    weights.tracking = 0.0;
    weights.collision = 1.0;
    const double cost = compute_cost(traj, traj, weights, nullptr, nullptr,
                                     nullptr, &checker, nullptr,
                                     LimitPenaltyScales{});
    EXPECT_NEAR(cost, 3.0, 1e-9);
}

TEST(Cost, ZeroWeightsGiveZeroCost) {
    const Trajectory a = straight(10, 0.01, 0.01);
    const Trajectory b = straight(10, 0.01, 0.5);
    CostWeights weights;
    weights.tracking = 0.0;
    const double cost = compute_cost(a, b, weights, nullptr, nullptr, nullptr,
                                     nullptr, nullptr, LimitPenaltyScales{});
    EXPECT_NEAR(cost, 0.0, 1e-12);
}
