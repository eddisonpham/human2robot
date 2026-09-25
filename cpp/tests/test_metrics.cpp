#include <h2r/metrics.hpp>

#include <h2r/analytic_chain.hpp>
#include <h2r/collision_checker.hpp>
#include <h2r/constraints.hpp>
#include <h2r/derivatives.hpp>

#include <gtest/gtest.h>

#include <cmath>

using namespace h2r;

namespace {

Trajectory smooth_traj(int n, double dt) {
    Trajectory traj(dt, {});
    for (int i = 0; i < n; ++i) {
        traj.positions.push_back(
            Eigen::VectorXd::Constant(2, std::sin(0.5 * i * dt)));
    }
    return traj;
}

}  // namespace

TEST(Metrics, SmoothTrajectoryHasLowJerk) {
    const TrajectoryMetrics m = compute_metrics(smooth_traj(50, 0.01));
    EXPECT_LT(m.max_jerk, 100.0);
    EXPECT_GT(m.max_velocity, 0.0);
}

TEST(Metrics, CountsJointViolations) {
    Trajectory traj = smooth_traj(10, 0.01);
    traj.positions[3] = Eigen::VectorXd::Constant(2, 2.0);
    JointLimits limits;
    limits.lower = Eigen::VectorXd::Constant(2, -1.0);
    limits.upper = Eigen::VectorXd::Constant(2, 1.0);
    const TrajectoryMetrics m =
        compute_metrics(traj, nullptr, &limits, nullptr, nullptr, nullptr, nullptr);
    EXPECT_EQ(m.joint_limit_violations, 1);
}

TEST(Metrics, CountsVelocityViolations) {
    Trajectory traj(0.01, {});
    for (int i = 0; i < 5; ++i) {
        traj.positions.push_back(Eigen::VectorXd::Constant(1, 10.0 * i));
    }
    const Eigen::VectorXd max_vel = Eigen::VectorXd::Constant(1, 1.0);
    const TrajectoryMetrics m =
        compute_metrics(traj, nullptr, nullptr, &max_vel, nullptr, nullptr, nullptr);
    EXPECT_GT(m.velocity_limit_violations, 0);
}

TEST(Metrics, CountsInvalidConfigurations) {
    Trajectory traj(0.01, {});
    for (int i = 0; i < 6; ++i) {
        traj.positions.push_back(
            Eigen::VectorXd::Constant(1, std::sin(0.5 * i * 0.01)));
    }
    h2r::FunctionalCollisionChecker checker(
        [](const Eigen::VectorXd& q) { return q(0) > 0.002 ? 1.0 : 0.0; }, 0.5);
    const TrajectoryMetrics m =
        compute_metrics(traj, nullptr, nullptr, nullptr, nullptr, &checker, nullptr);
    EXPECT_EQ(m.invalid_configurations, 5);
}

TEST(Metrics, TrackingErrorZeroForIdentical) {
    const Trajectory traj = smooth_traj(20, 0.01);
    const TrajectoryMetrics m = compute_metrics(traj, &traj, nullptr, nullptr,
                                                nullptr, nullptr, nullptr);
    EXPECT_NEAR(m.tracking_error, 0.0, 1e-12);
}

TEST(Metrics, EeTrackingErrorWithModel) {
    h2r::AnalyticChainModel model(std::vector<double>{1.0, 1.0});
    Trajectory traj(0.1, {});
    Trajectory reference(0.1, {});
    for (int i = 0; i < 5; ++i) {
        Eigen::VectorXd q(2);
        q << 0.1 * i, 0.0;
        traj.positions.push_back(q);
        reference.positions.push_back(q);
    }
    traj.positions[2](0) += 0.1;
    const TrajectoryMetrics m = compute_metrics(traj, &reference, nullptr, nullptr,
                                                nullptr, nullptr, &model);
    EXPECT_GT(m.ee_tracking_error, 0.0);
    EXPECT_DOUBLE_EQ(m.tracking_error, 0.1 / 5.0);
}
