#include <h2r/constraints.hpp>
#include <h2r/derivatives.hpp>

#include <gtest/gtest.h>

#include <random>

using namespace h2r;

namespace {

Trajectory make_traj(int n, double dt) {
    Trajectory traj(dt, {});
    for (int i = 0; i < n; ++i) {
        traj.positions.push_back(Eigen::VectorXd::Zero(2));
    }
    return traj;
}

}  // namespace

TEST(ProjectJointLimits, ClipsViolations) {
    Trajectory traj = make_traj(3, 0.01);
    traj.positions[1] = Eigen::VectorXd::Constant(2, 5.0);
    JointLimits limits;
    limits.lower = Eigen::VectorXd::Constant(2, -1.0);
    limits.upper = Eigen::VectorXd::Constant(2, 1.0);
    const Trajectory out = project_joint_limits(traj, limits);
    EXPECT_DOUBLE_EQ(out.positions[1](0), 1.0);
    EXPECT_DOUBLE_EQ(out.positions[0](0), 0.0);
}

TEST(ProjectJointLimits, PreservesLengthAndOrder) {
    Trajectory traj = make_traj(10, 0.01);
    JointLimits limits;
    limits.lower = Eigen::VectorXd::Constant(2, -0.1);
    limits.upper = Eigen::VectorXd::Constant(2, 0.1);
    const Trajectory out = project_joint_limits(traj, limits);
    EXPECT_EQ(out.num_timesteps(), 10);
    for (int i = 1; i < 10; ++i) {
        EXPECT_GE(out.positions[i](0), out.positions[i - 1](0) - 1e-12);
    }
}

TEST(ProjectVelocityLimits, RemovesViolations) {
    Trajectory traj = make_traj(5, 0.1);
    for (int i = 0; i < 5; ++i) {
        traj.positions[i] = Eigen::VectorXd::Constant(1, 10.0 * i);
    }
    const Eigen::VectorXd max_vel = Eigen::VectorXd::Constant(1, 1.0);
    const Trajectory out = project_velocity_limits(traj, max_vel);
    const Trajectory vel = velocity(out);
    for (const auto& v : vel.positions) {
        EXPECT_LE(v(0), 1.0 + 1e-9);
        EXPECT_GE(v(0), -1.0 - 1e-9);
    }
}

TEST(ProjectVelocityLimits, ValidTrajectoryUnchanged) {
    Trajectory traj = make_traj(5, 0.1);
    for (int i = 0; i < 5; ++i) {
        traj.positions[i] = Eigen::VectorXd::Constant(1, 0.05 * i);
    }
    const Eigen::VectorXd max_vel = Eigen::VectorXd::Constant(1, 1.0);
    const Trajectory out = project_velocity_limits(traj, max_vel);
    for (int i = 0; i < 5; ++i) {
        EXPECT_TRUE(out.positions[i].isApprox(traj.positions[i], 1e-12));
    }
}

TEST(ProjectAccelerationLimits, RemovesViolations) {
    Trajectory traj = make_traj(8, 0.1);
    for (int i = 0; i < 8; ++i) {
        const double v = (i % 2 == 0) ? 1.0 : -1.0;
        traj.positions[i] = Eigen::VectorXd::Constant(1, v);
    }
    const Eigen::VectorXd max_acc = Eigen::VectorXd::Constant(1, 0.5);
    const Trajectory out = project_acceleration_limits(traj, max_acc);
    const Trajectory acc = acceleration(out);
    for (int i = 1; i < acc.num_timesteps() - 1; ++i) {
        EXPECT_LE(acc.positions[i](0), 0.5 + 1e-9);
        EXPECT_GE(acc.positions[i](0), -0.5 - 1e-9);
    }
}

TEST(ProjectAll, SatisfiesAllConstraints) {
    Trajectory traj = make_traj(20, 0.05);
    std::mt19937 gen(3);
    std::uniform_real_distribution<double> dist(-3.0, 3.0);
    for (auto& q : traj.positions) {
        q = Eigen::VectorXd::Constant(2, dist(gen));
    }
    JointLimits limits;
    limits.lower = Eigen::VectorXd::Constant(2, -1.0);
    limits.upper = Eigen::VectorXd::Constant(2, 1.0);
    const Eigen::VectorXd max_vel = Eigen::VectorXd::Constant(2, 2.0);
    const Eigen::VectorXd max_acc = Eigen::VectorXd::Constant(2, 8.0);
    const Trajectory out = project_all(traj, limits, max_vel, max_acc);
    EXPECT_EQ(out.num_timesteps(), 20);
    for (const auto& q : out.positions) {
        EXPECT_TRUE((q.array() <= 1.0 + 1e-9).all());
        EXPECT_TRUE((q.array() >= -1.0 - 1e-9).all());
    }
    const Trajectory vel = velocity(out);
    for (const auto& v : vel.positions) {
        EXPECT_TRUE((v.array().abs() <= 2.0 + 1e-9).all());
    }
}
