#include <h2r/trajectory.hpp>

#include <gtest/gtest.h>

using namespace h2r;

namespace {

Trajectory make_traj() {
    Trajectory traj(0.1, {});
    for (int i = 0; i < 5; ++i) {
        traj.positions.push_back(Eigen::VectorXd::Constant(2, 0.5 * i));
    }
    return traj;
}

TEST(Trajectory, EmptyDefaults) {
    Trajectory traj;
    EXPECT_EQ(traj.num_timesteps(), 0);
    EXPECT_EQ(traj.dof(), 0);
    EXPECT_DOUBLE_EQ(traj.duration(), 0.0);
    EXPECT_DOUBLE_EQ(traj.dt, 0.01);
}

TEST(Trajectory, SizeAccessors) {
    const Trajectory traj = make_traj();
    EXPECT_EQ(traj.num_timesteps(), 5);
    EXPECT_EQ(traj.dof(), 2);
    EXPECT_DOUBLE_EQ(traj.duration(), 0.4);
}

TEST(Trajectory, DofOfEmptyPositions) {
    Trajectory traj(0.01, {});
    EXPECT_EQ(traj.dof(), 0);
}

}  // namespace
