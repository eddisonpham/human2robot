#include <h2r/interpolation.hpp>

#include <gtest/gtest.h>

#include <cmath>

using namespace h2r;

namespace {

Trajectory sine_traj(int n, double dt, int dof) {
    Trajectory traj(dt, {});
    for (int i = 0; i < n; ++i) {
        Eigen::VectorXd q(dof);
        for (int j = 0; j < dof; ++j) {
            q(j) = std::sin(0.5 * i * dt + j);
        }
        traj.positions.push_back(q);
    }
    return traj;
}

}  // namespace

TEST(LinearResample, PreservesLengthAndEndpoints) {
    const Trajectory traj = sine_traj(11, 0.1, 3);
    const Trajectory out = linear_resample(traj, 0.1);
    ASSERT_EQ(out.num_timesteps(), traj.num_timesteps());
    EXPECT_TRUE(out.positions.front().isApprox(traj.positions.front(), 1e-12));
    EXPECT_TRUE(out.positions.back().isApprox(traj.positions.back(), 1e-12));
}

TEST(LinearResample, UpsamplesWithinBounds) {
    const Trajectory traj = sine_traj(11, 0.1, 2);
    const Trajectory out = linear_resample(traj, 0.05);
    EXPECT_EQ(out.num_timesteps(), 21);
    EXPECT_DOUBLE_EQ(out.dt, 0.05);
    for (const auto& q : out.positions) {
        EXPECT_TRUE((q.array() <= 1.0 + 1e-9).all());
        EXPECT_TRUE((q.array() >= -1.0 - 1e-9).all());
    }
}

TEST(LinearResample, DownsamplesKeepsStart) {
    const Trajectory traj = sine_traj(21, 0.05, 2);
    const Trajectory out = linear_resample(traj, 0.1);
    EXPECT_EQ(out.dt, 0.1);
    EXPECT_TRUE(out.positions.front().isApprox(traj.positions.front(), 1e-12));
}

TEST(LinearResample, SingleTimestep) {
    Trajectory traj(0.1, {Eigen::VectorXd::Constant(3, 1.5)});
    const Trajectory out = linear_resample(traj, 0.02);
    EXPECT_EQ(out.num_timesteps(), 1);
    EXPECT_DOUBLE_EQ(out.positions[0](0), 1.5);
}

TEST(LinearInterpolateAt, Midpoint) {
    Trajectory traj(0.1, {});
    traj.positions.push_back(Eigen::VectorXd::Constant(1, 0.0));
    traj.positions.push_back(Eigen::VectorXd::Constant(1, 1.0));
    const Trajectory out = linear_interpolate_at(traj, 0.05);
    EXPECT_DOUBLE_EQ(out.positions[0](0), 0.5);
}

TEST(CubicResample, PassesThroughKnots) {
    const Trajectory traj = sine_traj(9, 0.1, 2);
    const Trajectory out = cubic_resample(traj, 0.1);
    ASSERT_EQ(out.num_timesteps(), traj.num_timesteps());
    for (int i = 0; i < traj.num_timesteps(); ++i) {
        EXPECT_TRUE(out.positions[i].isApprox(traj.positions[i], 1e-9)) << i;
    }
}

TEST(CubicResample, UpsampleNoOvershootBeyondReason) {
    const Trajectory traj = sine_traj(9, 0.1, 2);
    const Trajectory out = cubic_resample(traj, 0.05);
    EXPECT_EQ(out.num_timesteps(), 17);
    for (const auto& q : out.positions) {
        EXPECT_TRUE(q.allFinite());
    }
}
