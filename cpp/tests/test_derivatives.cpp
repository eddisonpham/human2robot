#include <h2r/derivatives.hpp>

#include <gtest/gtest.h>

#include <cmath>

using namespace h2r;

namespace {

Trajectory constant_traj(int n, double dt, double value, int dof) {
    Trajectory traj(dt, {});
    for (int i = 0; i < n; ++i) {
        traj.positions.push_back(Eigen::VectorXd::Constant(dof, value));
    }
    return traj;
}

Trajectory ramp_traj(int n, double dt, double slope, int dof) {
    Trajectory traj(dt, {});
    for (int i = 0; i < n; ++i) {
        traj.positions.push_back(Eigen::VectorXd::Constant(dof, slope * i * dt));
    }
    return traj;
}

Trajectory parabola_traj(int n, double dt, double a, int dof) {
    Trajectory traj(dt, {});
    for (int i = 0; i < n; ++i) {
        const double t = i * dt;
        traj.positions.push_back(
            Eigen::VectorXd::Constant(dof, a * t * t));
    }
    return traj;
}

}  // namespace

TEST(Derivatives, ConstantHasZeroVelocity) {
    const Trajectory vel = velocity(constant_traj(10, 0.01, 2.0, 3));
    ASSERT_EQ(vel.num_timesteps(), 10);
    for (const auto& v : vel.positions) {
        EXPECT_TRUE(v.isZero(1e-12));
    }
}

TEST(Derivatives, RampVelocityEqualsSlope) {
    const double slope = 1.7;
    const Trajectory vel = velocity(ramp_traj(20, 0.01, slope, 2));
    for (int i = 1; i < 19; ++i) {
        EXPECT_NEAR(vel.positions[i](0), slope, 1e-9);
    }
}

TEST(Derivatives, ParabolaAccelerationEqualsTwiceA) {
    const double a = 0.8;
    const Trajectory acc = acceleration(parabola_traj(30, 0.01, a, 1));
    for (int i = 1; i < 29; ++i) {
        EXPECT_NEAR(acc.positions[i](0), 2.0 * a, 1e-7);
    }
}

TEST(Derivatives, ParabolaJerkNearZeroInterior) {
    const Trajectory jrk = jerk(parabola_traj(30, 0.01, 0.5, 1));
    for (int i = 3; i < 27; ++i) {
        EXPECT_NEAR(jrk.positions[i](0), 0.0, 1e-5);
    }
}

TEST(Derivatives, LengthAndDofPreserved) {
    const Trajectory traj = parabola_traj(15, 0.02, 1.0, 4);
    EXPECT_EQ(velocity(traj).num_timesteps(), 15);
    EXPECT_EQ(acceleration(traj).num_timesteps(), 15);
    EXPECT_EQ(jerk(traj).num_timesteps(), 15);
    EXPECT_EQ(jerk(traj).dof(), 4);
}

TEST(Derivatives, ShortTrajectoriesSafe) {
    Trajectory traj(0.01, {Eigen::VectorXd::Constant(2, 1.0)});
    EXPECT_EQ(velocity(traj).num_timesteps(), 1);
    EXPECT_EQ(acceleration(traj).num_timesteps(), 1);
    Trajectory empty;
    EXPECT_EQ(velocity(empty).num_timesteps(), 0);
    EXPECT_EQ(acceleration(empty).num_timesteps(), 0);
}
