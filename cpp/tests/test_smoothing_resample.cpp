#include <h2r/interpolation.hpp>
#include <h2r/smoothing.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <random>

using namespace h2r;

namespace {

Trajectory noisy_traj(int n, double dt, int dof, unsigned seed) {
    Trajectory traj(dt, {});
    std::mt19937 gen(seed);
    std::normal_distribution<double> dist(0.0, 0.05);
    for (int i = 0; i < n; ++i) {
        Eigen::VectorXd q(dof);
        for (int j = 0; j < dof; ++j) {
            q(j) = std::sin(0.5 * i * dt) + dist(gen);
        }
        traj.positions.push_back(q);
    }
    return traj;
}

}  // namespace

TEST(Smoothing, PreservesLengthAndDt) {
    const Trajectory traj = noisy_traj(50, 0.01, 3, 1);
    const Trajectory out = moving_average_smooth(traj, 5);
    EXPECT_EQ(out.num_timesteps(), 50);
    EXPECT_DOUBLE_EQ(out.dt, 0.01);
}

TEST(Smoothing, ReducesHighFrequencyVariance) {
    const Trajectory traj = noisy_traj(100, 0.01, 2, 42);
    const Trajectory out = moving_average_smooth(traj, 7);
    auto variance = [](const Trajectory& t) {
        Eigen::VectorXd mean = Eigen::VectorXd::Zero(t.dof());
        for (const auto& q : t.positions) {
            mean += q;
        }
        mean /= static_cast<double>(t.num_timesteps());
        double var = 0.0;
        for (const auto& q : t.positions) {
            var += (q - mean).squaredNorm();
        }
        return var;
    };
    EXPECT_LT(variance(out), variance(traj));
}

TEST(Smoothing, WindowOneIsIdentity) {
    const Trajectory traj = noisy_traj(10, 0.01, 2, 7);
    const Trajectory out = moving_average_smooth(traj, 1);
    for (int i = 0; i < 10; ++i) {
        EXPECT_TRUE(out.positions[i].isApprox(traj.positions[i], 1e-12));
    }
}

TEST(ResampleRoundtrip, DownThenUpApproximatesOriginal) {
    Trajectory traj(0.01, {});
    for (int i = 0; i < 200; ++i) {
        traj.positions.push_back(
            Eigen::VectorXd::Constant(1, std::sin(0.05 * i * 0.01)));
    }
    const Trajectory down = linear_resample(traj, 0.05);
    EXPECT_EQ(down.num_timesteps(), 41);
    const Trajectory up = cubic_resample(down, 0.01);
    EXPECT_EQ(up.num_timesteps(), 201);
    for (int i = 0; i < 200; ++i) {
        EXPECT_NEAR(up.positions[i](0), traj.positions[i](0), 1e-3);
    }
}
