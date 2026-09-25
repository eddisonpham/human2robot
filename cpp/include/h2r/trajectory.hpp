#pragma once

#include <Eigen/Dense>
#include <vector>

namespace h2r {

struct JointLimits {
    Eigen::VectorXd lower;
    Eigen::VectorXd upper;

    int dof() const { return static_cast<int>(lower.size()); }
};

struct Trajectory {
    double dt = 0.01;
    std::vector<Eigen::VectorXd> positions;

    Trajectory() = default;
    Trajectory(double dt_in, std::vector<Eigen::VectorXd> positions_in)
        : dt(dt_in), positions(std::move(positions_in)) {}

    int num_timesteps() const { return static_cast<int>(positions.size()); }
    int dof() const { return positions.empty() ? 0 : static_cast<int>(positions[0].size()); }
    double duration() const {
        return num_timesteps() <= 1 ? 0.0 : (num_timesteps() - 1) * dt;
    }
};

}  // namespace h2r
