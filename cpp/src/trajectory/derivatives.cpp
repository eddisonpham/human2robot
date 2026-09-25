#include <h2r/derivatives.hpp>

namespace h2r {

Trajectory velocity(const Trajectory& traj) {
    Trajectory out;
    out.dt = traj.dt;
    const int n = traj.num_timesteps();
    if (n == 0) {
        return out;
    }
    if (n == 1) {
        out.positions.push_back(Eigen::VectorXd::Zero(traj.dof()));
        return out;
    }
    out.positions.reserve(n);
    out.positions.push_back((traj.positions[1] - traj.positions[0]) / traj.dt);
    for (int i = 1; i < n - 1; ++i) {
        out.positions.push_back(
            (traj.positions[i + 1] - traj.positions[i - 1]) / (2.0 * traj.dt));
    }
    out.positions.push_back(
        (traj.positions[n - 1] - traj.positions[n - 2]) / traj.dt);
    return out;
}

Trajectory acceleration(const Trajectory& traj) {
    Trajectory out;
    out.dt = traj.dt;
    const int n = traj.num_timesteps();
    if (n == 0) {
        return out;
    }
    if (n < 3) {
        for (int i = 0; i < n; ++i) {
            out.positions.push_back(Eigen::VectorXd::Zero(traj.dof()));
        }
        return out;
    }
    out.positions.reserve(n);
    out.positions.push_back(Eigen::VectorXd::Zero(traj.dof()));
    for (int i = 1; i < n - 1; ++i) {
        out.positions.push_back((traj.positions[i + 1] - 2.0 * traj.positions[i] +
                                 traj.positions[i - 1]) /
                                (traj.dt * traj.dt));
    }
    out.positions.push_back(Eigen::VectorXd::Zero(traj.dof()));
    return out;
}

Trajectory jerk(const Trajectory& traj) {
    return velocity(acceleration(traj));
}

}  // namespace h2r
