#include <h2r/constraints.hpp>

namespace h2r {

Trajectory project_joint_limits(const Trajectory& traj,
                                const JointLimits& limits) {
    Trajectory out = traj;
    if (limits.dof() == 0) {
        return out;
    }
    for (auto& q : out.positions) {
        if (q.size() != limits.dof()) {
            continue;
        }
        q = q.cwiseMax(limits.lower).cwiseMin(limits.upper);
    }
    return out;
}

Trajectory project_velocity_limits(const Trajectory& traj,
                                   const Eigen::VectorXd& max_velocity) {
    Trajectory out = traj;
    const int n = out.num_timesteps();
    if (n < 2 || max_velocity.size() == 0) {
        return out;
    }
    for (int i = 1; i < n; ++i) {
        const Eigen::VectorXd prev = out.positions[i - 1];
        Eigen::VectorXd delta = out.positions[i] - prev;
        const Eigen::VectorXd bound = max_velocity * out.dt;
        for (int j = 0; j < delta.size(); ++j) {
            const double lo = -bound(j);
            const double hi = bound(j);
            delta(j) = std::clamp(delta(j), lo, hi);
        }
        out.positions[i] = prev + delta;
    }
    return out;
}

Trajectory project_acceleration_limits(const Trajectory& traj,
                                       const Eigen::VectorXd& max_acceleration) {
    Trajectory out = traj;
    const int n = out.num_timesteps();
    if (n < 3 || max_acceleration.size() == 0) {
        return out;
    }
    const Eigen::VectorXd vel_bound = max_acceleration * out.dt * out.dt;
    for (int i = 1; i < n - 1; ++i) {
        const Eigen::VectorXd& prev = out.positions[i - 1];
        Eigen::VectorXd delta = out.positions[i + 1] - 2.0 * out.positions[i] + prev;
        for (int j = 0; j < delta.size(); ++j) {
            const double lo = -vel_bound(j);
            const double hi = vel_bound(j);
            delta(j) = std::clamp(delta(j), lo, hi);
        }
        out.positions[i + 1] = 2.0 * out.positions[i] - prev + delta;
    }
    return out;
}

Trajectory project_all(const Trajectory& traj,
                       const JointLimits& limits,
                       const Eigen::VectorXd& max_velocity,
                       const Eigen::VectorXd& max_acceleration) {
    Trajectory out = project_joint_limits(traj, limits);
    out = project_velocity_limits(out, max_velocity);
    out = project_acceleration_limits(out, max_acceleration);
    out = project_velocity_limits(out, max_velocity);
    out = project_joint_limits(out, limits);
    return out;
}

}  // namespace h2r
