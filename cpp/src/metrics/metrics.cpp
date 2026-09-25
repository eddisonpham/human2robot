#include <h2r/metrics.hpp>

#include <h2r/derivatives.hpp>

#include <algorithm>
#include <cmath>

namespace h2r {

namespace {

int count_joint_violations(const Trajectory& traj, const JointLimits& limits) {
    int count = 0;
    for (const auto& q : traj.positions) {
        const Eigen::ArrayXd below = (limits.lower.array() - q.array()).max(0.0);
        const Eigen::ArrayXd above = (q.array() - limits.upper.array()).max(0.0);
        if ((below + above).sum() > 1e-9) {
            ++count;
        }
    }
    return count;
}

int count_component_violations(const std::vector<Eigen::VectorXd>& values,
                               const Eigen::VectorXd& bounds) {
    int count = 0;
    for (const auto& v : values) {
        if ((v.array().abs() > bounds.array()).any()) {
            ++count;
        }
    }
    return count;
}

}  // namespace

TrajectoryMetrics compute_metrics(const Trajectory& traj,
                                  const Trajectory* reference,
                                  const JointLimits* limits,
                                  const Eigen::VectorXd* max_velocity,
                                  const Eigen::VectorXd* max_acceleration,
                                  const CollisionChecker* checker,
                                  const RobotModel* model) {
    TrajectoryMetrics m;
    const Trajectory vel = velocity(traj);
    const Trajectory acc = acceleration(traj);
    const Trajectory jrk = jerk(traj);

    auto max_abs = [](const Trajectory& t) {
        double worst = 0.0;
        for (const auto& v : t.positions) {
            worst = std::max(worst, v.array().abs().maxCoeff());
        }
        return worst;
    };
    m.max_velocity = max_abs(vel);
    m.max_acceleration = max_abs(acc);
    m.max_jerk = max_abs(jrk);

    double smoothness = 0.0;
    for (const auto& v : jrk.positions) {
        smoothness += v.squaredNorm();
    }
    m.smoothness = smoothness * traj.dt;

    if (limits != nullptr) {
        m.joint_limit_violations = count_joint_violations(traj, *limits);
    }
    if (max_velocity != nullptr) {
        m.velocity_limit_violations =
            count_component_violations(vel.positions, *max_velocity);
    }
    if (max_acceleration != nullptr) {
        m.acceleration_limit_violations =
            count_component_violations(acc.positions, *max_acceleration);
    }
    if (checker != nullptr) {
        for (const auto& q : traj.positions) {
            if (!checker->configuration_valid(q)) {
                ++m.invalid_configurations;
            }
        }
    }
    if (reference != nullptr && reference->num_timesteps() == traj.num_timesteps()) {
        double total = 0.0;
        for (int i = 0; i < traj.num_timesteps(); ++i) {
            total += (traj.positions[i] - reference->positions[i]).norm();
        }
        m.tracking_error = total / static_cast<double>(traj.num_timesteps());
    }
    if (model != nullptr && reference != nullptr &&
        reference->num_timesteps() == traj.num_timesteps()) {
        double total = 0.0;
        for (int i = 0; i < traj.num_timesteps(); ++i) {
            const Eigen::VectorXd ee = model->forward_kinematics(traj.positions[i]);
            const Eigen::VectorXd ee_ref =
                model->forward_kinematics(reference->positions[i]);
            total += (ee - ee_ref).norm();
        }
        m.ee_tracking_error = total / static_cast<double>(traj.num_timesteps());
    }
    return m;
}

TrajectoryMetrics compute_metrics(const Trajectory& traj) {
    return compute_metrics(traj, nullptr, nullptr, nullptr, nullptr, nullptr,
                           nullptr);
}

}  // namespace h2r
