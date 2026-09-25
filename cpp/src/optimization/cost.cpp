#include <h2r/cost.hpp>

#include <h2r/derivatives.hpp>

#include <algorithm>
#include <cmath>

namespace h2r {

namespace {

double tracking_cost(const Trajectory& traj, const Trajectory& reference) {
    if (reference.positions.empty() || traj.positions.empty()) {
        return 0.0;
    }
    double total = 0.0;
    const int n = std::min(traj.num_timesteps(), reference.num_timesteps());
    for (int i = 0; i < n; ++i) {
        const Eigen::VectorXd diff = traj.positions[i] - reference.positions[i];
        total += diff.squaredNorm();
    }
    return total / static_cast<double>(n);
}

double squared_magnitude_cost(const Trajectory& values) {
    if (values.positions.empty()) {
        return 0.0;
    }
    double total = 0.0;
    for (const auto& v : values.positions) {
        total += v.squaredNorm();
    }
    return total / static_cast<double>(values.positions.size());
}

double violation_cost(const std::vector<Eigen::VectorXd>& values,
                      const Eigen::VectorXd& bounds, double scale) {
    if (values.empty() || bounds.size() == 0) {
        return 0.0;
    }
    double total = 0.0;
    for (const auto& v : values) {
        const Eigen::VectorXd excess =
            (v.array().abs() - bounds.array()).max(0.0);
        total += excess.squaredNorm();
    }
    return scale * total / static_cast<double>(values.size());
}

double joint_violation_cost(const Trajectory& traj, const JointLimits& limits,
                            double scale) {
    if (traj.positions.empty() || limits.dof() == 0) {
        return 0.0;
    }
    double total = 0.0;
    for (const auto& q : traj.positions) {
        const Eigen::VectorXd below =
            (limits.lower.array() - q.array()).max(0.0);
        const Eigen::VectorXd above =
            (q.array() - limits.upper.array()).max(0.0);
        total += below.squaredNorm() + above.squaredNorm();
    }
    return scale * total / static_cast<double>(traj.positions.size());
}

}  // namespace

double compute_cost(const Trajectory& traj,
                    const Trajectory& reference,
                    const CostWeights& weights,
                    const JointLimits* limits,
                    const Eigen::VectorXd* max_velocity,
                    const Eigen::VectorXd* max_acceleration,
                    const CollisionChecker* checker,
                    const RobotModel* model,
                    const LimitPenaltyScales& scales) {
    double cost = 0.0;
    cost += weights.tracking * tracking_cost(traj, reference);
    if (weights.velocity > 0.0) {
        cost += weights.velocity * squared_magnitude_cost(velocity(traj));
    }
    if (weights.acceleration > 0.0) {
        cost += weights.acceleration * squared_magnitude_cost(acceleration(traj));
    }
    if (weights.jerk > 0.0) {
        cost += weights.jerk * squared_magnitude_cost(jerk(traj));
    }
    if (weights.limits > 0.0) {
        if (limits != nullptr) {
            cost += weights.limits *
                    joint_violation_cost(traj, *limits, scales.joint);
        }
        if (max_velocity != nullptr) {
            cost += weights.limits *
                    violation_cost(velocity(traj).positions, *max_velocity,
                                   scales.velocity);
        }
        if (max_acceleration != nullptr) {
            cost += weights.limits *
                    violation_cost(acceleration(traj).positions,
                                   *max_acceleration, scales.acceleration);
        }
    }
    if (weights.collision > 0.0 && checker != nullptr && !traj.positions.empty()) {
        double total = 0.0;
        for (const auto& q : traj.positions) {
            total += checker->collision_cost(q);
        }
        cost += weights.collision * total / static_cast<double>(traj.positions.size());
    }
    (void)model;
    return cost;
}

}  // namespace h2r
