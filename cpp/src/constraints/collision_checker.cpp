#include <h2r/collision_checker.hpp>

namespace h2r {

bool CollisionChecker::trajectory_valid(const Trajectory& traj) const {
    for (const auto& q : traj.positions) {
        if (!configuration_valid(q)) {
            return false;
        }
    }
    return true;
}

FunctionalCollisionChecker::FunctionalCollisionChecker(ConfigCostFn cost_fn,
                                                       double free_threshold)
    : cost_fn_(std::move(cost_fn)), free_threshold_(free_threshold) {}

bool FunctionalCollisionChecker::configuration_valid(
    const Eigen::VectorXd& q) const {
    return cost_fn_(q) <= free_threshold_;
}

double FunctionalCollisionChecker::collision_cost(
    const Eigen::VectorXd& q) const {
    return cost_fn_(q);
}

bool AlwaysValidChecker::configuration_valid(const Eigen::VectorXd&) const {
    return true;
}

double AlwaysValidChecker::collision_cost(const Eigen::VectorXd&) const {
    return 0.0;
}

}  // namespace h2r
