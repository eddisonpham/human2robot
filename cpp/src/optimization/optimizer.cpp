#include <h2r/optimizer.hpp>

#include <h2r/constraints.hpp>

#include <cmath>
#include <random>

namespace h2r {

TrajectoryOptimizer::TrajectoryOptimizer(OptimizerConfig config)
    : config_(std::move(config)) {}

void TrajectoryOptimizer::set_collision_checker(const CollisionChecker* checker) {
    checker_ = checker;
}

OptimizerResult TrajectoryOptimizer::optimize(const Trajectory& initial,
                                              const Trajectory& reference) const {
    OptimizerResult result;
    result.trajectory = initial;
    result.initial_cost =
        compute_cost(result.trajectory, reference, config_.weights,
                     &config_.limits, &config_.max_velocity,
                     &config_.max_acceleration, checker_, nullptr,
                     config_.scales);
    result.final_cost = result.initial_cost;

    if (initial.positions.empty()) {
        return result;
    }

    Trajectory current = project_all(initial, config_.limits,
                                     config_.max_velocity,
                                     config_.max_acceleration);
    double current_cost =
        compute_cost(current, reference, config_.weights, &config_.limits,
                     &config_.max_velocity, &config_.max_acceleration, checker_,
                     nullptr, config_.scales);

    Trajectory best = current;
    double best_cost = current_cost;

    std::mt19937 gen(static_cast<unsigned>(config_.seed));
    std::normal_distribution<double> dist(0.0, 1.0);

    const double improvement_target = result.initial_cost * 0.01;
    const bool same_length =
        reference.num_timesteps() == current.num_timesteps();
    for (int iter = 0; iter < config_.max_iterations; ++iter) {
        Trajectory candidate = current;
        const double alpha =
            config_.step_size * std::pow(0.99, static_cast<double>(iter));
        for (int i = 0; i < candidate.num_timesteps(); ++i) {
            Eigen::VectorXd noise = Eigen::VectorXd::Zero(candidate.dof());
            for (int j = 0; j < candidate.dof(); ++j) {
                noise(j) = dist(gen);
            }
            if (same_length) {
                candidate.positions[i] +=
                    alpha * (reference.positions[i] - candidate.positions[i])
                    - alpha * 0.1 * noise;
            } else {
                candidate.positions[i] -= alpha * noise;
            }
        }
        candidate = project_all(candidate, config_.limits, config_.max_velocity,
                                config_.max_acceleration);
        const double candidate_cost =
            compute_cost(candidate, reference, config_.weights, &config_.limits,
                         &config_.max_velocity, &config_.max_acceleration,
                         checker_, nullptr, config_.scales);
        result.iterations = iter + 1;
        if (candidate_cost < current_cost) {
            current = candidate;
            current_cost = candidate_cost;
            if (candidate_cost < best_cost) {
                best = candidate;
                best_cost = candidate_cost;
            }
            if (result.initial_cost - current_cost >=
                config_.convergence_tolerance + improvement_target) {
                result.converged = true;
                break;
            }
        } else if (candidate_cost > current_cost * 2.0) {
            break;
        }
    }

    result.trajectory = best;
    result.final_cost = best_cost;
    if (result.initial_cost - best_cost >= config_.convergence_tolerance) {
        result.converged = true;
    }
    return result;
}

}  // namespace h2r
