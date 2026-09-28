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
    const double projected_initial_cost = current_cost;
    result.projected_initial_cost = projected_initial_cost;

    std::mt19937 gen(static_cast<unsigned>(config_.seed));
    std::normal_distribution<double> dist(0.0, 1.0);

    // The improvement bar is a fraction of the cost the search starts from,
    // which is the projected input. Using the unprojected cost here would set
    // the bar against a baseline projection alone usually beats, so a search
    // that did nothing could still satisfy it.
    const double improvement_target = projected_initial_cost * 0.01;
    const bool same_length =
        reference.num_timesteps() == current.num_timesteps();

    // Backtracking line search. The step direction mixes a pull toward the
    // reference with per-timestep Gaussian noise, and the cost is dominated by
    // smoothness terms, so a fixed-size step almost always overshoots: the
    // independent per-sample noise raises jerk far more than the tracking pull
    // lowers it. At the original fixed step this made the first candidate
    // exceed twice the current cost on most inputs, which tripped the abort
    // guard at iteration 1 and left the search with zero improvement. Halving
    // the step until the cost actually falls keeps the search descending
    // instead of aborting, and costs nothing when the first trial already
    // works.
    constexpr int kMaxBacktracks = 24;
    for (int iter = 0; iter < config_.max_iterations; ++iter) {
        const double base_alpha =
            config_.step_size * std::pow(0.99, static_cast<double>(iter));

        Trajectory accepted = current;
        double accepted_cost = current_cost;
        bool found_descent = false;
        for (int trial = 0; trial < kMaxBacktracks; ++trial) {
            const double alpha =
                base_alpha * std::pow(0.5, static_cast<double>(trial));
            Trajectory candidate = current;
            for (int i = 0; i < candidate.num_timesteps(); ++i) {
                Eigen::VectorXd noise = Eigen::VectorXd::Zero(candidate.dof());
                for (int j = 0; j < candidate.dof(); ++j) {
                    noise(j) = dist(gen);
                }
                if (same_length) {
                    candidate.positions[i] +=
                        alpha * (reference.positions[i] - candidate.positions[i])
                        - alpha * config_.noise_scale * noise;
                } else {
                    candidate.positions[i] -= alpha * config_.noise_scale * noise;
                }
            }
            candidate = project_all(candidate, config_.limits,
                                    config_.max_velocity,
                                    config_.max_acceleration);
            const double candidate_cost =
                compute_cost(candidate, reference, config_.weights,
                             &config_.limits, &config_.max_velocity,
                             &config_.max_acceleration, checker_, nullptr,
                             config_.scales);
            if (candidate_cost < current_cost) {
                accepted = candidate;
                accepted_cost = candidate_cost;
                found_descent = true;
                break;
            }
        }
        result.iterations = iter + 1;
        if (!found_descent) {
            break;
        }
        current = accepted;
        current_cost = accepted_cost;
        if (current_cost < best_cost) {
            best = current;
            best_cost = current_cost;
        }
        if (projected_initial_cost - current_cost >=
            config_.convergence_tolerance + improvement_target) {
            result.converged = true;
            break;
        }
    }

    result.trajectory = best;
    result.final_cost = best_cost;
    // Convergence is measured against the projected starting point, which is
    // where the search actually begins. Comparing against the unprojected
    // input is wrong whenever projection itself raises the cost, because the
    // limit-violation penalty can exceed the smoothness the projection adds.
    // Those sequences would then report final_cost above initial_cost and
    // could never satisfy the tolerance, even though the optimizer improved on
    // its own starting point and never took a non-descending step.
    if (projected_initial_cost - best_cost >= config_.convergence_tolerance) {
        result.converged = true;
    }
    if (projected_initial_cost > 0.0) {
        result.improvement_pct =
            100.0 * (1.0 - best_cost / projected_initial_cost);
    }
    return result;
}

}  // namespace h2r
