#pragma once

#include <h2r/collision_checker.hpp>
#include <h2r/cost.hpp>
#include <h2r/robot_model.hpp>
#include <h2r/trajectory.hpp>

namespace h2r {

struct OptimizerConfig {
    CostWeights weights;
    JointLimits limits;
    Eigen::VectorXd max_velocity;
    Eigen::VectorXd max_acceleration;
    LimitPenaltyScales scales;
    int max_iterations = 100;
    double convergence_tolerance = 1e-4;
    double step_size = 0.1;
    // Scale of the per-timestep Gaussian perturbation, relative to step_size.
    // Independent per-sample noise is adversarial for a cost dominated by
    // smoothness terms, because it raises jerk far faster than the tracking
    // pull lowers it. Set to 0 for a pure gradient-like tracking step.
    double noise_scale = 0.1;
    int num_threads = 1;
    int seed = 0;
};

struct OptimizerResult {
    Trajectory trajectory;
    // Cost of the input as given, before constraint projection.
    double initial_cost = 0.0;
    // Cost of the projected input, which is where the search begins. Comparing
    // final_cost against this, not against initial_cost, is what makes
    // convergence and improvement_pct meaningful.
    double projected_initial_cost = 0.0;
    // Best cost found by the search, never above projected_initial_cost.
    double final_cost = 0.0;
    int iterations = 0;
    bool converged = false;
    // Relative cost reduction the search achieved against its starting point.
    // Zero means the search made no progress at all.
    double improvement_pct = 0.0;
};

class TrajectoryOptimizer {
public:
    explicit TrajectoryOptimizer(OptimizerConfig config);

    void set_collision_checker(const CollisionChecker* checker);

    OptimizerResult optimize(const Trajectory& initial,
                             const Trajectory& reference) const;

private:
    OptimizerConfig config_;
    const CollisionChecker* checker_ = nullptr;
};

}  // namespace h2r
