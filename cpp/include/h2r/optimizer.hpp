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
    int num_threads = 1;
    int seed = 0;
};

struct OptimizerResult {
    Trajectory trajectory;
    double initial_cost = 0.0;
    double final_cost = 0.0;
    int iterations = 0;
    bool converged = false;
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
