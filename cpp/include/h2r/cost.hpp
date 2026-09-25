#pragma once

#include <h2r/collision_checker.hpp>
#include <h2r/robot_model.hpp>
#include <h2r/trajectory.hpp>

namespace h2r {

struct CostWeights {
    double tracking = 1.0;
    double velocity = 0.0;
    double acceleration = 0.0;
    double jerk = 0.0;
    double collision = 0.0;
    double limits = 0.0;
};

struct LimitPenaltyScales {
    double velocity = 1.0;
    double acceleration = 1.0;
    double joint = 1.0;
};

double compute_cost(const Trajectory& traj,
                    const Trajectory& reference,
                    const CostWeights& weights,
                    const JointLimits* limits,
                    const Eigen::VectorXd* max_velocity,
                    const Eigen::VectorXd* max_acceleration,
                    const CollisionChecker* checker,
                    const RobotModel* model,
                    const LimitPenaltyScales& scales);

}  // namespace h2r
