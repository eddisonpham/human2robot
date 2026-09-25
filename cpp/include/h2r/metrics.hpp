#pragma once

#include <h2r/collision_checker.hpp>
#include <h2r/robot_model.hpp>
#include <h2r/trajectory.hpp>

namespace h2r {

struct TrajectoryMetrics {
    int joint_limit_violations = 0;
    int velocity_limit_violations = 0;
    int acceleration_limit_violations = 0;
    int invalid_configurations = 0;
    double max_velocity = 0.0;
    double max_acceleration = 0.0;
    double max_jerk = 0.0;
    double smoothness = 0.0;
    double tracking_error = 0.0;
    double ee_tracking_error = 0.0;
};

TrajectoryMetrics compute_metrics(const Trajectory& traj,
                                  const Trajectory* reference,
                                  const JointLimits* limits,
                                  const Eigen::VectorXd* max_velocity,
                                  const Eigen::VectorXd* max_acceleration,
                                  const CollisionChecker* checker,
                                  const RobotModel* model);

TrajectoryMetrics compute_metrics(const Trajectory& traj);

}  // namespace h2r
