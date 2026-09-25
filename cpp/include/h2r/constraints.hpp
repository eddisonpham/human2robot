#pragma once

#include <h2r/trajectory.hpp>

namespace h2r {

Trajectory project_joint_limits(const Trajectory& traj, const JointLimits& limits);

Trajectory project_velocity_limits(const Trajectory& traj,
                                   const Eigen::VectorXd& max_velocity);

Trajectory project_acceleration_limits(const Trajectory& traj,
                                       const Eigen::VectorXd& max_acceleration);

Trajectory project_all(const Trajectory& traj,
                       const JointLimits& limits,
                       const Eigen::VectorXd& max_velocity,
                       const Eigen::VectorXd& max_acceleration);

}  // namespace h2r
