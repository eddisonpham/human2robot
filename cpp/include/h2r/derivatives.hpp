#pragma once

#include <h2r/trajectory.hpp>

namespace h2r {

Trajectory velocity(const Trajectory& traj);
Trajectory acceleration(const Trajectory& traj);
Trajectory jerk(const Trajectory& traj);

}  // namespace h2r
