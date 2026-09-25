#pragma once

#include <h2r/trajectory.hpp>

namespace h2r {

Trajectory moving_average_smooth(const Trajectory& traj, int window);

}  // namespace h2r
