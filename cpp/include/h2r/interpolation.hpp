#pragma once

#include <h2r/trajectory.hpp>

namespace h2r {

Trajectory linear_resample(const Trajectory& traj, double target_dt);

Trajectory cubic_resample(const Trajectory& traj, double target_dt);

Trajectory linear_interpolate_at(const Trajectory& traj, double time);

}  // namespace h2r
