#pragma once

#include <h2r/trajectory.hpp>

#include <string>
#include <vector>

namespace h2r {

Trajectory load_csv(const std::string& path, double dt);

Trajectory load_csv(const std::string& path, double dt, bool has_header);

void save_csv(const Trajectory& traj, const std::string& path);

void save_csv(const Trajectory& traj, const std::string& path, bool write_header);

std::vector<Trajectory> load_demo_directory(const std::string& dir, double dt);

}  // namespace h2r
