#include <h2r/smoothing.hpp>

#include <algorithm>

namespace h2r {

Trajectory moving_average_smooth(const Trajectory& traj, int window) {
    Trajectory out;
    out.dt = traj.dt;
    if (traj.positions.empty() || window <= 1) {
        return traj;
    }
    const int n = traj.num_timesteps();
    const int half = std::max(1, window / 2);
    out.positions.reserve(n);
    for (int i = 0; i < n; ++i) {
        const int lo = std::max(0, i - half);
        const int hi = std::min(n - 1, i + half);
        Eigen::VectorXd acc = Eigen::VectorXd::Zero(traj.dof());
        for (int j = lo; j <= hi; ++j) {
            acc += traj.positions[j];
        }
        out.positions.push_back(acc / static_cast<double>(hi - lo + 1));
    }
    return out;
}

}  // namespace h2r
