#include <h2r/interpolation.hpp>

#include <algorithm>
#include <cmath>

namespace h2r {

namespace {

Eigen::VectorXd lerp(const Eigen::VectorXd& a, const Eigen::VectorXd& b, double t) {
    return a + t * (b - a);
}

int clamp_index(int idx, int size) {
    return std::clamp(idx, 0, size - 1);
}

}  // namespace

Trajectory linear_interpolate_at(const Trajectory& traj, double time) {
    Trajectory out;
    out.dt = traj.dt;
    if (traj.positions.empty()) {
        return out;
    }
    const int n = traj.num_timesteps();
    const double raw = time / traj.dt;
    const int i0 = clamp_index(static_cast<int>(std::floor(raw)), n);
    const int i1 = clamp_index(i0 + 1, n);
    const double t = std::clamp(raw - static_cast<double>(i0), 0.0, 1.0);
    out.positions.push_back(lerp(traj.positions[i0], traj.positions[i1], t));
    return out;
}

Trajectory linear_resample(const Trajectory& traj, double target_dt) {
    Trajectory out;
    out.dt = target_dt;
    if (traj.positions.empty()) {
        return out;
    }
    const int n = traj.num_timesteps();
    if (n == 1) {
        out.positions.push_back(traj.positions[0]);
        return out;
    }
    const double duration = traj.duration();
    const int target_n =
        std::max(1, static_cast<int>(std::ceil(duration / target_dt)) + 1);
    out.positions.reserve(target_n);
    int cursor = 0;
    for (int i = 0; i < target_n; ++i) {
        const double time = i * target_dt;
        while (cursor < n - 2 && (cursor + 1) * traj.dt < time) {
            ++cursor;
        }
        const double seg_start = cursor * traj.dt;
        const double seg_end = (cursor + 1) * traj.dt;
        double t = 0.0;
        if (seg_end > seg_start) {
            t = std::clamp((time - seg_start) / (seg_end - seg_start), 0.0, 1.0);
        }
        out.positions.push_back(
            lerp(traj.positions[cursor], traj.positions[cursor + 1], t));
    }
    return out;
}

namespace {

Eigen::VectorXd cubic_hermite(const Eigen::VectorXd& p0, const Eigen::VectorXd& p1,
                              const Eigen::VectorXd& m0, const Eigen::VectorXd& m1,
                              double dt, double t) {
    const double t2 = t * t;
    const double t3 = t2 * t;
    const double h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
    const double h10 = (t3 - 2.0 * t2 + t) * dt;
    const double h01 = -2.0 * t3 + 3.0 * t2;
    const double h11 = (t3 - t2) * dt;
    return h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1;
}

}  // namespace

Trajectory cubic_resample(const Trajectory& traj, double target_dt) {
    Trajectory out;
    out.dt = target_dt;
    if (traj.positions.empty()) {
        return out;
    }
    const int n = traj.num_timesteps();
    if (n < 3) {
        return linear_resample(traj, target_dt);
    }
    const double duration = traj.duration();
    const int target_n =
        std::max(1, static_cast<int>(std::ceil(duration / target_dt)) + 1);
    out.positions.reserve(target_n);

    std::vector<Eigen::VectorXd> slopes(n);
    slopes[0] = (traj.positions[1] - traj.positions[0]) / traj.dt;
    slopes[n - 1] = (traj.positions[n - 1] - traj.positions[n - 2]) / traj.dt;
    for (int i = 1; i < n - 1; ++i) {
        slopes[i] = (traj.positions[i + 1] - traj.positions[i - 1]) /
                    (2.0 * traj.dt);
    }

    int cursor = 0;
    for (int i = 0; i < target_n; ++i) {
        const double time = i * target_dt;
        while (cursor < n - 2 && (cursor + 1) * traj.dt < time) {
            ++cursor;
        }
        const double seg_start = cursor * traj.dt;
        const double seg_end = (cursor + 1) * traj.dt;
        double t = 0.0;
        if (seg_end > seg_start) {
            t = std::clamp((time - seg_start) / (seg_end - seg_start), 0.0, 1.0);
        }
        out.positions.push_back(cubic_hermite(
            traj.positions[cursor], traj.positions[cursor + 1], slopes[cursor],
            slopes[cursor + 1], traj.dt, t));
    }
    return out;
}

}  // namespace h2r
