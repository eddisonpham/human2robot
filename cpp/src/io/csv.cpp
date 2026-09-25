#include <h2r/csv.hpp>

#include <algorithm>
#include <charconv>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace h2r {

namespace {

std::vector<std::string> split_line(const std::string& line) {
    std::vector<std::string> parts;
    std::stringstream stream(line);
    std::string item;
    while (std::getline(stream, item, ',')) {
        parts.push_back(item);
    }
    return parts;
}

}  // namespace

Trajectory load_csv(const std::string& path, double dt, bool has_header) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("cannot open file: " + path);
    }
    Trajectory traj;
    traj.dt = dt;
    std::string line;
    bool first = true;
    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }
        if (first && has_header) {
            first = false;
            continue;
        }
        first = false;
        Eigen::VectorXd row;
        std::vector<std::string> parts = split_line(line);
        row.resize(static_cast<int>(parts.size()));
        for (int i = 0; i < row.size(); ++i) {
            row(i) = std::stod(parts[static_cast<size_t>(i)]);
        }
        traj.positions.push_back(row);
    }
    return traj;
}

Trajectory load_csv(const std::string& path, double dt) {
    return load_csv(path, dt, false);
}

void save_csv(const Trajectory& traj, const std::string& path, bool write_header) {
    std::ofstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("cannot open file: " + path);
    }
    if (traj.positions.empty()) {
        return;
    }
    if (write_header) {
        const int dof = traj.dof();
        for (int i = 0; i < dof; ++i) {
            file << "q" << i;
            if (i < dof - 1) {
                file << ",";
            }
        }
        file << "\n";
    }
    file.setf(std::ios::scientific);
    file.precision(17);
    for (const auto& q : traj.positions) {
        for (int i = 0; i < q.size(); ++i) {
            file << q(i);
            if (i < q.size() - 1) {
                file << ",";
            }
        }
        file << "\n";
    }
}

void save_csv(const Trajectory& traj, const std::string& path) {
    save_csv(traj, path, false);
}

std::vector<Trajectory> load_demo_directory(const std::string& dir, double dt) {
    (void)dir;
    (void)dt;
    return {};
}

}  // namespace h2r
