#include <h2r/csv.hpp>

#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>
#include <string>

using namespace h2r;

namespace {

std::string write_temp(const std::string& content) {
    const std::string path =
        std::string(std::tmpnam(nullptr)) + "_h2r_test.csv";
    std::ofstream file(path);
    file << content;
    return path;
}

Trajectory sample_traj() {
    Trajectory traj(0.02, {});
    for (int i = 0; i < 4; ++i) {
        Eigen::VectorXd q(3);
        q << 0.1 * i, -0.2 * i, 0.3 * i;
        traj.positions.push_back(q);
    }
    return traj;
}

}  // namespace

TEST(CsvIO, RoundtripPreservesValues) {
    const Trajectory traj = sample_traj();
    const std::string path =
        std::string(std::tmpnam(nullptr)) + "_h2r_roundtrip.csv";
    save_csv(traj, path);
    const Trajectory loaded = load_csv(path, 0.02);
    ASSERT_EQ(loaded.num_timesteps(), 4);
    EXPECT_DOUBLE_EQ(loaded.dt, 0.02);
    for (int i = 0; i < 4; ++i) {
        EXPECT_TRUE(loaded.positions[i].isApprox(traj.positions[i], 1e-15));
    }
    std::remove(path.c_str());
}

TEST(CsvIO, HeaderSkippedWhenPresent) {
    const std::string path =
        write_temp("q0,q1\n0.1,0.2\n0.3,0.4\n");
    const Trajectory loaded = load_csv(path, 0.01, true);
    ASSERT_EQ(loaded.num_timesteps(), 2);
    EXPECT_DOUBLE_EQ(loaded.positions[1](1), 0.4);
    std::remove(path.c_str());
}

TEST(CsvIO, MissingFileThrows) {
    EXPECT_THROW(load_csv("definitely_missing_h2r_file.csv", 0.01),
                 std::runtime_error);
}

TEST(CsvIO, SaveHeaderMatchesDof) {
    const std::string path =
        std::string(std::tmpnam(nullptr)) + "_h2r_header.csv";
    save_csv(sample_traj(), path, true);
    std::ifstream file(path);
    std::string line;
    std::getline(file, line);
    EXPECT_EQ(line, "q0,q1,q2");
    std::remove(path.c_str());
}
