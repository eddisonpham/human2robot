#pragma once

#include <h2r/robot_model.hpp>

namespace h2r {

class AnalyticChainModel : public RobotModel {
public:
    explicit AnalyticChainModel(const std::vector<double>& link_lengths);

    int dof() const override;

    Eigen::VectorXd forward_kinematics(const Eigen::VectorXd& q) const override;

    Eigen::MatrixXd jacobian(const Eigen::VectorXd& q) const override;

    JointLimits joint_limits() const override;

    void set_joint_limits(const JointLimits& limits);

    void set_base_origin(const Eigen::Vector2d& origin);

private:
    std::vector<double> link_lengths_;
    JointLimits limits_;
    Eigen::Vector2d base_{0.0, 0.0};
};

}  // namespace h2r
