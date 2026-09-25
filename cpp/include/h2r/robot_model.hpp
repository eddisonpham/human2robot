#pragma once

#include <h2r/trajectory.hpp>

namespace h2r {

class RobotModel {
public:
    virtual ~RobotModel() = default;

    virtual int dof() const = 0;

    virtual Eigen::VectorXd forward_kinematics(const Eigen::VectorXd& q) const = 0;

    virtual Eigen::MatrixXd jacobian(const Eigen::VectorXd& q) const = 0;

    virtual JointLimits joint_limits() const = 0;
};

}  // namespace h2r
