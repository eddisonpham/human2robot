#include <h2r/analytic_chain.hpp>

#include <cmath>
#include <numbers>

namespace h2r {

AnalyticChainModel::AnalyticChainModel(const std::vector<double>& link_lengths)
    : link_lengths_(link_lengths) {
    const int n = static_cast<int>(link_lengths_.size());
    limits_.lower = Eigen::VectorXd::Constant(n, -std::numbers::pi);
    limits_.upper = Eigen::VectorXd::Constant(n, std::numbers::pi);
}

int AnalyticChainModel::dof() const {
    return static_cast<int>(link_lengths_.size());
}

Eigen::VectorXd AnalyticChainModel::forward_kinematics(
    const Eigen::VectorXd& q) const {
    const int n = dof();
    double x = base_.x();
    double y = base_.y();
    double angle = 0.0;
    for (int i = 0; i < n; ++i) {
        angle += q(i);
        x += link_lengths_[i] * std::cos(angle);
        y += link_lengths_[i] * std::sin(angle);
    }
    Eigen::VectorXd ee(2);
    ee << x, y;
    return ee;
}

Eigen::MatrixXd AnalyticChainModel::jacobian(const Eigen::VectorXd& q) const {
    const int n = dof();
    Eigen::MatrixXd J = Eigen::MatrixXd::Zero(2, n);
    double x = base_.x();
    double y = base_.y();
    std::vector<double> angles(n);
    double angle = 0.0;
    for (int i = 0; i < n; ++i) {
        angle += q(i);
        angles[i] = angle;
        x += link_lengths_[i] * std::cos(angle);
        y += link_lengths_[i] * std::sin(angle);
    }
    for (int j = 0; j < n; ++j) {
        double dx = 0.0;
        double dy = 0.0;
        for (int i = j; i < n; ++i) {
            dx -= link_lengths_[i] * std::sin(angles[i]);
            dy += link_lengths_[i] * std::cos(angles[i]);
        }
        J(0, j) = dx;
        J(1, j) = dy;
    }
    return J;
}

JointLimits AnalyticChainModel::joint_limits() const {
    return limits_;
}

void AnalyticChainModel::set_joint_limits(const JointLimits& limits) {
    limits_ = limits;
}

void AnalyticChainModel::set_base_origin(const Eigen::Vector2d& origin) {
    base_ = origin;
}

}  // namespace h2r
