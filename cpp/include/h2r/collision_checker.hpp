#pragma once

#include <h2r/robot_model.hpp>
#include <h2r/trajectory.hpp>

#include <functional>

namespace h2r {

class CollisionChecker {
public:
    virtual ~CollisionChecker() = default;

    virtual bool configuration_valid(const Eigen::VectorXd& q) const = 0;

    virtual bool trajectory_valid(const Trajectory& traj) const;

    virtual double collision_cost(const Eigen::VectorXd& q) const = 0;
};

using ConfigCostFn = std::function<double(const Eigen::VectorXd&)>;

class FunctionalCollisionChecker : public CollisionChecker {
public:
    FunctionalCollisionChecker(ConfigCostFn cost_fn, double free_threshold);

    bool configuration_valid(const Eigen::VectorXd& q) const override;

    double collision_cost(const Eigen::VectorXd& q) const override;

private:
    ConfigCostFn cost_fn_;
    double free_threshold_;
};

class AlwaysValidChecker : public CollisionChecker {
public:
    bool configuration_valid(const Eigen::VectorXd& q) const override;

    double collision_cost(const Eigen::VectorXd& q) const override;
};

}  // namespace h2r
