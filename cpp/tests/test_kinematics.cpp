#include <h2r/analytic_chain.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <numbers>

using namespace h2r;

namespace {

AnalyticChainModel two_link() {
    return AnalyticChainModel(std::vector<double>{1.0, 1.0});
}

}  // namespace

TEST(AnalyticChain, DofMatchesLinks) {
    const auto model = two_link();
    EXPECT_EQ(model.dof(), 2);
}

TEST(AnalyticChain, ZeroConfigEndEffector) {
    const auto model = two_link();
    const Eigen::VectorXd q = Eigen::VectorXd::Zero(2);
    const Eigen::VectorXd ee = model.forward_kinematics(q);
    EXPECT_NEAR(ee(0), 2.0, 1e-12);
    EXPECT_NEAR(ee(1), 0.0, 1e-12);
}

TEST(AnalyticChain, NinetyDegreeElbow) {
    const auto model = two_link();
    Eigen::VectorXd q(2);
    q << 0.0, std::numbers::pi / 2.0;
    const Eigen::VectorXd ee = model.forward_kinematics(q);
    EXPECT_NEAR(ee(0), 1.0, 1e-12);
    EXPECT_NEAR(ee(1), 1.0, 1e-12);
}

TEST(AnalyticChain, JacobianDimension) {
    const auto model = two_link();
    const Eigen::MatrixXd J = model.jacobian(Eigen::VectorXd::Zero(2));
    EXPECT_EQ(J.rows(), 2);
    EXPECT_EQ(J.cols(), 2);
}

TEST(AnalyticChain, JacobianMatchesFiniteDifference) {
    const auto model = AnalyticChainModel(std::vector<double>{0.5, 0.8, 1.1});
    Eigen::VectorXd q(3);
    q << 0.3, -0.7, 1.2;
    const Eigen::MatrixXd J = model.jacobian(q);
    const double eps = 1e-7;
    for (int j = 0; j < 3; ++j) {
        Eigen::VectorXd qp = q, qm = q;
        qp(j) += eps;
        qm(j) -= eps;
        const Eigen::VectorXd dp =
            (model.forward_kinematics(qp) - model.forward_kinematics(qm)) /
            (2.0 * eps);
        EXPECT_NEAR(J(0, j), dp(0), 1e-6) << "row0 col" << j;
        EXPECT_NEAR(J(1, j), dp(1), 1e-6) << "row1 col" << j;
    }
}

TEST(AnalyticChain, DefaultLimitsSymmetricPi) {
    const auto model = two_link();
    const JointLimits limits = model.joint_limits();
    EXPECT_TRUE(limits.lower.isApprox(Eigen::VectorXd::Constant(2, -std::numbers::pi), 1e-12));
    EXPECT_TRUE(limits.upper.isApprox(Eigen::VectorXd::Constant(2, std::numbers::pi), 1e-12));
}

TEST(AnalyticChain, CustomLimitsRespected) {
    auto model = two_link();
    JointLimits limits;
    limits.lower = Eigen::VectorXd::Constant(2, -0.5);
    limits.upper = Eigen::VectorXd::Constant(2, 0.5);
    model.set_joint_limits(limits);
    const JointLimits got = model.joint_limits();
    EXPECT_DOUBLE_EQ(got.upper(1), 0.5);
}

TEST(RobotModelInterface, PolymorphicUse) {
    std::unique_ptr<h2r::RobotModel> model =
        std::make_unique<h2r::AnalyticChainModel>(std::vector<double>{1.0});
    Eigen::VectorXd q(1);
    q << std::numbers::pi / 2.0;
    const Eigen::VectorXd ee = model->forward_kinematics(q);
    EXPECT_NEAR(ee(1), 1.0, 1e-12);
}
