#pragma once
#include <memory>
#include <string>

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include "drake/common/test_utilities/maybe_pause_for_user.h"
#include "drake/geometry/meshcat.h"
#include "drake/geometry/optimization/hpolyhedron.h"
#include "drake/geometry/optimization/hyperellipsoid.h"
#include "drake/geometry/optimization/vpolytope.h"
#include "drake/geometry/test_utilities/meshcat_environment.h"

namespace drake {
namespace planning {

using common::MaybePauseForUser;
using Eigen::Vector2d;
using Eigen::VectorXd;
using geometry::Meshcat;
using geometry::Rgba;
using geometry::optimization::HPolyhedron;
using geometry::optimization::Hyperellipsoid;
using geometry::optimization::VPolytope;
using symbolic::Variable;

// Reproduced from the IrisInConfigurationSpace unit tests.
// One prismatic link with joint limits.  Iris should return the joint limits.
class JointLimits : public ::testing::Test {
 public:
  const std::string urdf = R"(
<robot name="limits">
  <link name="movable">
    <collision>
      <geometry><box size="1 1 1"/></geometry>
    </collision>
  </link>
  <joint name="movable" type="prismatic">
    <axis xyz="1 0 0"/>
    <limit lower="-2" upper="2"/>
    <parent link="world"/>
    <child link="movable"/>
  </joint>
</robot>
)";

  const Vector1d sample = Vector1d::Zero();
  Hyperellipsoid starting_ellipsoid =
      Hyperellipsoid::MakeHypersphere(1e-2, sample);

 protected:
  void run_checks_on_region(const HPolyhedron& region) {
    const double kTol = 1e-5;
    const double qmin = -2.0, qmax = 2.0;
    EXPECT_TRUE(region.PointInSet(Vector1d{qmin + kTol}));
    EXPECT_TRUE(region.PointInSet(Vector1d{qmax - kTol}));
    EXPECT_FALSE(region.PointInSet(Vector1d{qmin - kTol}));
    EXPECT_FALSE(region.PointInSet(Vector1d{qmax + kTol}));
  }
};

// Reproduced from the IrisInConfigurationSpace unit tests.
// A simple double pendulum with link lengths `l1` and `l2` with a sphere at the
// tip of radius `r` between two (fixed) walls at `w` from the origin.  The
// true configuration space is - w + r ≤ l₁s₁ + l₂s₁₊₂ ≤ w - r.  These regions
// are visualized at https://www.desmos.com/calculator/ff0hbnkqhm.
class DoublePendulum : public ::testing::Test {
 public:
  const double l1 = 2.0;
  const double l2 = 1.0;
  const double r = .5;
  const double w = 1.83;
  const std::string urdf = fmt::format(
      R"(
<robot name="double_pendulum">
  <link name="fixed">
    <collision name="right">
      <origin rpy="0 0 0" xyz="{w_plus_one_half} 0 0"/>
      <geometry><box size="1 1 10"/></geometry>
    </collision>
    <collision name="left">
      <origin rpy="0 0 0" xyz="-{w_plus_one_half} 0 0"/>
      <geometry><box size="1 1 10"/></geometry>
    </collision>
  </link>
  <joint name="fixed_link_weld" type="fixed">
    <parent link="world"/>
    <child link="fixed"/>
  </joint>
  <link name="link1"/>
  <joint name="joint1" type="revolute">
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57"/>
    <parent link="world"/>
    <child link="link1"/>
  </joint>
  <link name="link2">
    <collision name="ball">
      <origin rpy="0 0 0" xyz="0 0 -{l2}"/>
      <geometry><sphere radius="{r}"/></geometry>
    </collision>
  </link>
  <joint name="joint2" type="revolute">
    <origin rpy="0 0 0" xyz="0 0 -{l1}"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57"/>
    <parent link="link1"/>
    <child link="link2"/>
  </joint>
</robot>
)",
      fmt::arg("w_plus_one_half", w + .5), fmt::arg("l1", l1),
      fmt::arg("l2", l2), fmt::arg("r", r));

  Hyperellipsoid starting_ellipsoid;
  Vector2d sample;
  std::shared_ptr<Meshcat> meshcat;

 protected:
  void SetUp() override {
    sample = Vector2d::Zero();
    starting_ellipsoid = Hyperellipsoid::MakeHypersphere(1e-2, sample);
    meshcat = geometry::GetTestEnvironmentMeshcat();
  }
  void run_checks_on_region(const HPolyhedron& region) {
    EXPECT_EQ(region.ambient_dimension(), 2);
    // Confirm that we've found a substantial region.
    EXPECT_GE(region.MaximumVolumeInscribedEllipsoid().Volume(), 2.0);

    EXPECT_TRUE(region.PointInSet(Vector2d{.4, 0.0}));
    EXPECT_FALSE(region.PointInSet(Vector2d{.5, 0.0}));
    EXPECT_TRUE(region.PointInSet(Vector2d{.3, .3}));
    EXPECT_FALSE(region.PointInSet(Vector2d{.4, .3}));
    EXPECT_TRUE(region.PointInSet(Vector2d{-.4, 0.0}));
    EXPECT_FALSE(region.PointInSet(Vector2d{-.5, 0.0}));
    EXPECT_TRUE(region.PointInSet(Vector2d{-.3, -.3}));
    EXPECT_FALSE(region.PointInSet(Vector2d{-.4, -.3}));
  }
  void plot_env_and_region(const HPolyhedron& region) {
    meshcat->Set2dRenderMode(math::RigidTransformd(Eigen::Vector3d{0, 0, 1}),
                             -3.25, 3.25, -3.25, 3.25);
    meshcat->SetProperty("/Grid", "visible", true);
    Eigen::RowVectorXd theta2s =
        Eigen::RowVectorXd::LinSpaced(100, -1.57, 1.57);
    Eigen::Matrix3Xd points = Eigen::Matrix3Xd::Zero(3, 2 * theta2s.size() + 1);
    const double c = -w + r;
    for (int i = 0; i < theta2s.size(); ++i) {
      const double a = l1 + l2 * std::cos(theta2s[i]),
                   b = l2 * std::sin(theta2s[i]);
      // wolfram solve a*sin(q) + b*cos(q) = c for q
      points(0, i) =
          2 * std::atan((std::sqrt(a * a + b * b - c * c) + a) / (b + c)) +
          M_PI;
      points(1, i) = theta2s[i];
      points(0, points.cols() - i - 2) =
          2 * std::atan((std::sqrt(a * a + b * b - c * c) + a) / (b - c)) -
          M_PI;
      points(1, points.cols() - i - 2) = theta2s[i];
    }
    points.col(points.cols() - 1) = points.col(0);
    meshcat->SetLine("True C_free", points, 2.0, Rgba(0, 0, 1));
    VPolytope vregion = VPolytope(region).GetMinimalRepresentation();
    points.resize(3, vregion.vertices().cols() + 1);
    points.topLeftCorner(2, vregion.vertices().cols()) = vregion.vertices();
    points.topRightCorner(2, 1) = vregion.vertices().col(0);
    points.bottomRows<1>().setZero();
    meshcat->SetLine("IRIS Region", points, 2.0, Rgba(0, 1, 0));

    MaybePauseForUser();
  }
};
}  // namespace planning
}  // namespace drake
