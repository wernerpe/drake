#pragma once
#include <string>

#include <Eigen/Dense>
#include <gtest/gtest.h>

#include "drake/geometry/optimization/hpolyhedron.h"
#include "drake/geometry/optimization/hyperellipsoid.h"

namespace drake {
namespace planning {

using Eigen::Vector2d;
using Eigen::VectorXd;
using geometry::Meshcat;
using geometry::optimization::HPolyhedron;
using geometry::optimization::Hyperellipsoid;
using geometry::optimization::VPolytope;
using symbolic::Variable;

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

}  // namespace planning
}  // namespace drake
