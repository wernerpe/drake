#include "drake/planning/iris/iris_from_clique_cover_template_internal.h"

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/common/test_utilities/maybe_pause_for_user.h"
#include "drake/geometry/optimization/iris.h"
#include "drake/planning/iris/hpolyhedron_point_sampler.h"
#include "drake/planning/iris/test/box_in_corner_test_fixture.h"

namespace drake {
namespace planning {
namespace iris {
using geometry::optimization::HPolyhedron;
// Makes an HPolyhedronPointSampler from the joint limits of the collision
// checker's plant.
HPolyhedronPointSampler MakeDefaultHPolyhedronPointSampler(
    const CollisionChecker& checker, int mixing_steps = 10,
    const std::optional<Eigen::VectorXd> first_point = std::nullopt) {
  const HPolyhedron domain =
      HPolyhedron::MakeBox(checker.plant().GetPositionLowerLimits(),
                           checker.plant().GetPositionUpperLimits());
  return HPolyhedronPointSampler(domain, mixing_steps, first_point);
}

TEST_F(BoxInCornerTestFixture, SampleCollisionFreePointsNoSets) {
  int num_points = 100;
  HPolyhedronPointSampler sampler =
      MakeDefaultHPolyhedronPointSampler(*checker);
  Eigen::MatrixXd sampled_points;
  internal::SampleCollisionFreePoints(num_points, *checker, &generator,
                                      &sampler, &sampled_points);
  EXPECT_EQ(sampled_points.rows(), checker->plant().num_positions());
  EXPECT_EQ(sampled_points.cols(), num_points);

  for (int i = 0; i < sampled_points.cols(); ++i) {
    EXPECT_TRUE(checker->CheckConfigCollisionFree(sampled_points.col(i)));
  }

  int num_points2 = 10;
  Eigen::MatrixXd sampled_points2 = sampled_points;
  internal::SampleCollisionFreePoints(num_points2, *checker, &generator,
                                      &sampler, &sampled_points);
  EXPECT_EQ(sampled_points.rows(), checker->plant().num_positions());
  EXPECT_EQ(sampled_points.cols(), num_points2);
  for (int i = 0; i < num_points2; ++i) {
    EXPECT_TRUE(checker->CheckConfigCollisionFree(sampled_points.col(i)));
  }

  // Now we ask that the sampled points be outside the first 2 regions of the
  // manual decomposition.
  std::vector<HPolyhedron> sets;
  sets.push_back(manual_decomposition[0].MakeHPolyhedron());
  sets.push_back(manual_decomposition[1].MakeHPolyhedron());
  internal::SampleCollisionFreePoints(num_points, *checker, &generator,
                                      &sampler, &sampled_points, &sets);
  for (int i = 0; i < sampled_points.cols(); ++i) {
    EXPECT_TRUE(checker->CheckConfigCollisionFree(sampled_points.col(i)));
    EXPECT_FALSE(sets[0].PointInSet(sampled_points.col(i)));
    EXPECT_FALSE(sets[1].PointInSet(sampled_points.col(i)));
  }
}

}  // namespace iris
}  // namespace planning
}  // namespace drake
