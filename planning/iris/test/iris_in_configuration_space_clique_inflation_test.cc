#include "drake/planning/iris/iris_in_configuration_space_clique_inflation.h"

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/common/test_utilities/maybe_pause_for_user.h"
#include "drake/geometry/optimization/iris.h"
#include "drake/planning/iris/test/box_in_corner_test_fixture.h"

namespace drake {
namespace planning {
namespace iris {
namespace {
using common::MaybePauseForUser;
using geometry::optimization::HPolyhedron;
using geometry::optimization::VPolytope;

/**
 * Sample num_samples_in_test//2 points in region1 and num_samples_in_test//2 in
 * region2 and count the number in both regions. Returns true if number in both
 * regions / num_samples is more than threshold.
 * @param region1
 * @param region2
 * @param num_samples_in_test
 * @param threshold
 * @return
 */
bool RegionsAreApproximatelyTheSame(const HPolyhedron& region1,
                                    const HPolyhedron& region2,
                                    int num_samples_in_test, double threshold,
                                    RandomGenerator* generator,
                                    int mixing_steps = 20) {
  DRAKE_THROW_UNLESS(region1.ambient_dimension() ==
                     region2.ambient_dimension());
  int num_samples = num_samples_in_test / 2;
  Eigen::VectorXd last_point = region1.ChebyshevCenter();
  int num_in_both = 0;
  for (int i = 0; i < num_samples_in_test / 2; i++) {
    last_point = region1.UniformSample(generator, last_point, mixing_steps);
    if (region2.PointInSet(last_point)) {
      ++num_in_both;
    }
  }
  last_point = region2.ChebyshevCenter();
  for (int i = num_samples_in_test / 2; i < num_samples_in_test; i++) {
    last_point = region2.UniformSample(generator, last_point, mixing_steps);
    if (region1.PointInSet(last_point)) {
      ++num_in_both;
    }
  }
  return num_in_both / static_cast<double>(num_samples) > threshold;
}

TEST_F(BoxInCornerTestFixture, IrisInConfigurationSpaceCliqueInflationTest) {
  Eigen::MatrixXd points{2, 4};
  double x = 1;
  double y = 0.1;
  // clang-format off
  points << x,  x, -x, -x,
            y, -y,  y, -y;
  // clang-format on
  Draw2dPointsToMeshcat(points, meshcat, "region_1_points_");
  IrisInConfigurationSpaceCliqueInflation set_builder{
      *checker, options.iris_options, 1e-6};

  HPolyhedron region0 = set_builder.BuildRegion(points);
  Draw2dVPolytope(VPolytope(region0).GetMinimalRepresentation(), "region0",
                  Eigen::Vector3d{0, 1, 0}, meshcat);
  EXPECT_TRUE(RegionsAreApproximatelyTheSame(
      region0, manual_decomposition[5].MakeHPolyhedron(), 100, 0.9,
      &generator));

  // clang-format off
  points << y, -y,  y, -y,
            x,  x, -x, -x;
  // clang-format on
  Draw2dPointsToMeshcat(points, meshcat, "region_2_points_");
  HPolyhedron region1 = set_builder.BuildRegion(points);
  Draw2dVPolytope(VPolytope(region1).GetMinimalRepresentation(), "region1",
                  Eigen::Vector3d{0, 1, 0}, meshcat);
  EXPECT_TRUE(RegionsAreApproximatelyTheSame(
      region1, manual_decomposition[4].MakeHPolyhedron(), 100, 0.9,
      &generator));

  MaybePauseForUser();
}

TEST_F(BoxInCornerTestFixture,
       IrisInConfigurationSpaceCliqueInflationSettersAndGetters) {
  geometry::optimization::IrisOptions iris_options{};
  iris_options.iteration_limit = 10000;
  IrisInConfigurationSpaceCliqueInflation set_builder{*checker, iris_options,
                                                      1e-6};
  EXPECT_EQ(set_builder.iris_options().iteration_limit, 10000);

  iris_options.iteration_limit = 1;
  set_builder.set_iris_options(iris_options);
  EXPECT_EQ(set_builder.iris_options().iteration_limit, 1);

  EXPECT_EQ(
      set_builder.rank_tol_for_minimum_volume_circumscribed_ellipsoid().value(),
      1e-6);
  set_builder.set_rank_tol_for_minimum_volume_circumscribed_ellipsoid(
      std::nullopt);
  EXPECT_FALSE(set_builder.rank_tol_for_minimum_volume_circumscribed_ellipsoid()
                   .has_value());
}

}  // namespace
}  // namespace iris
}  // namespace planning
}  // namespace drake
