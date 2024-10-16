#include "drake/planning/iris/iris_in_configuration_space_clique_inflation.h"

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/common/test_utilities/maybe_pause_for_user.h"
#include "drake/geometry/optimization/iris.h"
#include "drake/planning/iris/test/box_in_corner_test_fixture.h"
#include "drake/planning/iris/test/clique_cover_test_utils.h"

namespace drake {
namespace planning {
namespace iris {
namespace {
using common::MaybePauseForUser;
using geometry::optimization::HPolyhedron;
using geometry::optimization::VPolytope;

TEST_F(BoxInCornerTestFixture, IrisInConfigurationSpaceCliqueInflationTest) {
  Eigen::MatrixXd points{2, 4};
  double x = 1;
  double y = 0.1;
  // clang-format off
  points << x,  x, -x, -x,
            y, -y,  y, -y;
  // clang-format on
  internal::Draw2dPointsToMeshcat(points, meshcat, "region_1_points_");
  IrisInConfigurationSpaceCliqueInflation set_builder{
      *checker, options.iris_options, 1e-6};

  HPolyhedron region0 = set_builder.BuildRegion(points);
  internal::Draw2dVPolytope(VPolytope(region0).GetMinimalRepresentation(),
                            "region0", Eigen::Vector3d{0, 1, 0}, meshcat);
  EXPECT_TRUE(internal::RegionsAreApproximatelyTheSame(
      region0, manual_decomposition[5].MakeHPolyhedron(), 100, 0.9,
      &generator));

  // clang-format off
  points << y, -y,  y, -y,
            x,  x, -x, -x;
  // clang-format on
  internal::Draw2dPointsToMeshcat(points, meshcat, "region_2_points_");
  HPolyhedron region1 = set_builder.BuildRegion(points);
  internal::Draw2dVPolytope(VPolytope(region1).GetMinimalRepresentation(),
                            "region1", Eigen::Vector3d{0, 1, 0}, meshcat);
  EXPECT_TRUE(internal::RegionsAreApproximatelyTheSame(
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
