#include "drake/planning/iris/iris_zo_from_clique_builder.h"

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

TEST_F(BoxInCornerTestFixture, IrisZoFromCliqueBuilderTest) {
  Eigen::MatrixXd points{2, 4};
  double x = 1;
  double y = 0.1;
  // clang-format off
  points << x,  x, -x, -x,
            y, -y,  y, -y;
  // clang-format on
  internal::Draw2dPointsToMeshcat(points, meshcat, "region_1_points_");
  IrisZoOptions options;
  options.meshcat = meshcat;
  IrisZoFromCliqueBuilder set_builder{*checker, std::nullopt, options, 1e-6};

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

TEST_F(BoxInCornerTestFixture, IrisZoFromCliqueBuilderSettersAndGetters) {
  IrisZoOptions options{};
  options.max_iterations = 10000;
  Eigen::MatrixXd A{3, 2};
  // clang-format off
  A << 1, 0,
       0, 1,
       1, 1;
  // clang-format on
  Eigen::VectorXd b = Eigen::VectorXd::Ones(3);
  HPolyhedron domain{A, b};
  IrisZoFromCliqueBuilder set_builder{*checker, domain, options, 1e-6};
  EXPECT_EQ(set_builder.options().max_iterations, 10000);

  options.max_iterations = 1;
  set_builder.set_options(options);
  EXPECT_EQ(set_builder.options().max_iterations, 1);

  EXPECT_EQ(
      set_builder.rank_tol_for_minimum_volume_circumscribed_ellipsoid().value(),
      1e-6);
  set_builder.set_rank_tol_for_minimum_volume_circumscribed_ellipsoid(
      std::nullopt);
  EXPECT_FALSE(set_builder.rank_tol_for_minimum_volume_circumscribed_ellipsoid()
                   .has_value());

  EXPECT_TRUE(CompareMatrices(set_builder.domain().A(), domain.A()));
  EXPECT_TRUE(CompareMatrices(set_builder.domain().b(), domain.b()));

  HPolyhedron domain2{A.topRows(2), b.topRows(2)};
  set_builder.set_domain(domain2);
  EXPECT_TRUE(CompareMatrices(set_builder.domain().A(), domain2.A()));
  EXPECT_TRUE(CompareMatrices(set_builder.domain().b(), domain2.b()));

  Eigen::MatrixXd badA{3, 3};
  Eigen::VectorXd badb{3};

  EXPECT_THROW(set_builder.set_domain(HPolyhedron{badA, badb}), std::exception);
}

}  // namespace
}  // namespace iris
}  // namespace planning
}  // namespace drake
