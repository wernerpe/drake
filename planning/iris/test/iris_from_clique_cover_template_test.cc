#include "drake/planning/iris/iris_from_clique_cover_template.h"

#include <gtest/gtest.h>

#include "drake/common/find_resource.h"
#include "drake/common/ssize.h"
#include "drake/common/test_utilities/expect_throws_message.h"
#include "drake/common/test_utilities/maybe_pause_for_user.h"
#include "drake/geometry/optimization/hpolyhedron.h"
#include "drake/geometry/optimization/hyperrectangle.h"
#include "drake/geometry/optimization/vpolytope.h"
#include "drake/geometry/test_utilities/meshcat_environment.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/planning/graph_algorithms/max_clique_solver_via_mip.h"
#include "drake/planning/iris/hpolyhedron_point_sampler.h"
#include "drake/planning/iris/iris_from_clique_cover_v2.h"
#include "drake/planning/iris/test/box_in_corner_test_fixture.h"
#include "drake/planning/iris/test/clique_cover_test_utils.h"
#include "drake/planning/robot_diagram_builder.h"
#include "drake/planning/scene_graph_collision_checker.h"
#include "drake/solvers/gurobi_solver.h"
#include "drake/solvers/mosek_solver.h"
#include "drake/solvers/solver_options.h"
#include "drake/systems/framework/diagram_builder.h"

namespace drake {
namespace planning {
namespace iris {
namespace {
using common::MaybePauseForUser;
using Eigen::Vector2d;
using geometry::Meshcat;
using geometry::Rgba;
using geometry::optimization::ConvexSets;
using geometry::optimization::HPolyhedron;
using geometry::optimization::Hyperrectangle;
using geometry::optimization::IrisOptions;
using geometry::optimization::VPolytope;

// Note that this test actually checks that IrisFromCliqueCoverTemplate is
// implemented correctly and that PointsToCliqueCoverSets is implemented
// correctly as IrisInConfigurationSpaceFromCliqueCoverV2 implements the former
// which calls the latter.
TEST_F(BoxInCornerTestFixture,
       IrisInConfigurationSpaceCliqueInflationTestCenterRegion) {
  iris_from_clique_cover_options.num_points_per_visibility_round = 250;
  // TODO(Alexandre.Amice) use a different set of
  // iris_from_clique_cover_options.
  iris_from_clique_cover_options.partition = true;
  iris_from_clique_cover_options.sample_outside_of_sets = true;
  std::vector<geometry::optimization::HPolyhedron> sets;
  IrisInConfigurationSpaceFromCliqueCoverV2(
      *checker, iris_from_clique_cover_options, &generator, &sets, nullptr);

  EXPECT_EQ(ssize(sets), 6);

  // Show the IrisFromCliqueCoverDecomposition
  for (int i = 0; i < ssize(sets); ++i) {
    // Choose a random color.
    for (int j = 0; j < color.size(); ++j) {
      color[j] = abs(gaussian(generator));
    }
    color.normalize();
    VPolytope vregion = VPolytope(sets.at(i)).GetMinimalRepresentation();
    internal::Draw2dVPolytope(vregion,
                              fmt::format("iris_from_clique_cover_greedy{}", i),
                              color, meshcat);
  }

  // Now check the coverage by drawing points from the manual decomposition and
  // checking if they are inside the IrisFromCliqueCover decomposition.
  int num_samples_per_set = 1000;
  int num_in_automatic_decomposition = 0;
  for (const auto& manual_set : manual_decomposition) {
    for (int i = 0; i < num_samples_per_set; ++i) {
      Eigen::Vector2d sample = manual_set.UniformSample(&generator);
      for (const auto& set : sets) {
        if (set.PointInSet(sample)) {
          ++num_in_automatic_decomposition;
          break;
        }
      }
    }
  }
  double coverage_estimate =
      static_cast<double>(num_in_automatic_decomposition) /
      static_cast<double>(num_samples_per_set * ssize(manual_decomposition));
  // We set the termination threshold to be at 0.9 with 1000 points for a
  // coverage check. This number is low enough that the test passes regardless
  // of the random seed. (The probability of success is larger than 1-1e-9).
  EXPECT_GE(coverage_estimate, 0.8);

  MaybePauseForUser();
}

}  // namespace
}  // namespace iris
}  // namespace planning
}  // namespace drake
