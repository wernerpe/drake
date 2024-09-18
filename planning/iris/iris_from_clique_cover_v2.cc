#include "drake/planning/iris/iris_from_clique_cover_v2.h"

#include <algorithm>
#include <memory>
#include <optional>
#include <vector>

#include "drake/planning/graph_algorithms/max_clique_solver_base.h"
#include "drake/planning/graph_algorithms/max_clique_solver_via_greedy.h"
#include "drake/planning/graph_algorithms/max_clique_solver_via_mip.h"
#include "drake/planning/graph_algorithms/min_clique_cover_solver_via_greedy.h"
#include "drake/planning/iris/hpolyhedron_point_sampler.h"
#include "drake/planning/iris/iris_in_configuration_space_clique_inflation.h"

namespace drake {
namespace planning {
namespace iris {
using geometry::optimization::HPolyhedron;
using geometry::optimization::Hyperellipsoid;
namespace {
std::unique_ptr<planning::graph_algorithms::MaxCliqueSolverBase>
MakeDefaultMaxCliqueSolver() {
  return std::unique_ptr<planning::graph_algorithms::MaxCliqueSolverBase>(
      new planning::graph_algorithms::MaxCliqueSolverViaGreedy());
}

}  // namespace

void IrisInConfigurationSpaceFromCliqueCoverV2(
    const CollisionChecker& checker, const IrisFromCliqueCoverOptions& options,
    RandomGenerator* generator, std::vector<HPolyhedron>* sets,
    const planning::graph_algorithms::MaxCliqueSolverBase*
        max_clique_solver_ptr) {
  DRAKE_THROW_UNLESS(options.coverage_termination_threshold > 0);
  DRAKE_THROW_UNLESS(options.iteration_limit > 0);

  // Note: Even though the iris_options.bounding_region may be provided,
  // IrisInConfigurationSpace (currently) requires finite joint limits.
  DRAKE_THROW_UNLESS(
      checker.plant().GetPositionLowerLimits().array().isFinite().all());
  DRAKE_THROW_UNLESS(
      checker.plant().GetPositionUpperLimits().array().isFinite().all());

  const HPolyhedron domain = options.iris_options.bounding_region.value_or(
      HPolyhedron::MakeBox(checker.plant().GetPositionLowerLimits(),
                           checker.plant().GetPositionUpperLimits()));
  DRAKE_THROW_UNLESS(domain.ambient_dimension() ==
                     checker.plant().num_positions());

  std::unique_ptr<planning::graph_algorithms::MaxCliqueSolverBase>
      default_max_clique_solver;
  // Only construct the default solver if max_clique_solver is null.
  if (max_clique_solver_ptr == nullptr) {
    default_max_clique_solver = MakeDefaultMaxCliqueSolver();
    log()->info("Using default max clique solver MaxCliqueSolverViaGreedy.");
  }

  const graph_algorithms::MaxCliqueSolverBase* max_clique_solver =
      max_clique_solver_ptr == nullptr ? default_max_clique_solver.get()
                                       : max_clique_solver_ptr;

  // Override options which are set too aggressively.
  const int minimum_clique_size = std::max(options.minimum_clique_size,
                                           checker.plant().num_positions() + 1);
  graph_algorithms::MinCliqueCoverSolverViaGreedy min_clique_cover_solver(
      *max_clique_solver, minimum_clique_size);

  std::unique_ptr<PointSamplerBase> point_sampler =
      std::make_unique<HPolyhedronPointSampler>(domain);

  std::unique_ptr<RegionFromCliqueBase> set_builder =
      std::make_unique<IrisInConfigurationSpaceCliqueInflation>(
          checker, options.iris_options,
          options.rank_tol_for_minimum_volume_circumscribed_ellipsoid);

  IrisInConfigurationSpaceFromCliqueCoverTemplate(
      options, checker, generator, point_sampler.get(),
      &min_clique_cover_solver, set_builder.get(), sets);
}

}  // namespace iris
}  // namespace planning
}  // namespace drake
