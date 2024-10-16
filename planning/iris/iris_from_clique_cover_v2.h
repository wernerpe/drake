#pragma once

#include <vector>

#include "drake/common/parallelism.h"
#include "drake/geometry/optimization/hpolyhedron.h"
#include "drake/planning/collision_checker.h"
#include "drake/planning/graph_algorithms/max_clique_solver_base.h"
#include "drake/planning/iris/iris_from_clique_cover_options.h"
#include "drake/planning/iris/iris_from_clique_cover_template.h"

namespace drake {
namespace planning {
namespace iris {

void IrisInConfigurationSpaceFromCliqueCoverV2(
    const CollisionChecker& checker, const IrisFromCliqueCoverOptions& options,
    RandomGenerator* generator,
    std::vector<geometry::optimization::HPolyhedron>* sets,
    const planning::graph_algorithms::MaxCliqueSolverBase*
        max_clique_solver_ptr);

}  // namespace iris
}  // namespace planning
}  // namespace drake
