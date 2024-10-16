#pragma once

#include <algorithm>
#include <memory>
#include <optional>
#include <vector>

#include "drake/common/parallelism.h"
#include "drake/geometry/meshcat.h"
#include "drake/geometry/optimization/convex_set.h"
#include "drake/geometry/optimization/hpolyhedron.h"
#include "drake/geometry/optimization/iris.h"
#include "drake/planning/graph_algorithms/max_clique_solver_base.h"
#include "drake/planning/graph_algorithms/max_clique_solver_via_greedy.h"
#include "drake/planning/iris/iris_from_clique_cover_options.h"
#include "drake/planning/scene_graph_collision_checker.h"

namespace drake {
namespace planning {
/**
 * Cover the configuration space in Iris regions using the Visibility Clique
 * Cover Algorithm as described in
 *
 * P. Werner, A. Amice, T. Marcucci, D. Rus, R. Tedrake "Approximating Robot
 * Configuration Spaces with few Convex Sets using Clique Covers of Visibility
 * Graphs" In 2024 IEEE Internation Conference on Robotics and Automation.
 * https://arxiv.org/abs/2310.02875
 *
 * @param checker The collision checker containing the plant and its associated
 * scene_graph.
 * @param generator There are points in the algorithm requiring randomness. The
 * generator controls this source of randomness.
 * @param sets [in/out] initial sets covering the space (potentially empty).
 * The cover is written into this vector.
 * @param max_clique_solver The min clique cover problem is approximatley solved
 * by repeatedly solving max clique on the uncovered graph and adding this
 * largest clique to the cover. The max clique problem is solved by this solver.
 * If parallelism is set to allow more than 1 thread, then the solver **must**
 * be implemented in C++.
 *
 * If nullptr is passed as the `max_clique_solver`, then max clique will be
 * solved using an instance of MaxCliqueSolverViaGreedy, which is a fast
 * heuristic. If higher quality cliques are desired, consider changing the
 * solver to an instance of MaxCliqueSolverViaMip. Currently, the padding in the
 * collision checker is not forwarded to the algorithm, and therefore the final
 * regions do not necessarily respect this padding. Effectively, this means that
 * the regions are generated as if the padding is set to 0. This behavior may be
 * adjusted in the future at the resolution of #18830.
 *
 * Note that MaxCliqueSolverViaMip requires the availability of a
 * Mixed-Integer Linear Programming solver (e.g. Gurobi and/or Mosek). We
 * recommend enabling those solvers if possible because they produce higher
 * quality cliques (https://drake.mit.edu/bazel.html#proprietary_solvers). The
 * method will throw if @p max_clique_solver cannot solve the max clique
 * problem.
 */
void IrisInConfigurationSpaceFromCliqueCover(
    const CollisionChecker& checker,
    const iris::IrisFromCliqueCoverOptions& options, RandomGenerator* generator,
    std::vector<geometry::optimization::HPolyhedron>* sets,
    const planning::graph_algorithms::MaxCliqueSolverBase* max_clique_solver =
        nullptr);

}  // namespace planning
}  // namespace drake
