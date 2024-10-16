#pragma once

#include <memory>
#include <optional>
#include <vector>

#include "drake/common/parallelism.h"
#include "drake/geometry/meshcat.h"
#include "drake/geometry/optimization/hpolyhedron.h"
#include "drake/planning/collision_checker.h"
#include "drake/planning/graph_algorithms/max_clique_solver_base.h"
#include "drake/planning/graph_algorithms/min_clique_cover_solver_base.h"
#include "drake/planning/iris/adjacency_matrix_builder_base.h"
#include "drake/planning/iris/iris_from_clique_cover_options.h"
#include "drake/planning/iris/point_sampler_base.h"
#include "drake/planning/iris/region_from_clique_base.h"

namespace drake {
namespace planning {
namespace iris {

/**
 * Given a set of points, computes the visibility graph, computes a clique
 * cover, and then builds convex sets.
 * @param points
 * @param partition whether a clique cover or partition is computed. See the
 * implementation of MaxCliqueCoverSolverBase for details.
 * @param checker Used to build the visibility graph.
 * @param min_clique_cover_solver[in] Computes the clique cover. Cannot be
 * nullptr.
 * @param set_builder[in] Converts cliques to a convex set. Cannot be nullptr
 * @param parallelism Parallelism used to build visibility graph.
 */
std::vector<geometry::optimization::HPolyhedron> PointsToCliqueCoverSets(
    const Eigen::Ref<const Eigen::MatrixXd>& points,
    const CollisionChecker& checker,
    graph_algorithms::MinCliqueCoverSolverBase* min_clique_cover_solver,
    RegionFromCliqueBase* set_builder,
    Parallelism parallelism = Parallelism::Max(), bool partition = true,
    std::shared_ptr<geometry::Meshcat> meshcat = nullptr);
/**
 * @param points
 * @param partition whether a clique cover or partition is computed. See the
 * implementation of MaxCliqueCoverSolverBase for details.
 * @param graph_builder[in] Builds the visibilty graph. Cannot be nullptr
 * @param min_clique_cover_solver[in] Computes the clique cover. Cannot be
 * nullptr.
 * @param set_builder[in] Converts cliques to a convex set. Cannot be nullptr
 * @return The convex sets built from the cliques
 */
std::vector<geometry::optimization::HPolyhedron> PointsToCliqueCoverSets(
    const Eigen::Ref<const Eigen::MatrixXd>& points,
    AdjacencyMatrixBuilderBase* adjacency_matrix_builder,
    graph_algorithms::MinCliqueCoverSolverBase* min_clique_cover_solver,
    RegionFromCliqueBase* set_builder, bool partition = true,
    std::shared_ptr<geometry::Meshcat> meshcat = nullptr);

/**
 *
 * @param options
 * @param checker Checks whether points are in collision.
 * @param point_sampler [in] draws points from the base configuration space.
 * This sampler should draw points from both the collision-free and in-collision
 * cspace. Cannot be nullptr.
 * @param min_clique_cover_solver [in] Solves the min clique cover problem.
 * Cannot be nullptr
 * @param set_builder [in] Converts a set of points to a region. Cannot be
 * nullptr
 * @param sets [in/out] The convex cover of the configuration space. New sets
 * are appended to the end of this. Cannot be nullptr.
 * @param adjacency_matrix_builder [in] An optional argument to enable better
 * visibility graph construction. If this is null, then VisibilityGraph will be
 * used.
 */
void IrisInConfigurationSpaceFromCliqueCoverTemplate(
    const IrisFromCliqueCoverOptions& options, const CollisionChecker& checker,
    RandomGenerator* generator, PointSamplerBase* point_sampler,
    graph_algorithms::MinCliqueCoverSolverBase* min_clique_cover_solver,
    RegionFromCliqueBase* set_builder,
    std::vector<geometry::optimization::HPolyhedron>* sets,
    AdjacencyMatrixBuilderBase* adjacency_matrix_builder = nullptr);

}  // namespace iris
}  // namespace planning
}  // namespace drake
