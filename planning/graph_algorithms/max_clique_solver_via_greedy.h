#pragma once

#include <Eigen/Sparse>

#include "drake/planning/graph_algorithms/max_clique_solver_base.h"

namespace drake {
namespace planning {
namespace graph_algorithms {

/**
 * Approximately solves the maximum clique problem via a greedy
 * heuristic. Vertices are greedily added to the clique based on their degree of
 * connectivity to the remaining candidate vertices. Candidate vertices are
 * non-clique members that share an edge with every clique member. The algorithm
 * initializes the clique with an empty set and makes every vertex a candidate,
 * then the degree of connectivity of every vertex is computed and candidate
 * vertex with the highest degree is added to the clique. Afterwards, the
 * adjacency matrix and new candidate list are updated and the previous two
 * steps are repeated until no candidates are left.
 */
class MaxCliqueSolverViaGreedy final : public MaxCliqueSolverBase {
 public:
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(MaxCliqueSolverViaGreedy);
  MaxCliqueSolverViaGreedy() = default;

 private:
  VectorX<bool> DoSolveMaxClique(
      const Eigen::SparseMatrix<bool>& adjacency_matrix) const final;
};

}  // namespace graph_algorithms
}  // namespace planning
}  // namespace drake
