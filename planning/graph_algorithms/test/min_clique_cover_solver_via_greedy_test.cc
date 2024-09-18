#include "drake/planning/graph_algorithms/min_clique_cover_solver_via_greedy.h"

#include <exception>
#include <iostream>

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/common/test_utilities/expect_throws_message.h"
#include "drake/planning/graph_algorithms/max_clique_solver_via_greedy.h"
#include "drake/planning/graph_algorithms/max_clique_solver_via_mip.h"
#include "drake/planning/graph_algorithms/test/common_graphs.h"

namespace drake {
namespace planning {
namespace graph_algorithms {
namespace {

// Test min clique cover. Compare against the expected size of
// the solution and ensure that the result is one of the true maximum cliques in
// the graph.
using clique_solution_type = std::vector<std::set<int>>;
// Make the clique cover a set of sets to simplify comparison.
using comparable_clique_solution_type = std::set<std::set<int>>;

// Print the set on one line
void PrintSet(const std::set<int>& mySet) {
  std::cout << "{ ";
  for (const auto& elem : mySet) {
    std::cout << elem << " ";
  }
  std::cout << "}" << std::endl;
}
void PrintSetSet(const std::set<std::set<int>>& mySet) {
  std::cout << "{ ";
  for (const auto& elem : mySet) {
    PrintSet(elem);
    std::cout << std::endl;
  }
  std::cout << "}" << std::endl;
}

void TestMinCliqueCover(
    const Eigen::Ref<const Eigen::SparseMatrix<bool>>& adjacency_matrix,
    const std::vector<comparable_clique_solution_type>& possible_solutions,
    MinCliqueCoverSolverViaGreedy* solver) {
  clique_solution_type min_clique_cover_vect =
      solver->SolveMinCliqueCover(adjacency_matrix);
  // Convert the vector of sets into a set of sets so that comparison is easier.
  comparable_clique_solution_type min_clique_cover(
      min_clique_cover_vect.begin(), min_clique_cover_vect.end());

  bool solution_match_found = false;
  for (const auto& possible_solution : possible_solutions) {
    PrintSetSet(min_clique_cover);
    PrintSetSet(possible_solution);
    std::cout << std::endl;
    if (min_clique_cover == possible_solution) {
      solution_match_found = true;
      break;
    }
  }
  EXPECT_TRUE(solution_match_found);
}

GTEST_TEST(MinCliqueCoverSolverViaGreedyTest,
           TestConstructorSettersAndGetters) {
  // Test the default constructor.
  MaxCliqueSolverViaGreedy max_clique_solver{};
  MinCliqueCoverSolverViaGreedy solver{max_clique_solver, 3};

  EXPECT_EQ(solver.get_min_clique_size(), 3);

  solver.set_min_clique_size(5);
  EXPECT_EQ(solver.get_min_clique_size(), 5);
}

GTEST_TEST(MinCliqueCoverSolverViaGreedyTestTest, BullGraph) {
  Eigen::SparseMatrix<bool> graph = internal::BullGraph();
  comparable_clique_solution_type solution;
  solution.insert(std::initializer_list<int>{1, 2, 3});
  solution.insert(std::initializer_list<int>{1, 0});
  solution.insert(std::initializer_list<int>{3, 4});

  std::vector<comparable_clique_solution_type> possible_solutions;
  possible_solutions.push_back(solution);

  MinCliqueCoverSolverViaGreedy solver{MaxCliqueSolverViaGreedy(), 1};

  TestMinCliqueCover(graph, possible_solutions, &solver);
}

GTEST_TEST(MinCliqueCoverSolverViaGreedyTestTest, BullGraphEarlyTermination) {
  // The FullyConnectedPlusFullBipartiteGraph graph has a clique number of size
  // 3.
  Eigen::SparseMatrix<bool> graph = internal::BullGraph();
  comparable_clique_solution_type solution;
  solution.insert(std::initializer_list<int>{1, 2, 3});

  std::vector<comparable_clique_solution_type> possible_solutions;
  possible_solutions.push_back(solution);

  MinCliqueCoverSolverViaGreedy solver{MaxCliqueSolverViaGreedy(), 3};
  TestMinCliqueCover(graph, possible_solutions, &solver);
}

// TODO(Alexandre.Amice) make more complete set of tests

// GTEST_TEST(MinCliqueCoverSolverViaGreedyTestTest,
//            FullyConnectedPlusFullBipartiteGraphMipMaxClique) {
//   // The FullyConnectedPlusFullBipartiteGraph graph has a clique number of
//   size
//   // 3.
//   Eigen::SparseMatrix<bool> graph =
//       internal::FullyConnectedPlusFullBipartiteGraph();
//   VectorX<bool> solution(9);
//
//   // The max cliuqe solutions are pairs of vertices on the bipartite graph.
//   solution << true, true, true, false, false, false, false, false, false;
//
//   std::vector<VectorX<bool>> possible_solutions;
//   possible_solutions.push_back(solution);
//
//   TestMaxCliqueViaMip(graph, 3, possible_solutions);
// }

}  // namespace
}  // namespace graph_algorithms
}  // namespace planning
}  // namespace drake
