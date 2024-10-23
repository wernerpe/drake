#include "drake/planning/iris/barycentric_vpolytope_sampler.h"

#include <optional>
#include <random>
#include <set>

#include <common_robotics_utilities/parallelism.hpp>

#include "drake/common/random.h"
#include "drake/common/ssize.h"
#include "drake/geometry/optimization/vpolytope.h"
#include "drake/planning/iris/point_sampler_base.h"

namespace drake {
namespace planning {
namespace iris {
using common_robotics_utilities::parallelism::DegreeOfParallelism;
using common_robotics_utilities::parallelism::ParallelForBackend;
using common_robotics_utilities::parallelism::StaticParallelForIndexLoop;

namespace {
// Sample the unit simplex using the algorithm presented in "Sampling Uniformly
// from the Unit Simplex" by Smith and Tromble.
// https://www.cs.cmu.edu/~nasmith/papers/smith+tromble.tr04.pdf
Eigen::VectorXd SampleUnitSimplex(int dim, RandomGenerator* generator) {
  std::uniform_int_distribution<int> distribution(1, RAND_MAX - 1);
  const int M = RAND_MAX - dim;
  std::set<int> x{0, RAND_MAX};

  while (ssize(x) < dim + 1) {
    x.insert(distribution(*generator));
  }
  Eigen::VectorXd y{dim};
  int ctr = 0;
  for (auto x_it = x.begin(); x_it != std::prev(x.end()); ++x_it) {
    y(ctr++) = static_cast<double>(*(std::next(x_it)) - *x_it) / M;
  }
  return y;
}
}  // namespace

BarycentricVPolytopeSampler::BarycentricVPolytopeSampler(
    const geometry::optimization::VPolytope& polytope,
    bool sampling_always_returns_vertices)
    : PointSamplerBase(),
      domain_(polytope),
      sampling_always_returns_vertices_(sampling_always_returns_vertices) {}

Eigen::MatrixXd BarycentricVPolytopeSampler::DoSamplePoints(
    int num_points, RandomGenerator* generator, Parallelism parallelism) {
  unused(parallelism);
  Eigen::MatrixXd alphas{domain_.vertices().cols(), num_points};
  int starting_i = 0;
  if (sampling_always_returns_vertices_ &&
      num_points >= domain_.vertices().cols()) {
    alphas.leftCols(domain_.vertices().cols()) = Eigen::MatrixXd::Identity(
        domain_.vertices().cols(), domain_.vertices().cols());
    starting_i = domain_.vertices().cols();
  }

  for (int i = starting_i; i < num_points; i++) {
    alphas.col(i) = SampleUnitSimplex(domain_.vertices().cols(), generator);
  }
  return domain_.vertices() * alphas;
}

}  // namespace iris
}  // namespace planning
}  // namespace drake
