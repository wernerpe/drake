#pragma once

#include "drake/common/random.h"
#include "drake/geometry/optimization/vpolytope.h"
#include "drake/planning/iris/point_sampler_base.h"

namespace drake {
namespace planning {
namespace iris {

/**
 * Given a VPolytope with vertices as the columns of a matrix V, returns a
 * random sample inside the polytope by sampling a vector a such that ∑ᵢaᵢ = 1
 * and aᵢ ≥ 0 and return V * a. This sample is uniformly distributed in the
 * generalized barycentric coordinates of the VPolytope.
 *
 * If sampling_always_returns_vertices is true and the requested number of
 * points is larger than the number of vertices, the first |vertices| samples
 * will be the vertices of the domain.
 */
class BarycentricVPolytopeSampler final : public PointSamplerBase {
 public:
  explicit BarycentricVPolytopeSampler(
      const geometry::optimization::VPolytope& domain,
      bool sampling_always_returns_vertices = false);

  const geometry::optimization::VPolytope& domain() const { return domain_; }

  void set_domain(const geometry::optimization::VPolytope& domain) {
    domain_ = domain;
  }

  bool sampling_always_returns_vertices() const {
    return sampling_always_returns_vertices_;
  }

  void set_sampling_always_returns_vertices(
      bool sampling_always_returns_vertices) {
    sampling_always_returns_vertices_ = sampling_always_returns_vertices;
  }

 private:
  geometry::optimization::VPolytope domain_;
  bool sampling_always_returns_vertices_;

  Eigen::MatrixXd DoSamplePoints(int num_points, RandomGenerator* generator,
                                 Parallelism parallelism) override;
};

}  // namespace iris
}  // namespace planning
}  // namespace drake
