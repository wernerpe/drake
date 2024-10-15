#pragma once
#include <optional>

#include "drake/common/random.h"
#include "drake/geometry/optimization/hpolyhedron.h"
#include "drake/planning/iris/point_sampler_base.h"

namespace drake {
namespace planning {
namespace iris {

/**
 * This class samples points uniformly from the HPolyhedron domain. See
 * UniformSample in HPolyhedron for details.
 */
class HPolyhedronPointSampler final : public PointSamplerBase {
 public:
  explicit HPolyhedronPointSampler(
      const geometry::optimization::HPolyhedron& domain, int mixing_steps = 10,
      const std::optional<Eigen::VectorXd> first_point = std::nullopt);

  int mixing_steps() const { return mixing_steps_; }

  void set_mixing_steps(int mixing_steps) { mixing_steps_ = mixing_steps; }

  /**
   * Returns the last point sampled from this sampler. This is used as the
   * initial point in hit-and-run sampling to draw the next point.
   * @return
   */
  const Eigen::VectorXd& last_point() const { return last_point_; }

 private:
  geometry::optimization::HPolyhedron domain_;
  int mixing_steps_;
  Eigen::VectorXd last_point_;

  Eigen::MatrixXd DoSamplePoints(int num_points, RandomGenerator* generator,
                                 Parallelism parallelism) override;
};

}  // namespace iris
}  // namespace planning
}  // namespace drake
