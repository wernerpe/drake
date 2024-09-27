#include "drake/planning/iris/hpolyhedron_point_sampler.h"

namespace drake {
namespace planning {
namespace iris {
HPolyhedronPointSampler::HPolyhedronPointSampler(
    const geometry::optimization::HPolyhedron& domain, int mixing_steps,
    const std::optional<Eigen::VectorXd> first_point)
    : domain_(domain),
      mixing_steps_(mixing_steps),
      last_point_(first_point.value_or(domain.ChebyshevCenter())) {
  DRAKE_THROW_UNLESS(domain.ambient_dimension() == last_point_.rows());
}

Eigen::MatrixXd HPolyhedronPointSampler::DoSamplePoints(
    int num_points, RandomGenerator* generator, Parallelism parallelism) {
  unused(parallelism);
  Eigen::MatrixXd points(domain_.ambient_dimension(), num_points);
  for (int i = 0; i < num_points; ++i) {
    last_point_ = domain_.UniformSample(generator, last_point_, mixing_steps_);
    points.col(i) = last_point_;
  }
  return points;
}
}  // namespace iris
}  // namespace planning
}  // namespace drake
