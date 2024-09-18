#include "drake/planning/iris/point_sampler_base.h"
namespace drake {
namespace planning {
namespace iris {

Eigen::MatrixXd PointSamplerBase::SamplePoints(int num_points,
                                               RandomGenerator* generator,
                                               Parallelism parallelism) {
  return DoSamplePoints(num_points, generator, parallelism);
}

}  // namespace iris
}  // namespace planning
}  // namespace drake
