#pragma once

#include <Eigen/Dense>
#include <drake/common/random.h>
#include <drake/geometry/optimization/hpolyhedron.h>

namespace drake {
namespace planning {
namespace iris {
namespace internal {

/**
 * Sample num_samples_in_test//2 points in region1 and num_samples_in_test//2 in
 * region2 and count the number in both regions. Returns true if number in both
 * regions / num_samples is more than threshold.
 * @param region1
 * @param region2
 * @param num_samples_in_test
 * @param threshold
 * @return
 */
bool RegionsAreApproximatelyTheSame(
    const geometry::optimization::HPolyhedron& region1,
    const geometry::optimization::HPolyhedron& region2, int num_samples_in_test,
    double threshold, RandomGenerator* generator, int mixing_steps = 20) {
  DRAKE_THROW_UNLESS(region1.ambient_dimension() ==
                     region2.ambient_dimension());
  int num_samples = num_samples_in_test / 2;
  Eigen::VectorXd last_point = region1.ChebyshevCenter();
  int num_in_both = 0;
  for (int i = 0; i < num_samples_in_test / 2; i++) {
    last_point = region1.UniformSample(generator, last_point, mixing_steps);
    if (region2.PointInSet(last_point)) {
      ++num_in_both;
    }
  }
  last_point = region2.ChebyshevCenter();
  for (int i = num_samples_in_test / 2; i < num_samples_in_test; i++) {
    last_point = region2.UniformSample(generator, last_point, mixing_steps);
    if (region1.PointInSet(last_point)) {
      ++num_in_both;
    }
  }
  return num_in_both / static_cast<double>(num_samples) > threshold;
}

}  // namespace internal
}  // namespace iris
}  // namespace planning
}  // namespace drake
