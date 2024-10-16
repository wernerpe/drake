#pragma once

#include <memory>
#include <string>

#include <Eigen/Dense>
#include <drake/common/random.h>
#include <drake/geometry/optimization/hpolyhedron.h>

#include "drake/geometry/meshcat.h"

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
    double threshold, RandomGenerator* generator, int mixing_steps = 20);

// Draw a two dimensional polytope in meshcat.
void Draw2dVPolytope(const geometry::optimization::VPolytope& polytope,
                     const std::string& meshcat_name,
                     const Eigen::Ref<const Eigen::Vector3d>& color,
                     std::shared_ptr<geometry::Meshcat> meshcat);

// Draws two dimensional points to meshcat as spheres.
void Draw2dPointsToMeshcat(const Eigen::Ref<const Eigen::Matrix2Xd>& points,
                           std::shared_ptr<geometry::Meshcat> meshcat,
                           std::string meshcat_name = "points_",
                           double sphere_size = 0.1,
                           geometry::Rgba rgba = geometry::Rgba(0.1, 0.1, 0.1,
                                                                1.0));

}  // namespace internal
}  // namespace iris
}  // namespace planning
}  // namespace drake
