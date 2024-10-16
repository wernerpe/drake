#include "drake/planning/iris/test/clique_cover_test_utils.h"

#include "drake/common/drake_throw.h"
#include "drake/geometry/optimization/hpolyhedron.h"
#include "drake/geometry/optimization/vpolytope.h"

namespace drake {
namespace planning {
namespace iris {
namespace internal {

using math::RigidTransform;
bool RegionsAreApproximatelyTheSame(
    const geometry::optimization::HPolyhedron& region1,
    const geometry::optimization::HPolyhedron& region2, int num_samples_in_test,
    double threshold, RandomGenerator* generator, int mixing_steps) {
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

// Draw a two dimensional polytope in meshcat.
void Draw2dVPolytope(const geometry::optimization::VPolytope& polytope,
                     const std::string& meshcat_name,
                     const Eigen::Ref<const Eigen::Vector3d>& color,
                     std::shared_ptr<geometry::Meshcat> meshcat) {
  DRAKE_THROW_UNLESS(polytope.ambient_dimension() == 2);
  Eigen::Matrix3Xd points =
      Eigen::Matrix3Xd::Zero(3, polytope.vertices().cols() + 1);
  points.topLeftCorner(2, polytope.vertices().cols()) = polytope.vertices();
  points.topRightCorner(2, 1) = polytope.vertices().col(0);
  points.bottomRows<1>().setZero();

  meshcat->SetLine(meshcat_name, points, 2.0,
                   geometry::Rgba(color(0), color(1), color(2)));
}

void Draw2dPointsToMeshcat(const Eigen::Ref<const Eigen::Matrix2Xd>& points,
                           std::shared_ptr<geometry::Meshcat> meshcat,
                           std::string meshcat_name, double sphere_size,
                           geometry::Rgba rgba) {
  for (int i = 0; i < points.cols(); ++i) {
    Eigen::Vector3d cur_point;
    cur_point << points(0, i), points(1, i), 0;
    meshcat->SetObject(meshcat_name + std::to_string(i),
                       geometry::Sphere(sphere_size), rgba);
    meshcat->SetTransform(meshcat_name + std::to_string(i),
                          RigidTransform<double>(cur_point));
  }
}

}  // namespace internal
}  // namespace iris
}  // namespace planning
}  // namespace drake
