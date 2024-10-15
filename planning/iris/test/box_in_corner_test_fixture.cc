#include "drake/planning/iris/test/box_in_corner_test_fixture.h"
namespace drake {
namespace planning {
namespace iris {
using math::RigidTransform;

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
}  // namespace iris
}  // namespace planning
}  // namespace drake
