#pragma once

#include "drake/common/drake_copyable.h"
#include "drake/geometry/optimization/hpolyhedron.h"

namespace drake {
namespace planning {
namespace iris {

/**
 * An abstract base class for implementing ways to build a convex set from a
 * clique of points
 */
class RegionFromCliqueBase {
 public:
  RegionFromCliqueBase() = default;
  /**
   * Given a set of points, build a convex set.
   */
  geometry::optimization::HPolyhedron BuildRegion(
      const Eigen::Ref<const Eigen::MatrixXd>& clique_points);

  virtual ~RegionFromCliqueBase() {}

 protected:
  // We put the copy/move/assignment constructors as protected to avoid copy
  // slicing. The inherited final subclasses should put them in public
  // functions.
  DRAKE_DEFAULT_COPY_AND_MOVE_AND_ASSIGN(RegionFromCliqueBase);

  virtual geometry::optimization::HPolyhedron DoBuildRegion(
      const Eigen::Ref<const Eigen::MatrixXd>& clique_points) = 0;
};

}  // namespace iris
}  // namespace planning
}  // namespace drake
