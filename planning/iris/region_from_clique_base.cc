#include "drake/planning/iris/region_from_clique_base.h"

namespace drake {
namespace planning {
namespace iris {

geometry::optimization::HPolyhedron RegionFromCliqueBase::BuildRegion(
    const Eigen::Ref<const Eigen::MatrixXd>& clique_points) {
  return DoBuildRegion(clique_points);
}

}  // namespace iris
}  // namespace planning
}  // namespace drake
