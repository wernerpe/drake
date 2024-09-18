#pragma once

#include <memory>
#include <optional>

#include "drake/geometry/optimization/hpolyhedron.h"
#include "drake/geometry/optimization/hyperellipsoid.h"
#include "drake/geometry/optimization/iris.h"
#include "drake/planning/collision_checker.h"
#include "drake/planning/iris/region_from_clique_base.h"

namespace drake {
namespace planning {
namespace iris {

/**
 * Given a clique, this class constructs a region by computing the maximum
 * volume inscribed ellipsoid of the clique and then running
 * IrisFromInConfigurationSpace with the initial point and metric set to the
 * cetner of the ellipse and the ellipse respectively and using the plant and
 * scene graph contained in checker.
 *
 * TODO(Alexandre.Amice) currently I clone the collision checker. This is
 * wasteful so consider a better design when going to master.
 */
class IrisInConfigurationSpaceCliqueInflation final
    : public RegionFromCliqueBase {
 public:
  IrisInConfigurationSpaceCliqueInflation(
      const CollisionChecker& checker,
      const geometry::optimization::IrisOptions& iris_options,
      std::optional<double>
          rank_tol_for_minimum_volume_circumscribed_ellipsoid = std::nullopt);

  /**
   * The options used during IrisInConfigurationSpace.
   * @return
   */
  const geometry::optimization::IrisOptions& iris_options() const {
    return iris_options_;
  }

  /**
   * Sets the options used during IrisInConfigurationSpace.
   * @return
   */
  void set_iris_options(
      const geometry::optimization::IrisOptions& iris_options) {
    iris_options_ = iris_options;
  }
  /**
   * The rank tolerance used to decide whether an ellipsoid is rank-deficient.
   * If this is nullopt, then the default rank_tol argument is used when
   * calling MinimumVolumeCircumscribedEllipsoid. See
   * Hyperellipsoid::MinimumVolumeCircumscribedEllipsoid.
   * @return
   */
  std::optional<double> rank_tol_for_minimum_volume_circumscribed_ellipsoid()
      const {
    return rank_tol_for_minimum_volume_circumscribed_ellipsoid_;
  }

  void set_rank_tol_for_minimum_volume_circumscribed_ellipsoid(
      std::optional<double>
          rank_tol_for_minimum_volume_circumscribed_ellipsoid) {
    rank_tol_for_minimum_volume_circumscribed_ellipsoid_ =
        rank_tol_for_minimum_volume_circumscribed_ellipsoid;
  }

 private:
  geometry::optimization::HPolyhedron DoBuildRegion(
      const Eigen::Ref<const Eigen::MatrixXd>& clique_points) final;

  geometry::optimization::IrisOptions iris_options_;
  std::optional<double> rank_tol_for_minimum_volume_circumscribed_ellipsoid_;
  std::unique_ptr<CollisionChecker> checker_;
};

}  // namespace iris
}  // namespace planning
}  // namespace drake
