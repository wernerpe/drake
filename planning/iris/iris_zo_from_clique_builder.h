#pragma once

#include <memory>
#include <optional>

#include "drake/geometry/optimization/hpolyhedron.h"
#include "drake/planning/collision_checker.h"
#include "drake/planning/iris/iris_zo.h"
#include "drake/planning/iris/region_from_clique_base.h"

namespace drake {
namespace planning {
namespace iris {

/**
 * Given a clique, this class constructs a region by computing the maximum
 * volume inscribed ellipsoid of the clique and then running
 * IrisZo with the initial point and metric set to the
 * center of the ellipse and the ellipse respectively and using the plant and
 * scene graph contained in checker.
 *
 * @param: domain The domain used during IrisZo. If not passed, then initialize
 * to the joint limits in the collision checker.
 *
 * TODO(Alexandre.Amice) currently I clone the collision checker. This is
 * wasteful so consider a better design when going to master.
 */
class IrisZoFromCliqueBuilder final : public RegionFromCliqueBase {
 public:
  IrisZoFromCliqueBuilder(
      const CollisionChecker& checker,
      const std::optional<geometry::optimization::HPolyhedron> domain =
          std::nullopt,
      const IrisZoOptions& options = IrisZoOptions(),
      std::optional<double>
          rank_tol_for_minimum_volume_circumscribed_ellipsoid = std::nullopt);

  /**
   * The options used during IrisZo.
   * @return
   */
  const IrisZoOptions& options() const { return options_; }

  /**
   * Sets the options used during IrisZo.
   * @return
   */
  void set_options(const IrisZoOptions& iris_options) {
    options_ = iris_options;
  }

  /**
   * The domain used during IrisZo.
   * @return
   */
  const geometry::optimization::HPolyhedron& domain() const { return domain_; }

  /**
   * Sets the domain used during IrisZo.
   * @return
   */
  void set_domain(const geometry::optimization::HPolyhedron& domain) {
    DRAKE_THROW_UNLESS(domain.ambient_dimension() ==
                       checker_->plant().num_positions());
    domain_ = domain;
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

  const std::unique_ptr<CollisionChecker> checker_;
  geometry::optimization::HPolyhedron domain_;
  IrisZoOptions options_;
  std::optional<double> rank_tol_for_minimum_volume_circumscribed_ellipsoid_;
};

}  // namespace iris
}  // namespace planning
}  // namespace drake
