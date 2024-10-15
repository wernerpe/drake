#include "drake/planning/iris/iris_zo_from_clique_builder.h"

#include "drake/geometry/optimization/hpolyhedron.h"
#include "drake/geometry/optimization/hyperellipsoid.h"

namespace drake {
namespace planning {
namespace iris {
using geometry::optimization::HPolyhedron;
using geometry::optimization::Hyperellipsoid;

IrisZoFromCliqueBuilder::IrisZoFromCliqueBuilder(
    const CollisionChecker& checker,
    const std::optional<geometry::optimization::HPolyhedron> domain,
    const IrisZoOptions& options,
    std::optional<double> rank_tol_for_minimum_volume_circumscribed_ellipsoid)
    : RegionFromCliqueBase(),
      checker_(checker.Clone()),
      domain_(domain.value_or(
          HPolyhedron::MakeBox(checker.plant().GetPositionLowerLimits(),
                               checker.plant().GetPositionUpperLimits()))),
      options_(options),
      rank_tol_for_minimum_volume_circumscribed_ellipsoid_(
          rank_tol_for_minimum_volume_circumscribed_ellipsoid) {
  DRAKE_THROW_UNLESS(domain_.ambient_dimension() ==
                     checker_->plant().num_positions());
}

HPolyhedron IrisZoFromCliqueBuilder::DoBuildRegion(
    const Eigen::Ref<const Eigen::MatrixXd>& clique_points) {
  // This may throw if the clique is degenerate.
  Hyperellipsoid clique_ellipse;
  if (rank_tol_for_minimum_volume_circumscribed_ellipsoid_.has_value()) {
    clique_ellipse = Hyperellipsoid::MinimumVolumeCircumscribedEllipsoid(
        clique_points,
        rank_tol_for_minimum_volume_circumscribed_ellipsoid_.value());
  } else {
    clique_ellipse =
        Hyperellipsoid::MinimumVolumeCircumscribedEllipsoid(clique_points);
  }

  if (!checker_->CheckConfigCollisionFree(clique_ellipse.center())) {
    // Find the nearest clique member to the center that is not in collision.
    Eigen::Index nearest_point_col;
    (clique_points - clique_ellipse.center())
        .colwise()
        .norm()
        .minCoeff(&nearest_point_col);
    Eigen::VectorXd center = clique_points.col(nearest_point_col);
    clique_ellipse = Hyperellipsoid(center, clique_ellipse.A());
  }
  return IrisZO(*checker_, clique_ellipse, domain_, options_);
}

void IrisZoFromCliqueBuilder::set_domain(
    const geometry::optimization::HPolyhedron& domain) {
  DRAKE_THROW_UNLESS(domain.ambient_dimension() ==
                     checker_->plant().num_positions());
  domain_ = domain;
}

}  // namespace iris
}  // namespace planning
}  // namespace drake
