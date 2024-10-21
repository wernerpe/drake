#include "drake/planning/iris/iris_np_from_clique_builder.h"

#include "drake/geometry/optimization/hpolyhedron.h"

namespace drake {
namespace planning {
namespace iris {
using geometry::optimization::HPolyhedron;
using geometry::optimization::Hyperellipsoid;

IrisNpFromCliqueBuilder::IrisNpFromCliqueBuilder(
    const CollisionChecker& checker,
    const geometry::optimization::IrisOptions& options,
    std::optional<double> rank_tol_for_minimum_volume_circumscribed_ellipsoid)
    : RegionFromCliqueBase(),
      iris_options_(options),
      rank_tol_for_minimum_volume_circumscribed_ellipsoid_(
          rank_tol_for_minimum_volume_circumscribed_ellipsoid),
      checker_(checker.Clone()) {}

HPolyhedron IrisNpFromCliqueBuilder::DoBuildRegion(
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

  if (checker_->CheckConfigCollisionFree(clique_ellipse.center())) {
    iris_options_.starting_ellipse = clique_ellipse;
  } else {
    // Find the nearest clique member to the center that is not in collision.
    Eigen::Index nearest_point_col;
    (clique_points - clique_ellipse.center())
        .colwise()
        .norm()
        .minCoeff(&nearest_point_col);
    Eigen::VectorXd center = clique_points.col(nearest_point_col);
    iris_options_.starting_ellipse = Hyperellipsoid(center, clique_ellipse.A());
  }
  checker_->UpdatePositions(iris_options_.starting_ellipse->center());
  log()->debug("Iris is constructing a set.");
  return IrisInConfigurationSpace(checker_->plant(), checker_->plant_context(0),
                                  iris_options_);
}

}  // namespace iris
}  // namespace planning
}  // namespace drake
