#include "drake/planning/iris/iris_in_configuration_space_clique_inflation.h"

#include "drake/geometry/optimization/hpolyhedron.h"

namespace drake {
namespace planning {
namespace iris {
using geometry::optimization::HPolyhedron;
using geometry::optimization::Hyperellipsoid;

IrisInConfigurationSpaceCliqueInflation::
    IrisInConfigurationSpaceCliqueInflation(
        const CollisionChecker& checker,
        const geometry::optimization::IrisOptions& iris_options,
        std::optional<double>
            rank_tol_for_minimum_volume_circumscribed_ellipsoid)
    : RegionFromCliqueBase(),
      iris_options_(iris_options),
      rank_tol_for_minimum_volume_circumscribed_ellipsoid_(
          rank_tol_for_minimum_volume_circumscribed_ellipsoid),
      checker_(checker.Clone()) {}

HPolyhedron IrisInConfigurationSpaceCliqueInflation::DoBuildRegion(
    const Eigen::Ref<const Eigen::MatrixXd>& clique_points) {
  //    Hyperellipsoid clique_ellipse;
  //    try {
  //      clique_ellipse =
  //      Hyperellipsoid::MinimumVolumeCircumscribedEllipsoid(
  //          clique_points,
  //          rank_tol_for_minimum_volume_circumscribed_ellipsoid_);
  //    } catch (const std::runtime_error& e) {
  //      log()->info("Iris failed to compute an ellipse for a clique.",
  //      e.what()); current_clique = computed_cliques->pop(); continue;
  //    }

  // Temporarily allow throws
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
