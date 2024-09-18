#pragma once
#include <optional>

#include "drake/common/parallelism.h"
#include "drake/geometry/optimization/iris.h"

namespace drake {
namespace planning {
namespace iris {

struct IrisFromCliqueCoverOptions {
  /**
   * The options used on internal calls to Iris.  Currently, it is recommended
   * to only run Iris for one iteration when building from a clique so as to
   * avoid discarding the information gained from the clique.
   *
   * Note that `IrisOptions` can optionally include a meshcat instance to
   * provide debugging visualization. If this is provided `IrisFromCliqueCover`
   * will provide debug visualization in meshcat showing where in configuration
   * space it is drawing from. However, if the parallelism option is set to
   * allow more than 1 thread, then the debug visualizations of internal Iris
   * calls will be disabled. This is due to a limitation of drawing to meshcat
   * from outside the main thread.
   */
  geometry::optimization::IrisOptions iris_options{.iteration_limit = 1};

  /**
   * The fraction of the domain that must be covered before we terminate the
   * algorithm.
   */
  double coverage_termination_threshold{0.7};

  /**
   * The maximum number of iterations of the algorithm.
   */
  int iteration_limit{100};

  /**
   * The number of points to sample when testing coverage.
   */
  int num_points_per_coverage_check{static_cast<int>(1e3)};

  /**
   * The amount of parallelism to use. This algorithm makes heavy use of
   * parallelism at many points and thus it is highly recommended to set this to
   * the maximum tolerable parallelism.
   */
  Parallelism parallelism{Parallelism::Max()};

  /**
   * The minimum size of the cliques used to construct a region. If this is set
   * lower than the ambient dimension of the space we are trying to cover, then
   * this option will be overridden to be at least 1 + the ambient dimension.
   */
  int minimum_clique_size{3};

  /**
   * Number of points to sample when building visibilty cliques. If this option
   * is less than twice the minimum clique size, it will be overridden to be at
   * least twice the minimum clique size. If the algorithm ever fails to find a
   * single clique in a visibility round, then the number of points in a
   * visibility round will be doubled.
   */
  int num_points_per_visibility_round{200};

  /**
   * The rank tolerance used for computing the
   * MinimumVolumeCircumscribedEllipsoid of a clique. See
   * @MinimumVolumeCircumscribedEllipsoid.
   */
  double rank_tol_for_minimum_volume_circumscribed_ellipsoid{1e-6};

  /**
   * The tolerance used for checking whether a point is contained inside an
   * HPolyhedron. See @ConvexSet::PointInSet.
   */
  double point_in_set_tol{1e-6};

  /** Confidence in the coverage threshold test. */
  double confidence{0.95};

  /** Whether to sample points outside of the collision-free sets at every
   * round. */
  bool sample_outside_of_sets{true};

  /** Whether to compute a clique cover or partition. See
   * MinCliqueCoverSolverBase for details. */
  bool partition{false};

  /** Points which are sampled during IrisFromCliqueCover are sampled in batches
   * of this size */
  std::optional<int> sampling_batch_size{std::nullopt};
};

}  // namespace iris
}  // namespace planning
}  // namespace drake
