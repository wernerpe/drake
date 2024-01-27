#pragma once

#include <filesystem>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <Eigen/Dense>

#include "drake/common/parallelism.h"
#include "drake/geometry/meshcat.h"
#include "drake/geometry/optimization/convex_set.h"
#include "drake/geometry/optimization/hpolyhedron.h"
#include "drake/planning/collision_checker.h"

namespace drake {
namespace geometry {
namespace optimization {

struct FastIrisOptions {
  /** Passes this object to an Archive.
  Refer to @ref yaml_serialization "YAML Serialization" for background.
  Note: This only serializes options that are YAML built-in types. */
  template <typename Archive>
  void Serialize(Archive* a) {
    a->Visit(DRAKE_NVP(num_particles));
    a->Visit(DRAKE_NVP(num_consecutive_failures));
    a->Visit(DRAKE_NVP(max_iterations));
    a->Visit(DRAKE_NVP(gradient_steps));
    a->Visit(DRAKE_NVP(bisection_steps));
    // a->Visit(DRAKE_NVP(num_resampling_steps));
    a->Visit(DRAKE_NVP(configuration_space_margin));
    a->Visit(DRAKE_NVP(random_seed));
  }

  FastIrisOptions() = default;

  /** Number of particles used to estimate the closest collision*/
  int num_particles = 1e3;

  /** Number of consecutive failures to find a collision through sampling the
   * polytope*/
  int num_consecutive_failures = 1;

  /** Number of resampling steps for the gradient updates*/
  // int num_resampling_steps = 1;

  /** Maximum number of rounds of adding faces to the polytope*/
  int max_iterations = 100;

  /** Maximum number of bisection steps per gradient step*/
  int bisection_steps = 10;

  /** Number of gradient steps per particle*/
  int gradient_steps = 1;
  
  /* Enables print statements indicating the progress of fast iris**/
  bool verbose = true;

  /** The initial polytope is guaranteed to contain the point if that point is
  collision-free. However, the IRIS alternation objectives do not include (and
  can not easily include) a constraint that the original sample point is
  contained. Therefore, the IRIS paper recommends that if containment is a
  requirement, then the algorithm should simply terminate early if alternations
  would ever cause the set to not contain the point. */
  // bool require_sample_point_is_contained{true};

  /** For IRIS in configuration space, we retreat by this margin from each
  C-space obstacle in order to avoid the possibility of requiring an infinite
  number of faces to approximate a curved boundary.
  */
  double configuration_space_margin{1e-2};

  /** For IRIS in configuration space, it can be beneficial to not only specify
  task-space obstacles (passed in through the plant) but also obstacles that are
  defined by convex sets in the configuration space. This option can be used to
  pass in such configuration space obstacles. */
  // ConvexSets configuration_obstacles{};

  /** The only randomization in IRIS is the random sampling done to find
  counter-examples for the additional constraints using in
  IrisInConfigurationSpace. Use this option to set the initial seed. */
  int random_seed{1234};

  /** Passing a meshcat instance may enable debugging visualizations; this
  currently and when the
  configuration space is <= 3 dimensional.*/
  std::shared_ptr<Meshcat> meshcat{};
};

/** Given a Seed point and an initial ellipsoidal metric, use sampling based
optimization to find a collision free polytope in cspace.*/

HPolyhedron FastIris(const planning::CollisionChecker& checker,
                     const Hyperellipsoid& starting_ellipsoid,
                     const HPolyhedron& domain,
                     const FastIrisOptions& options = FastIrisOptions());

}  // namespace optimization
}  // namespace geometry
}  // namespace drake