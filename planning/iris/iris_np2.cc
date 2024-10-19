#include "drake/planning/iris/iris_np2.h"

#include <iostream>

#include <algorithm>
#include <limits>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <common_robotics_utilities/parallelism.hpp>

#include "drake/common/symbolic/expression.h"
#include "drake/geometry/optimization/affine_ball.h"
#include "drake/geometry/optimization/cartesian_product.h"
#include "drake/geometry/optimization/convex_set.h"
#include "drake/geometry/optimization/iris_internal.h"
#include "drake/geometry/optimization/minkowski_sum.h"
#include "drake/geometry/optimization/vpolytope.h"
#include "drake/geometry/shape_specification.h"
#include "drake/math/autodiff_gradient.h"
#include "drake/multibody/plant/multibody_plant.h"
#include "drake/multibody/tree/joint.h"
#include "drake/multibody/tree/planar_joint.h"
#include "drake/multibody/tree/quaternion_floating_joint.h"
#include "drake/multibody/tree/revolute_joint.h"
#include "drake/multibody/tree/rpy_floating_joint.h"
#include "drake/planning/robot_diagram.h"
#include "drake/solvers/choose_best_solver.h"
#include "drake/solvers/ipopt_solver.h"
#include "drake/solvers/snopt_solver.h"
#include "drake/solvers/solve.h"

namespace drake {
namespace planning {

using Eigen::VectorXd;
using geometry::Role;
using geometry::SceneGraphInspector;
using geometry::optimization::ConvexSet;
using geometry::optimization::internal::ClosestCollisionProgram;
using geometry::optimization::internal::SamePointConstraint;
using multibody::MultibodyPlant;
using systems::Context;

namespace {
// Copied from iris.cc

using geometry::Box;
using geometry::Capsule;
using geometry::Convex;
using geometry::Cylinder;
using geometry::Ellipsoid;
using geometry::HalfSpace;
using geometry::Mesh;
using geometry::Sphere;

using geometry::FrameId;
using geometry::GeometryId;
using geometry::QueryObject;
using geometry::ShapeReifier;

using geometry::optimization::CartesianProduct;
using geometry::optimization::ConvexSet;
using geometry::optimization::HPolyhedron;
using geometry::optimization::Hyperellipsoid;
using geometry::optimization::MinkowskiSum;
using geometry::optimization::VPolytope;

// Constructs a ConvexSet for each supported Shape and adds it to the set.
class IrisConvexSetMaker final : public ShapeReifier {
 public:
  DRAKE_NO_COPY_NO_MOVE_NO_ASSIGN(IrisConvexSetMaker);

  IrisConvexSetMaker(const QueryObject<double>& query,
                     std::optional<FrameId> reference_frame)
      : query_{query}, reference_frame_{reference_frame} {};

  void set_reference_frame(const FrameId& reference_frame) {
    DRAKE_DEMAND(reference_frame.is_valid());
    *reference_frame_ = reference_frame;
  }

  void set_geometry_id(const GeometryId& geom_id) { geom_id_ = geom_id; }

  using ShapeReifier::ImplementGeometry;

  void ImplementGeometry(const Box&, void* data) {
    DRAKE_DEMAND(geom_id_.is_valid());
    auto& set = *static_cast<copyable_unique_ptr<ConvexSet>*>(data);
    // Note: We choose HPolyhedron over VPolytope here, but the IRIS paper
    // discusses a significant performance improvement using a "least-distance
    // programming" instance from CVXGEN that exploited the VPolytope
    // representation.  So we may wish to revisit this.
    set = std::make_unique<HPolyhedron>(query_, geom_id_, reference_frame_);
  }

  void ImplementGeometry(const Capsule&, void* data) {
    DRAKE_DEMAND(geom_id_.is_valid());
    auto& set = *static_cast<copyable_unique_ptr<ConvexSet>*>(data);
    set = std::make_unique<MinkowskiSum>(query_, geom_id_, reference_frame_);
  }

  void ImplementGeometry(const Cylinder&, void* data) {
    DRAKE_DEMAND(geom_id_.is_valid());
    auto& set = *static_cast<copyable_unique_ptr<ConvexSet>*>(data);
    set =
        std::make_unique<CartesianProduct>(query_, geom_id_, reference_frame_);
  }

  void ImplementGeometry(const Ellipsoid&, void* data) {
    DRAKE_DEMAND(geom_id_.is_valid());
    auto& set = *static_cast<copyable_unique_ptr<ConvexSet>*>(data);
    set = std::make_unique<Hyperellipsoid>(query_, geom_id_, reference_frame_);
  }

  void ImplementGeometry(const HalfSpace&, void* data) {
    DRAKE_DEMAND(geom_id_.is_valid());
    auto& set = *static_cast<copyable_unique_ptr<ConvexSet>*>(data);
    set = std::make_unique<HPolyhedron>(query_, geom_id_, reference_frame_);
  }

  void ImplementGeometry(const Sphere&, void* data) {
    DRAKE_DEMAND(geom_id_.is_valid());
    auto& set = *static_cast<copyable_unique_ptr<ConvexSet>*>(data);
    set = std::make_unique<Hyperellipsoid>(query_, geom_id_, reference_frame_);
  }

  void ImplementGeometry(const Convex&, void* data) {
    DRAKE_DEMAND(geom_id_.is_valid());
    auto& set = *static_cast<copyable_unique_ptr<ConvexSet>*>(data);
    set = std::make_unique<VPolytope>(query_, geom_id_, reference_frame_);
  }

  void ImplementGeometry(const Mesh&, void* data) {
    DRAKE_DEMAND(geom_id_.is_valid());
    auto& set = *static_cast<copyable_unique_ptr<ConvexSet>*>(data);
    set = std::make_unique<VPolytope>(query_, geom_id_, reference_frame_);
  }

 private:
  const QueryObject<double>& query_{};
  std::optional<FrameId> reference_frame_{};
  GeometryId geom_id_{};
};

struct GeometryPairWithDistance {
  GeometryId geomA;
  GeometryId geomB;
  double distance;

  GeometryPairWithDistance(GeometryId gA, GeometryId gB, double dist)
      : geomA(gA), geomB(gB), distance(dist) {}

  bool operator<(const GeometryPairWithDistance& other) const {
    return distance < other.distance;
  }
};

int unadaptive_test_samples(double p, double delta, double tau) {
  return static_cast<int>(-2 * std::log(delta) / (tau * tau * p) + 0.5);
}

int FindCollisionPairIndex(
    const MultibodyPlant<double>& plant, Context<double>* context,
    const Eigen::VectorXd& configuration,
    const std::vector<GeometryPairWithDistance>& sorted_pairs) {
  // Call ComputeSignedDistancePairClosestPoints for each pair of collision
  // geometries until finding a pair that is in collision and returning the
  // corresponding index
  int pair_in_collision = -1;
  int i_pair = 0;
  for (const auto& pair : sorted_pairs) {
    plant.SetPositions(context, configuration);
    auto query_object = plant.get_geometry_query_input_port()
                            .template Eval<QueryObject<double>>(*context);
    const double distance =
        query_object
            .ComputeSignedDistancePairClosestPoints(pair.geomA, pair.geomB)
            .distance;
    if (distance < 0.0) {
      pair_in_collision = i_pair;
      break;
    }
    ++i_pair;
  }

  return pair_in_collision;
}

// Add the tangent to the (scaled) ellipsoid at @p point as a
// constraint.
void AddTangentToPolytope(
    const Hyperellipsoid& E, const Eigen::Ref<const Eigen::VectorXd>& point,
    double configuration_space_margin,
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>* A,
    Eigen::VectorXd* b, int* num_constraints) {
  while (*num_constraints >= A->rows()) {
    // Increase pre-allocated polytope size.
    A->conservativeResize(A->rows() * 2, A->cols());
    b->conservativeResize(b->rows() * 2);
  }

  A->row(*num_constraints) =
      (E.A().transpose() * E.A() * (point - E.center())).normalized();
  (*b)[*num_constraints] =
      A->row(*num_constraints) * point - configuration_space_margin;
  if (A->row(*num_constraints) * E.center() > (*b)[*num_constraints]) {
    throw std::logic_error(
        "The current center of the IRIS region is within "
        "options.configuration_space_margin of being infeasible.  Check your "
        "sample point and/or any additional constraints you've passed in via "
        "the options. The configuration space surrounding the sample point "
        "must have an interior.");
  }
  *num_constraints += 1;
}
}  // namespace

HPolyhedron IrisNP2(VectorXd seed, const CollisionChecker& checker,
                    const IrisNP2Options& options) {
  // TODO(cohnt): Check inputs and/or preconditions.

  // Extract relevant objects from the CollisionChecker.
  const RobotDiagram<double>& model = checker.model();
  const multibody::MultibodyPlant<double>& plant = checker.plant();

  // Make the context we will use.
  std::unique_ptr<systems::Context<double>> model_context =
      model.CreateDefaultContext();
  systems::Context<double>& plant_context =
      model.GetMutableSubsystemContext(plant, model_context.get());
  plant.SetPositions(&plant_context, seed);

  const int nq = plant.num_positions();
  // const geometry::SceneGraph<double>& scene_graph =
  // checker.model().scene_graph();

  // Make the polytope and ellipsoid.
  HPolyhedron P_initial = HPolyhedron::MakeBox(plant.GetPositionLowerLimits(),
                                               plant.GetPositionUpperLimits());
  DRAKE_DEMAND(P_initial.A().rows() == 2 * nq);
  if (options.bounding_region) {
    DRAKE_DEMAND(options.bounding_region->ambient_dimension() == nq);
    P_initial = P_initial.Intersection(*options.bounding_region);
  }
  const double kEpsilonEllipsoid = 1e-2;
  Hyperellipsoid E = options.starting_ellipse.value_or(
      Hyperellipsoid::MakeHypersphere(kEpsilonEllipsoid, seed));

  // Make all of the convex sets and supporting quantities.
  auto query_object =
      plant.get_geometry_query_input_port().Eval<QueryObject<double>>(
          plant_context);
  const SceneGraphInspector<double>& inspector = query_object.inspector();
  IrisConvexSetMaker maker(query_object, inspector.world_frame_id());
  std::unordered_map<GeometryId, copyable_unique_ptr<ConvexSet>> sets{};
  std::unordered_map<GeometryId, const multibody::Frame<double>*> frames{};
  const std::vector<GeometryId> geom_ids =
      inspector.GetAllGeometryIds(Role::kProximity);
  copyable_unique_ptr<ConvexSet> temp_set;
  for (GeometryId geom_id : geom_ids) {
    // Make all sets in the local geometry frame.
    FrameId frame_id = inspector.GetFrameId(geom_id);
    maker.set_reference_frame(frame_id);
    maker.set_geometry_id(geom_id);
    inspector.GetShape(geom_id).Reify(&maker, &temp_set);
    sets.emplace(geom_id, std::move(temp_set));
    frames.emplace(geom_id, &plant.GetBodyFromFrameId(frame_id)->body_frame());
  }

  // Verify that the seed point is not in collision.
  auto pairs = inspector.GetCollisionCandidates();
  const int n = static_cast<int>(pairs.size());
  auto same_point_constraint =
      std::make_shared<SamePointConstraint>(&plant, plant_context);
  std::map<std::pair<GeometryId, GeometryId>, std::vector<VectorXd>>
      counter_examples;
  // As a surrogate for the true objective, the pairs are sorted by the distance
  // between each collision pair from the seed point configuration. This could
  // improve computation times and produce regions with fewer faces.
  std::vector<GeometryPairWithDistance> sorted_pairs;
  for (const auto& [geomA, geomB] : pairs) {
    const double distance =
        query_object.ComputeSignedDistancePairClosestPoints(geomA, geomB)
            .distance;
    if (distance < 0.0) {
      for (int i = 0; i < nq; ++i) {
        log()->info("seed: {}", seed(i));
      }

      throw std::runtime_error(
          fmt::format("The seed point is in collision; geometry {} is in "
                      "collision with geometry {}",
                      inspector.GetName(geomA), inspector.GetName(geomB)));
    }
    sorted_pairs.emplace_back(geomA, geomB, distance);
  }

  // On each iteration, we will build the collision-free polytope represented as
  // {x | A * x <= b}.  Here we pre-allocate matrices with a generous maximum
  // size.
  Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> A(
      P_initial.A().rows() + 2 * n, nq);
  VectorXd b(P_initial.A().rows() + 2 * n);
  A.topRows(P_initial.A().rows()) = P_initial.A();
  b.head(P_initial.A().rows()) = P_initial.b();

  int num_initial_constraints = P_initial.A().rows();

  DRAKE_THROW_UNLESS(P_initial.PointInSet(seed, 1e-12));
  double best_volume = E.Volume();
  int iteration = 0;
  VectorXd closest(nq);
  RandomGenerator generator(options.random_seed);

  const solvers::SolverInterface* solver =
      options.solver
          ? options.solver
          : solvers::MakeFirstAvailableSolver(
                {solvers::SnoptSolver::id(), solvers::IpoptSolver::id()})
                .get();

  const std::string seed_point_error_msg =
      "IRIS-NP2: require_sample_point_is_contained is true but "
      "the seed point exited the initial region. Does the provided "
      "options.starting_ellipse not contain the seed point?";
  const std::string seed_point_msg =
      "IRIS-NP2: terminating iterations because the seed point "
      "is no longer in the region.";
  const std::string termination_error_msg =
      "IRIS-NP2: the termination function returned false on "
      "the computation of the initial region. Are the provided "
      "options.starting_ellipse and options.termination_func compatible?";
  const std::string termination_msg =
      "IRIS-NP2: terminating iterations because "
      "options.termination_func returned false.";

  HPolyhedron P;

  // Set up constants for statistical tests.
  double outer_delta_min =
      options.delta * 6 /
      (M_PI * M_PI * options.iteration_limit * options.iteration_limit);

  double delta_min = outer_delta_min * 6 /
                     (M_PI * M_PI * options.max_iterations_separating_planes *
                      options.max_iterations_separating_planes);

  int N_max = unadaptive_test_samples(
      options.admissible_proportion_in_collision, delta_min, options.tau);

  if (options.verbose) {
    log()->info(
        "IRIS-NP2 finding region that is {} collision free with {} certainty ",
        options.admissible_proportion_in_collision, 1 - options.delta);
    log()->info("IRIS-NP2 worst case test requires {} samples.", N_max);
  }

  // TODO(cohnt): Do an argsort so we don't have to have two separate copies
  std::vector<Eigen::VectorXd> particles;
  std::vector<Eigen::VectorXd> particles_in_collision;
  particles.reserve(N_max);
  particles_in_collision.reserve(N_max);

  while (true) {
    log()->info("IRIS-NP2 iteration {}", iteration);
    int num_constraints = num_initial_constraints;
    HPolyhedron P_candidate = HPolyhedron(A.topRows(num_initial_constraints),
                                          b.head(num_initial_constraints));
    DRAKE_ASSERT(best_volume > 0);

    // Separating Planes Step
    int num_iterations_separating_planes = 0;

    double outer_delta =
        options.delta * 6 / (M_PI * M_PI * (iteration + 1) * (iteration + 1));

    // No need for decaying outer delta if we are guaranteed to terminate after
    // one step. In this case we can be less conservative and set it to our
    // total accepted error probability.
    if (options.iteration_limit == 1) {
      outer_delta = options.delta;
    }

    while (num_iterations_separating_planes <
           options.max_iterations_separating_planes) {
      // log()->info("starting inner loop.");
      int k_squared = num_iterations_separating_planes + 1;
      k_squared *= k_squared;
      double delta_k = outer_delta * 6 / (M_PI * M_PI * k_squared);
      int N_k = unadaptive_test_samples(
          options.admissible_proportion_in_collision, delta_k, options.tau);

      particles.resize(N_k);
      particles.at(0) = P_candidate.UniformSample(&generator, E.center(),
                                                  options.mixing_steps);
      // populate particles by uniform sampling
      for (int i = 1; i < N_k; ++i) {
        particles.at(i) = P_candidate.UniformSample(
            &generator, particles.at(i - 1), options.mixing_steps);
      }
      // Find all particles in collision
      std::vector<uint8_t> particle_col_free =
          checker.CheckConfigsCollisionFree(particles, options.parallelism);
      int number_particles_in_collision = 0;

      particles_in_collision.clear();
      for (size_t i = 0; i < particle_col_free.size(); ++i) {
        if (particle_col_free.at(i) == 0) {
          particles_in_collision.push_back(particles.at(i));
          ++number_particles_in_collision;
        }
      }

      // Sort collision order
      auto my_comparator = [](const VectorXd& t1, const VectorXd& t2,
                              const Hyperellipsoid& E_comparator) {
        return (t1 - E_comparator.center()).squaredNorm() <
               (t2 - E_comparator.center())
                   .squaredNorm();  // or use a custom compare function
      };
      std::sort(
          std::begin(particles_in_collision),
          std::begin(particles_in_collision) + number_particles_in_collision,
          std::bind(my_comparator, std::placeholders::_1, std::placeholders::_2,
                    E));

      if (options.verbose) {
        log()->info("IRIS-NP2 N_k {}, N_col {}, thresh {}", N_k,
                    number_particles_in_collision,
                    (1 - options.tau) *
                        options.admissible_proportion_in_collision * N_k);
      }

      // break if threshold is passed
      if (number_particles_in_collision <=
          (1 - options.tau) * options.admissible_proportion_in_collision *
              N_k) {
        break;
      }
      // warn user if test fails on last iteration
      if (num_iterations_separating_planes ==
          options.max_iterations_separating_planes - 1) {
        log()->warn(
            "IRIS-NP2 WARNING, separating planes hit max iterations without "
            "passing the bernoulli test, this voids the probabilistic "
            "guarantees!");
      }

      int num_hyperplanes_added = 0;
      for (const auto& particle : particles_in_collision) {
        if (num_hyperplanes_added > options.max_hyperplanes_per_iteration) {
          break;
        }
        if (!P_candidate.PointInSet(particle)) {
          log()->info("Not in the polytope!");
          continue;
        }

        std::pair<Eigen::VectorXd, int> closest_collision_info;

        closest_collision_info = std::make_pair(
            particle, FindCollisionPairIndex(plant, &plant_context, particle,
                                             sorted_pairs));

        if (closest_collision_info.second >=
            0) {  // pair is actually in collision
          auto pair_iterator =
              std::next(sorted_pairs.begin(), closest_collision_info.second);
          const auto collision_pair = *pair_iterator;

          ClosestCollisionProgram prog(
              same_point_constraint, *frames.at(collision_pair.geomA),
              *frames.at(collision_pair.geomB), *sets.at(collision_pair.geomA),
              *sets.at(collision_pair.geomB), E, A.topRows(num_constraints),
              b.head(num_constraints));
          prog.UpdatePolytope(A.topRows(num_constraints),
                                               b.head(num_constraints));
          if (prog.Solve(*solver, closest_collision_info.first,
                         options.solver_options, &closest)) {
            ++num_hyperplanes_added;
            AddTangentToPolytope(E, closest, options.configuration_space_margin,
                                 &A, &b, &num_constraints);
            P_candidate = HPolyhedron(A.topRows(num_constraints),
                                      b.head(num_constraints));
            if (options.require_sample_point_is_contained) {
              const bool seed_point_requirement =
                  A.row(num_constraints - 1) * seed <= b(num_constraints - 1);
              if (!seed_point_requirement) {
                if (iteration == 0) {
                  throw std::runtime_error(seed_point_error_msg);
                }
                log()->info(seed_point_msg);
                return P;
              }
            }
          } else {
            log()->info("Solve failed!");
          }
        }
      }

      if (options.verbose) {
        log()->info("Added {} hyperplanes", num_hyperplanes_added);
      }

      ++num_iterations_separating_planes;
    }

    P = HPolyhedron(A.topRows(num_constraints), b.head(num_constraints));

    iteration++;
    if (iteration >= options.iteration_limit) {
      log()->info(
          "IRIS-NP2: Terminating because the iteration limit "
          "{} has been reached.",
          options.iteration_limit);
      break;
    }

    E = P.MaximumVolumeInscribedEllipsoid();
    const double volume = E.Volume();
    const double delta_volume = volume - best_volume;
    if (delta_volume <= options.termination_threshold) {
      log()->info(
          "IRIS-NP2: Terminating because the hyperellipsoid "
          "volume change {} is below the threshold {}.",
          delta_volume, options.termination_threshold);
      break;
    } else if (delta_volume / best_volume <=
               options.relative_termination_threshold) {
      log()->info(
          "IRIS-NP2: Terminating because the hyperellipsoid "
          "relative volume change {} is below the threshold {}.",
          delta_volume / best_volume, options.relative_termination_threshold);
      break;
    }
    best_volume = volume;
  }
  return P;
}

}  // namespace planning
}  // namespace drake
