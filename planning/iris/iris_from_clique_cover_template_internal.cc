#include "drake/planning/iris/iris_from_clique_cover_template_internal.h"

#include <cmath>
#include <iostream>
#include <optional>
#include <vector>

#include "drake/common/drake_throw.h"
#include "drake/common/ssize.h"
#include "drake/geometry/optimization/hpolyhedron.h"
namespace drake {
namespace planning {
namespace iris {
namespace internal {
using geometry::optimization::HPolyhedron;

namespace {
// Convert an Eigen::MatrixXd to a std::vector<Eigen::VectorXd> where every
// entry of the std::vector is a column of the matrix.
std::vector<Eigen::VectorXd> ToColwiseStdVector(
    const Eigen::Ref<const Eigen::MatrixXd>& matrix) {
  std::vector<Eigen::VectorXd> vec;
  vec.reserve(static_cast<int>(matrix.cols()));

  // Convert each column of the matrix to an Eigen::Map<Eigen::VectorXd> and
  // store it in the vector
  for (int i = 0; i < matrix.cols(); ++i) {
    vec.push_back(matrix.col(i));
  }
  return vec;
}
}  // namespace

int unadaptive_test_samples(double p, double delta, double tau) {
  return static_cast<int>(-2 * std::log(delta) / (tau * tau * p) + 0.5);
}

void SampleCollisionFreePoints(
    int num_points, const CollisionChecker& checker, RandomGenerator* generator,
    PointSamplerBase* point_sampler, Eigen::MatrixXd* sampled_points,
    const std::vector<geometry::optimization::HPolyhedron>* sets,
    std::optional<int> batch_size_optional, Parallelism parallelism) {
  auto PointsInSets = [&sets](const Eigen::Ref<const Eigen::MatrixXd>& points) {
    std::vector<bool> ret(points.cols(), false);
    if (sets == nullptr) {
      return ret;
    }
    // TODO(Alexandre.Amice) parallelize this for loop if it becomes a
    // bottleneck.
    for (int i = 0; i < points.cols(); ++i) {
      for (const auto& set : *sets) {
        if (set.PointInSet(points.col(i))) {
          ret.at(i) = true;
          break;
        }
      }
    }
    return ret;
  };

  sampled_points->conservativeResize(checker.plant().num_positions(),
                                     num_points);
  int batch_size = batch_size_optional.value_or(num_points);
  int num_already_sampled = 0;
  while (num_already_sampled < num_points) {
    const Eigen::MatrixXd base_points =
        point_sampler->SamplePoints(batch_size, generator, parallelism);
    if (batch_size == 1) {
      if (checker.CheckConfigCollisionFree(base_points.col(0)) &&
          !PointsInSets(base_points).at(0)) {
        sampled_points->col(num_already_sampled++) = base_points.col(0);
      }
    } else {
      std::vector<uint8_t> config_collision_free =
          checker.CheckConfigsCollisionFree(ToColwiseStdVector(base_points),
                                            parallelism);
      std::vector<bool> config_not_in_set = PointsInSets(base_points);
      for (int i = 0; i < ssize(config_collision_free); ++i) {
        if (static_cast<bool>(config_collision_free[i]) &&
            !config_not_in_set[i]) {
          sampled_points->col(num_already_sampled++) = base_points.col(i);
          if (num_already_sampled >= num_points) {
            return;
          }
        }
      }
    }
  }
}

int ClassifyCollisionFreePoints(
    const Eigen::Ref<const Eigen::MatrixXd>& points,
    const std::vector<HPolyhedron>& sets,
    Eigen::MatrixXd* uncovered_collision_free_points,
    int* num_uncovered_collision_free_points) {
  DRAKE_DEMAND(uncovered_collision_free_points != nullptr);
  DRAKE_DEMAND(num_uncovered_collision_free_points != nullptr);
  uncovered_collision_free_points->conservativeResize(points.rows(),
                                                      points.cols());

  *num_uncovered_collision_free_points = 0;
  for (int i = 0; i < points.cols(); ++i) {
    // Check if the point is covered
    bool covered = false;
    for (const auto& set : sets) {
      if (set.PointInSet(points.col(i))) {
        covered = true;
        break;
      }
    }
    if (!covered) {
      uncovered_collision_free_points->col(
          *num_uncovered_collision_free_points) = points.col(i);
      ++(*num_uncovered_collision_free_points);
    }
  }
  return points.cols() - (*num_uncovered_collision_free_points);
}

bool IsSufficientlyCovered(
    double coverage_threshold, double confidence,
    const CollisionChecker& checker, const std::vector<HPolyhedron>& sets,
    RandomGenerator* generator, PointSamplerBase* point_sampler,
    Eigen::MatrixXd* sampled_points, int* num_points_sampled,
    Eigen::MatrixXd* uncovered_collision_free_points,
    int* num_uncovered_collision_free_points,
    std::optional<int> sampling_batch_size, Parallelism parallelism) {
  DRAKE_THROW_UNLESS(generator != nullptr);
  DRAKE_THROW_UNLESS(point_sampler != nullptr);
  DRAKE_THROW_UNLESS(uncovered_collision_free_points != nullptr);

  (*num_points_sampled) =
      unadaptive_test_samples(1 - coverage_threshold, 1 - confidence, 0.5);
  // The columns of points are the sampled configurations
  SampleCollisionFreePoints(
      *num_points_sampled, checker, generator, point_sampler, sampled_points,
      nullptr /* we want the to sample points inside the sets */,
      sampling_batch_size, parallelism);
  int num_covered_points = ClassifyCollisionFreePoints(
      *sampled_points, sets, uncovered_collision_free_points,
      num_uncovered_collision_free_points);
  std::cout << "Running Coverage check with " << *num_points_sampled
            << "points." << std::endl;
  std::cout << "num_covered_points / *num_points_sampled : "
            << (static_cast<double>(num_covered_points) / *num_points_sampled)
            << std::endl;
  std::cout << "coverage_threshold : " << coverage_threshold << std::endl
            << std::endl;
  return (static_cast<double>(num_covered_points) / *num_points_sampled) >
         coverage_threshold;
}

}  // namespace internal
}  // namespace iris
}  // namespace planning
}  // namespace drake
