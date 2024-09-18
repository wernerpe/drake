#pragma once

#include <optional>
#include <vector>

#include <Eigen/Dense>

#include "drake/common/parallelism.h"
#include "drake/geometry/optimization/hpolyhedron.h"
#include "drake/planning/collision_checker.h"
#include "drake/planning/iris/point_sampler_base.h"

namespace drake {
namespace planning {
namespace iris {
namespace internal {

// Taken from
// https://github.com/wernerpe/drake/blob/42bca879dc4a1d4a05a21c2279c37cd918f6e70a/planning/iris/fast_clique_inflation.cc#L49
// From conversation with Pete:
// @param p is the admissible fraction of the volume in collision
// @param delta is the admissible probability of making an error of the second
// kind. (The probability of terminating even though the condition is not met)
// @param tau = 0.5 dont worry about it. It is the decision threshold that
// trades off between computational cost and statistical power of the test.
int unadaptive_test_samples(double p, double delta, double tau = 0.5);

// Writes num_points collision free points into the first num_points columns of
// sampled_points by sampling from the point_sampler and rejecting those samples
// in collision. The points are sampled from point_sampler in batches of size
// batch_size.
// sampled_points is conservatively resized to be of size
// checker.plant().num_positions() x num_points.
// If batch_size is std::nullopt, then the batch_size is chosen internally.
// If sets is not nullptr, then the generated samples are sampled outside the
// sets contained in sets.
void SampleCollisionFreePoints(
    int num_points, const CollisionChecker& checker, RandomGenerator* generator,
    PointSamplerBase* point_sampler, Eigen::MatrixXd* sampled_points,
    const std::vector<geometry::optimization::HPolyhedron>* sets = nullptr,
    std::optional<int> batch_size = std::nullopt,
    Parallelism parallelism = Parallelism::Max());

// Returns the number of points which ARE covered by the sets. The number of
// points which are NOT covered are written into
// num_uncovered_collision_free_points. The left-most
// num_uncovered_collision_free_points columns of
// uncovered_collision_free_points are the uncovered collision-free points.
//
// @param uncovered_collision_free_points cannot be nullptr.
// @param num_uncovered_collision_free_points cannot be nullptr.
int ClassifyCollisionFreePoints(
    const Eigen::Ref<const Eigen::MatrixXd>& points,
    const std::vector<geometry::optimization::HPolyhedron>& sets,
    Eigen::MatrixXd* uncovered_collision_free_points,
    int* num_uncovered_collision_free_points);

// Tests whether at least coverage_threshold of the collision free space is
// covered by sets with confidence threshold equal to confidence.
//
//  @param checker
//
//  @param sets [in] The sets which points are tested for containment in.
//
//  @param generator [in] Cannot be nullptr.
//
//  @param point_sampler [in] Cannot be nullptr.
//
//  @param sampled_points [out] Cannot be nullptr. The points used to test
//  sufficient coverage are written into the first num_points_sampled columns of
//  sampled_points. This matrix will be resized to have the same number of
//  columns as points used to perform the test.
//
//  @param num_points_sampled [out] The number of points sampled during the
//  test.
//
//  @param uncovered_collision_free_points [out] the collision free
//  points drawn during the test which are NOT in sets. This matrix will be
//  conservatively resized to have the same number of columns as is used to test
//  whether the configuration space is sufficiently covered and only the first
//  num_uncovered_collision_free_points columns will correspond to
//  uncovered_collision_free_points.
//
//  @param num_uncovered_collision_free_points [out] The first
//  num_uncovered_collision_free_points columns of
//  uncovered_collision_free_points are collision-free and not in the
//  sets.
//
//  @param sampling_batch_size Points are sampled from the point sampler in
//  batches of size sampling_batch_size. An entire batch is checked by the
//  collision checker for collision.

bool IsSufficientlyCovered(
    double coverage_threshold, double confidence,
    const CollisionChecker& checker,
    const std::vector<geometry::optimization::HPolyhedron>& sets,
    RandomGenerator* generator, PointSamplerBase* point_sampler,
    Eigen::MatrixXd* sampled_points, int* num_points_sampled,
    Eigen::MatrixXd* uncovered_collision_free_points,
    int* num_uncovered_collision_free_points,
    std::optional<int> sampling_batch_size = std::nullopt,
    Parallelism parallelism = Parallelism::Max());

}  // namespace internal
}  // namespace iris
}  // namespace planning
}  // namespace drake
