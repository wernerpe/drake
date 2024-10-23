#include "drake/planning/iris/barycentric_vpolytope_sampler.h"

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"
#include "drake/common/test_utilities/maybe_pause_for_user.h"
#include "drake/geometry/optimization/hpolyhedron.h"
#include "drake/geometry/optimization/vpolytope.h"
#include "drake/geometry/test_utilities/meshcat_environment.h"
#include "drake/planning/iris/test/clique_cover_test_utils.h"

namespace drake {
namespace planning {
namespace iris {
namespace {
using common::MaybePauseForUser;

using geometry::Rgba;
using geometry::optimization::HPolyhedron;
using geometry::optimization::VPolytope;

GTEST_TEST(BarycentricVPolytopeSamplerTest, SampleUniform2dSimplex) {
  Eigen::MatrixXd vertices{2, 3};
  // clang-format off
  vertices << 1.0, 0.0, 0.0,
              0.0, 1.0, 0.0;
  // clang-format on
  VPolytope domain{vertices};
  BarycentricVPolytopeSampler sampler(domain);
  RandomGenerator generator{0};

  int num_samples = 1000;
  Eigen::MatrixXd samples = sampler.SamplePoints(num_samples, &generator);

  // This distribution should appear uniform.
  {
    std::shared_ptr<geometry::Meshcat> meshcat =
        geometry::GetTestEnvironmentMeshcat();
    meshcat->SetProperty("/Background", "visible", false);
    perception::PointCloud cloud(num_samples);
    cloud.mutable_xyzs().topRows<2>() = samples.cast<float>();
    cloud.mutable_xyzs().bottomRows<1>().setZero();
    meshcat->SetObject("samples", cloud, 0.01, Rgba(0, 0, 1));
    const Eigen::Vector3d color(1.0, 0.0, 0.0);
    internal::Draw2dVPolytope(domain.GetMinimalRepresentation(), "domain",
                              color, meshcat);

    MaybePauseForUser();
  }

  // Check that the sampled points are in the domain.
  for (int i = 0; i < num_samples; ++i) {
    EXPECT_TRUE(domain.PointInSet(samples.col(i), 1e-6));
  }
}

GTEST_TEST(BarycentricVPolytopeSamplerTest, SampleBiased3dSimplex) {
  // This test repeats the (0,0,0) point to bias the samples towards the origin.
  Eigen::MatrixXd simplex_vertices{3, 4};
  // clang-format off
  simplex_vertices << 1.0, 0.0, 0.0, 0.0,
                      0.0, 1.0, 0.0, 0.0,
                      0.0, 0.0, 1.0, 0.0;
  // clang-format on
  int num_zero_bias_points = 5;

  Eigen::MatrixXd vertices{3, simplex_vertices.cols() + num_zero_bias_points};
  vertices.leftCols(simplex_vertices.cols()) = simplex_vertices;
  vertices.rightCols(num_zero_bias_points) =
      Eigen::MatrixXd::Zero(3, num_zero_bias_points);

  VPolytope domain{vertices};
  BarycentricVPolytopeSampler sampler(domain);
  RandomGenerator generator{0};

  int num_samples = 1000;
  Eigen::MatrixXd samples = sampler.SamplePoints(num_samples, &generator);

  // This distribution should appear biased towards the origin.
  {
    std::shared_ptr<geometry::Meshcat> meshcat =
        geometry::GetTestEnvironmentMeshcat();
    meshcat->SetProperty("/Background", "visible", false);
    perception::PointCloud cloud(num_samples);
    cloud.mutable_xyzs().topRows<3>() = samples.cast<float>();
    meshcat->SetObject("samples", cloud, 0.01, Rgba(0, 0, 1));

    Eigen::Matrix3d face;
    const double line_width = 1;
    const geometry::Rgba color(1.0, 0, 0);
    std::vector<std::vector<int>> face_combinations = {
        {0, 1, 2, 0}, {0, 2, 3, 0}, {0, 3, 1, 0}, {1, 2, 3, 1}};
    for (int i = 0; i < 4; ++i) {
      for (int j = 0; j < 4; ++j) {
        face.col(j) = simplex_vertices.col(face_combinations[i][j]);
      }
      meshcat->SetLine(fmt::format("domain/face{}", i), face, line_width,
                       color);
    }
    MaybePauseForUser();
  }
  // Check that the sampled points are in the domain.
  for (int i = 0; i < num_samples; ++i) {
    EXPECT_TRUE(domain.PointInSet(samples.col(i), 1e-6));
  }

  sampler.set_sampling_always_returns_vertices(true);
  samples = sampler.SamplePoints(num_samples, &generator);
  for (int i = 0; i < domain.vertices().cols(); ++i) {
    EXPECT_TRUE(CompareMatrices(samples.col(i), domain.vertices().col(i)));
  }

  samples = sampler.SamplePoints(1, &generator);
  EXPECT_FALSE(CompareMatrices(samples.col(0), domain.vertices().col(0)));
}

GTEST_TEST(BarycentricVPolytopeSamplerTest, ConstructorSettersAndGettersTest) {
  VPolytope domain{HPolyhedron::MakeUnitBox(3)};
  BarycentricVPolytopeSampler sampler(domain, false);

  EXPECT_TRUE(CompareMatrices(sampler.domain().vertices(), domain.vertices()));

  VPolytope domain2{0.3 * domain.vertices()};
  EXPECT_FALSE(CompareMatrices(domain.vertices(), domain2.vertices()));

  sampler.set_domain(domain2);
  EXPECT_TRUE(CompareMatrices(sampler.domain().vertices(), domain2.vertices()));

  EXPECT_FALSE(sampler.sampling_always_returns_vertices());

  sampler.set_sampling_always_returns_vertices(true);
  EXPECT_TRUE(sampler.sampling_always_returns_vertices());
}

}  // namespace
}  // namespace iris
}  // namespace planning
}  // namespace drake
