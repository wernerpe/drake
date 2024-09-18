#include "drake/planning/iris/hpolyhedron_point_sampler.h"

#include <gtest/gtest.h>

#include "drake/common/test_utilities/eigen_matrix_compare.h"

namespace drake {
namespace planning {
namespace iris {
namespace {
using geometry::optimization::HPolyhedron;
GTEST_TEST(HPolyhedronPointSamplerTest, SamplePoints) {
  const int num_samples = 2;
  const int dimension = 5;
  const int mixing_steps = 3;
  Eigen::VectorXd first_point = Eigen::VectorXd::Ones(dimension);
  HPolyhedron domain = HPolyhedron::MakeUnitBox(dimension);
  HPolyhedronPointSampler sampler(domain, mixing_steps, first_point);
  RandomGenerator generator{0};
  Eigen::MatrixXd points = sampler.SamplePoints(num_samples, &generator);

  generator = RandomGenerator(0);
  Eigen::VectorXd last_point = first_point;
  Eigen::MatrixXd points_expected =
      Eigen::MatrixXd::Zero(dimension, num_samples);
  for (int i = 0; i < num_samples; ++i) {
    last_point = domain.UniformSample(&generator, last_point, mixing_steps);
    points_expected.col(i) = last_point;
  }
  EXPECT_TRUE(CompareMatrices(points, points_expected));
}

GTEST_TEST(HPolyhedronPointSamplerTest, ConstructorSettersAndGettersTest) {
  const int dimension = 5;
  const int mixing_steps = 3;
  Eigen::VectorXd first_point = Eigen::VectorXd::Ones(dimension);
  HPolyhedron domain = HPolyhedron::MakeUnitBox(dimension);
  HPolyhedronPointSampler sampler(domain, mixing_steps, first_point);
  EXPECT_EQ(sampler.mixing_steps(), mixing_steps);
  EXPECT_EQ(sampler.last_point(), first_point);
  sampler.set_mixing_steps(5);
  EXPECT_EQ(sampler.mixing_steps(), 5);

  HPolyhedronPointSampler sampler2(domain);
  EXPECT_EQ(sampler2.last_point(), domain.ChebyshevCenter());
}

}  // namespace
}  // namespace iris
}  // namespace planning
}  // namespace drake
