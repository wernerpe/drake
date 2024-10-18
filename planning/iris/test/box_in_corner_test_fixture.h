#pragma once
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "drake/common/find_resource.h"
#include "drake/common/ssize.h"
#include "drake/geometry/optimization/hpolyhedron.h"
#include "drake/geometry/optimization/hyperrectangle.h"
#include "drake/geometry/optimization/vpolytope.h"
#include "drake/geometry/test_utilities/meshcat_environment.h"
#include "drake/multibody/parsing/parser.h"
#include "drake/planning/iris/hpolyhedron_point_sampler.h"
#include "drake/planning/iris/iris_from_clique_cover_options.h"
#include "drake/planning/iris/test/clique_cover_test_utils.h"
#include "drake/planning/robot_diagram_builder.h"
#include "drake/planning/scene_graph_collision_checker.h"
#include "drake/systems/framework/diagram_builder.h"
namespace drake {
namespace planning {
namespace iris {

const char boxes_in_corners[] = R"""(
<robot name="boxes">
  <link name="fixed">
    <collision name="top_left">
      <origin rpy="0 0 0" xyz="-1 1 0"/>
      <geometry><box size="1.4 1.4 1.4"/></geometry>
    </collision>
    <collision name="top_right">
      <origin rpy="0 0 0" xyz="1 1 0"/>
      <geometry><box size="1.4 1.4 1.4"/></geometry>
    </collision>
    <collision name="bottom_left">
      <origin rpy="0 0 0" xyz="-1 -1 0"/>
      <geometry><box size="1.4 1.4 1.4"/></geometry>
    </collision>
    <collision name="bottom_right">
      <origin rpy="0 0 0" xyz="1 -1 0"/>
      <geometry><box size="1.4 1.4 1.4"/></geometry>
    </collision>
  </link>
  <joint name="fixed_link_weld" type="fixed">
    <parent link="world"/>
    <child link="fixed"/>
  </joint>
  <link name="movable">
    <collision name="sphere">
      <geometry><sphere radius="0.01"/></geometry>
    </collision>
  </link>
  <link name="for_joint"/>
  <joint name="x" type="prismatic">
    <axis xyz="1 0 0"/>
    <limit lower="-2" upper="2"/>
    <parent link="world"/>
    <child link="for_joint"/>
  </joint>
  <joint name="y" type="prismatic">
    <axis xyz="0 1 0"/>
    <limit lower="-2" upper="2"/>
    <parent link="for_joint"/>
    <child link="movable"/>
  </joint>
</robot>
)""";
/* A movable sphere with fixed boxes in all corners.
┌───────────────┐
│┌────┐   ┌────┐│
││    │   │    ││
│└────┘   └────┘│
│       o       │
│┌────┐   ┌────┐│
││    │   │    ││
│└────┘   └────┘│
└───────────────┘ */
class BoxInCornerTestFixture : public ::testing::Test {
 public:
  Eigen::MatrixXd SamplePointsFromManualDecompositionPiece(
      int num_points, int piece_number, RandomGenerator* rng,
      int mixing_steps = 10) {
    DRAKE_THROW_UNLESS(piece_number >= 0 && piece_number < 6);
    manual_decomposition_samplers[piece_number].set_mixing_steps(mixing_steps);
    return manual_decomposition_samplers[piece_number].SamplePoints(num_points,
                                                                    rng);
  }

 protected:
  void SetUp() override {
    params = CollisionCheckerParams();

    meshcat = geometry::GetTestEnvironmentMeshcat();
    meshcat->Delete("/drake");
    //    meshcat->Set2dRenderMode(math::RigidTransformd(Eigen::Vector3d{0, 0,
    //    1}),
    //                             -3.25, 3.25, -3.25, 3.25);
    meshcat->SetProperty("/Grid", "visible", true);
    // Draw the true cspace.
    Eigen::Matrix3Xd env_points(3, 5);
    // clang-format off
    env_points << -2, 2,  2, -2, -2,
                   2, 2, -2, -2,  2,
                   0, 0,  0,  0,  0;
    // clang-format on
    meshcat->SetLine("Domain", env_points, 8.0, geometry::Rgba(0, 0, 0));
    Eigen::Matrix3Xd centers(3, 4);
    double c = 1.0;
    // clang-format off
    centers << -c, c,  c, -c,
                c, c, -c, -c,
                0, 0,  0,  0;
    // clang-format on
    Eigen::Matrix3Xd obs_points(3, 5);
    // approximating offset due to sphere radius with fixed offset
    double s = 0.7 + 0.01;
    // clang-format off
    obs_points << -s, s,  s, -s, -s,
                   s, s, -s, -s,  s,
                   s, 0,  0,  0,  0;
    // clang-format on
    for (int obstacle_idx = 0; obstacle_idx < 4; ++obstacle_idx) {
      Eigen::Matrix3Xd obstacle = obs_points;
      obstacle.colwise() += centers.col(obstacle_idx);
      meshcat->SetLine(fmt::format("/obstacles/obs_{}", obstacle_idx), obstacle,
                       8.0, geometry::Rgba(0, 0, 0));
    }

    RobotDiagramBuilder<double> builder(0.0);
    params.robot_model_instances =
        builder.parser().AddModelsFromString(boxes_in_corners, "urdf");
    params.edge_step_size = 0.01;

    params.model = builder.Build();
    checker = std::make_unique<SceneGraphCollisionChecker>(std::move(params));
    iris_from_clique_cover_options.iris_options.meshcat = meshcat;

    iris_from_clique_cover_options.num_points_per_coverage_check = 1000;
    iris_from_clique_cover_options.num_points_per_visibility_round = 140;
    iris_from_clique_cover_options.coverage_termination_threshold = 0.9;
    iris_from_clique_cover_options.minimum_clique_size = 4;

    generator = RandomGenerator(0);

    // A manual convex decomposition of the space.
    manual_decomposition.push_back(geometry::optimization::Hyperrectangle(
        Eigen::Vector2d{-2, -2}, Eigen::Vector2d{-1.7, 2}));
    manual_decomposition.push_back(geometry::optimization::Hyperrectangle(
        Eigen::Vector2d{-2, -2}, Eigen::Vector2d{2, -1.7}));
    manual_decomposition.push_back(geometry::optimization::Hyperrectangle(
        Eigen::Vector2d{1.7, -2}, Eigen::Vector2d{2, 2}));
    manual_decomposition.push_back(geometry::optimization::Hyperrectangle(
        Eigen::Vector2d{-2, 1.7}, Eigen::Vector2d{2, 2}));
    manual_decomposition.push_back(geometry::optimization::Hyperrectangle(
        Eigen::Vector2d{-0.3, -2}, Eigen::Vector2d{0.3, 2}));
    manual_decomposition.push_back(geometry::optimization::Hyperrectangle(
        Eigen::Vector2d{-2, -0.3}, Eigen::Vector2d{2, 0.3}));

    color = Eigen::VectorXd::Zero(3);
    // Show the manual decomposition in the meshcat debugger.
    for (int i = 0; i < ssize(manual_decomposition); ++i) {
      // Choose a random color.
      for (int j = 0; j < color.size(); ++j) {
        color[j] = abs(gaussian(generator));
      }
      color.normalize();
      geometry::optimization::VPolytope vregion =
          geometry::optimization::VPolytope(
              manual_decomposition.at(i).MakeHPolyhedron())
              .GetMinimalRepresentation();
      internal::Draw2dVPolytope(
          vregion, fmt::format("manual_decomposition_{}", i), color, meshcat);
    }
  }

  CollisionCheckerParams params;
  std::shared_ptr<geometry::Meshcat> meshcat;
  std::unique_ptr<SceneGraphCollisionChecker> checker;
  IrisFromCliqueCoverOptions iris_from_clique_cover_options;
  RandomGenerator generator;
  std::vector<geometry::optimization::Hyperrectangle> manual_decomposition;
  std::vector<HPolyhedronPointSampler> manual_decomposition_samplers;
  std::normal_distribution<double> gaussian;
  Eigen::VectorXd color;
};
}  // namespace iris
}  // namespace planning
}  // namespace drake
