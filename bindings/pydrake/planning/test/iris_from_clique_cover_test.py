import unittest

import pydrake.planning as mut
from pydrake.common import RandomGenerator, Parallelism
from pydrake.planning import (
    RobotDiagramBuilder,
    SceneGraphCollisionChecker,
    CollisionCheckerParams,
)
from pydrake.solvers import MosekSolver, GurobiSolver, SnoptSolver
from pydrake.geometry.optimization import HPolyhedron, Hyperellipsoid
from pydrake.geometry.optimization import IrisInConfigurationSpace
from pydrake.geometry.optimization import IrisOptions
from pydrake.common.test_utilities import numpy_compare

import textwrap
import numpy as np
import scipy.sparse as sp
import scipy


def _snopt_and_mip_solver_available():
    mip_solver_available = (
       MosekSolver().available() and MosekSolver().enabled() or (
        GurobiSolver().available() and GurobiSolver().enabled()
        )
    )
    snopt_solver_available = (
            SnoptSolver().available() and SnoptSolver().enabled()
    )
    return mip_solver_available and snopt_solver_available


def cross_cspace_urdf():
    return """
        <robot name="boxes">
          <link name="fixed">
            <collision name="top_left">
              <origin rpy="0 0 0" xyz="-1 1 0"/>
              <geometry><box size="1 1 1"/></geometry>
            </collision>
            <collision name="top_right">
              <origin rpy="0 0 0" xyz="1 1 0"/>
              <geometry><box size="1 1 1"/></geometry>
            </collision>
            <collision name="bottom_left">
              <origin rpy="0 0 0" xyz="-1 -1 0"/>
              <geometry><box size="1 1 1"/></geometry>
            </collision>
            <collision name="bottom_right">
              <origin rpy="0 0 0" xyz="1 -1 0"/>
              <geometry><box size="1 1 1"/></geometry>
            </collision>
          </link>
          <joint name="fixed_link_weld" type="fixed">
            <parent link="world"/>
            <child link="fixed"/>
          </joint>
          <link name="movable">
            <collision name="sphere">
              <geometry><sphere radius="0.1"/></geometry>
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
"""


class TestIrisFromCliqueCover(unittest.TestCase):
    def _make_robot_diagram(self):
        # Code taken from
        # bindings/pydrake/planning/test/collision_checker_test.py
        builder = mut.RobotDiagramBuilder()
        scene_yaml = textwrap.dedent(
            """
        directives:
        - add_model:
            name: box
            file: package://drake/multibody/models/box.urdf
        - add_model:
            name: ground
            file: package://drake/planning/test_utilities/collision_ground_plane.sdf  # noqa
        - add_weld:
            parent: world
            child: ground::ground_plane_box
        """
        )
        builder.parser().AddModelsFromString(scene_yaml, "dmd.yaml")
        model_instance_index = builder.plant().GetModelInstanceByName("box")
        robot_diagram = builder.Build()
        return (robot_diagram, model_instance_index)

    def _make_scene_graph_collision_checker(self, use_provider, use_function):
        # Code taken from
        # bindings/pydrake/planning/test/collision_checker_test.py
        self.assertFalse(use_provider and use_function)

        robot, index = self._make_robot_diagram()
        plant = robot.plant()
        checker_kwargs = dict(
            model=robot, robot_model_instances=[index], edge_step_size=0.125
        )

        if use_provider:
            checker_kwargs[
                "distance_and_interpolation_provider"
            ] = mut.LinearDistanceAndInterpolationProvider(plant)
        if use_function:
            checker_kwargs[
                "configuration_distance_function"
            ] = self._configuration_distance

        return mut.SceneGraphCollisionChecker(**checker_kwargs)

    def test_iris_in_configuration_space_from_clique_cover_options(self):
        options = mut.IrisFromCliqueCoverOptions()
        options.iris_options.iteration_limit = 2
        self.assertEqual(options.iris_options.iteration_limit, 2)
        options.coverage_termination_threshold = 1e-3
        self.assertEqual(options.coverage_termination_threshold, 1e-3)
        options.iteration_limit = 10
        self.assertEqual(options.iteration_limit, 10)
        options.num_points_per_coverage_check = 100
        self.assertEqual(options.num_points_per_coverage_check, 100)
        options.parallelism = Parallelism(3)
        self.assertEqual(options.parallelism.num_threads(), 3)
        options.minimum_clique_size = 2
        self.assertEqual(options.minimum_clique_size, 2)
        options.num_points_per_visibility_round = 150
        self.assertEqual(options.num_points_per_visibility_round, 150)
        options.rank_tol_for_minimum_volume_circumscribed_ellipsoid = 1e-3
        self.assertEqual(
            options.rank_tol_for_minimum_volume_circumscribed_ellipsoid, 1e-3
        )
        options.point_in_set_tol = 1e-5
        self.assertEqual(options.point_in_set_tol, 1e-5)

        options.sample_outside_of_sets = False
        self.assertFalse(options.sample_outside_of_sets)
        options.partition = False
        self.assertFalse(options.partition)

        self.assertTrue(options.sampling_batch_size is None)
        options.sampling_batch_size = 10
        self.assertEqual(options.sampling_batch_size, 10)

    # IPOPT performs poorly on this test. We also need a MIP solver to
    # be available. Hence only run this test if both SNOPT and a MIP solver
    # are available.
    @unittest.skipUnless(
        _snopt_and_mip_solver_available(), "Requires Snopt and a MIP solver"
    )
    def test_iris_in_configuration_space_from_clique_cover(self):
        urdf = cross_cspace_urdf()
        params = dict(edge_step_size=0.125)
        builder = RobotDiagramBuilder()
        params["robot_model_instances"] = builder.parser().AddModelsFromString(
            urdf, "urdf"
        )
        params["model"] = builder.Build()
        checker = SceneGraphCollisionChecker(**params)

        options = mut.IrisFromCliqueCoverOptions()
        options.num_points_per_coverage_check = 10
        options.num_points_per_visibility_round = 25

        # We can achieve almost 100% coverage with 2 regions.
        options.coverage_termination_threshold = 0.99
        options.iteration_limit = 3

        generator = RandomGenerator(0)

        sets = mut.IrisInConfigurationSpaceFromCliqueCover(
            checker=checker, options=options, generator=generator, sets=[]
        )
        self.assertGreaterEqual(len(sets), 2)
        sets = mut.IrisInConfigurationSpaceFromCliqueCoverV2(
            checker=checker, options=options, generator=generator, sets=[]
        )
        self.assertGreaterEqual(len(sets), 2)

        class DummyMaxCliqueSolver(mut.MaxCliqueSolverBase):
            def __init__(self, name):
                mut.MaxCliqueSolverBase.__init__(self)
                self.name = name

            def DoSolveMaxClique(self, adjacency_matrix):
                return np.ones(adjacency_matrix.shape[1])

        # Check that a Python solver can be used with 1 thread.
        options.parallelism = Parallelism(1)
        sets = mut.IrisInConfigurationSpaceFromCliqueCover(
            checker=checker,
            options=options,
            generator=generator,
            sets=[],
            max_clique_solver=DummyMaxCliqueSolver("dummy"),
        )
        self.assertGreaterEqual(len(sets), 1)

    def test_adjacency_matrix_builder_base_subclassable(self):
        class DummyAdjacencyMatrixBuilder(mut.AdjacencyMatrixBuilderBase):
            def __init__(self, name):
                mut.AdjacencyMatrixBuilderBase.__init__(self)
                self.name = name

            def DoBuildAdjacencyMatrix(self, points):
                return np.ones((points.shape[1], points.shape[1]))

        builder = DummyAdjacencyMatrixBuilder("dummy")
        self.assertEqual(builder.name, "dummy")
        n, m = 3, 5
        points = np.ones((n, m))
        adjacency_matrix = builder.BuildAdjacencyMatrix(points=points)
        adjacency_matrix_dense = adjacency_matrix.todense()
        numpy_compare.assert_equal(
            adjacency_matrix_dense,
            np.ones((m, m))
        )

    def test_point_sampler_base_subclassable(self):
        class DummyPointSampler(mut.PointSamplerBase):
            def __init__(self, dimension):
                mut.PointSamplerBase.__init__(self)
                self.dimension = dimension
                self.cur_point = -1

            def DoSamplePoints(self, num_points, generator, parallelism):
                ret = np.zeros((self.dimension, num_points))
                for i in range(num_points):
                    self.cur_point += 1
                    ret[0, i] = self.cur_point
                return ret

        dim = 4
        builder = DummyPointSampler(dim)
        self.assertEqual(builder.dimension, dim)
        num_points = 3
        sampled_points_expected = np.zeros((dim, num_points))
        sampled_points_expected[0, 0] = 0
        sampled_points_expected[0, 1] = 1
        sampled_points_expected[0, 2] = 2
        sampled_points = builder.SamplePoints(
            num_points=num_points,
            generator=RandomGenerator(0),
            parallelism=Parallelism(1)
        )
        numpy_compare.assert_equal(
            sampled_points,
            sampled_points_expected
        )

    def test_hpolyhedron_point_sampler(self):
        domain = HPolyhedron.MakeUnitBox(3)
        first_point = np.ones(3)*0.25
        point_sampler = mut.HPolyhedronPointSampler(domain=domain,
                                                    mixing_steps=100,
                                                    first_point=first_point)
        self.assertEqual(point_sampler.mixing_steps(), 100)
        point_sampler.set_mixing_steps(mixing_steps=10)
        self.assertEqual(point_sampler.mixing_steps(), 10)
        numpy_compare.assert_equal(
            point_sampler.last_point(),
            first_point
        )
        generator = RandomGenerator(0)
        points = point_sampler.SamplePoints(6, generator)
        self.assertEqual(points.shape, (3, 6))

    def test_region_from_clique_base_subclassable(self):
        class DummyRegionBuilder(mut.RegionFromCliqueBase):
            def __init__(self, name):
                mut.RegionFromCliqueBase.__init__(self)
                self.name = name

            def DoBuildRegion(self, clique_points):
                lb = clique_points.min(axis=1)
                ub = clique_points.max(axis=1)
                return HPolyhedron.MakeBox(lb, ub)

        builder = DummyRegionBuilder("dummy")
        self.assertEqual(builder.name, "dummy")
        clique = np.array([
            [0, 0, 0],
            [-1, 0, 1],
            [0, -2, 1]
        ])
        region_expected = HPolyhedron.MakeBox(
            np.array([0, -1, -2]),
            np.array([0, 1, 1])
        )
        region = builder.BuildRegion(clique_points=clique)
        numpy_compare.assert_equal(
            region.A(),
            region_expected.A()
        )
        numpy_compare.assert_equal(
            region.b(),
            region_expected.b()
        )

    def test_iris_np_from_clique_builder(self):
        urdf = cross_cspace_urdf()
        params = dict(edge_step_size=0.125)
        builder = RobotDiagramBuilder()
        params["robot_model_instances"] = builder.parser().AddModelsFromString(
            urdf, "urdf"
        )
        params["model"] = builder.Build()
        checker = SceneGraphCollisionChecker(**params)

        iris_options = IrisOptions()

        set_builder = mut.IrisNpFromCliqueBuilder(
            checker=checker,
            options=iris_options,
            rank_tol_for_minimum_volume_circumscribed_ellipsoid=None,
        )

        self.assertTrue(
             set_builder.rank_tol_for_minimum_volume_circumscribed_ellipsoid()
             is None)

        set_builder.set_rank_tol_for_minimum_volume_circumscribed_ellipsoid(
            rank_tol_for_minimum_volume_circumscribed_ellipsoid=1e-6
        )

        self.assertEqual(
            set_builder.rank_tol_for_minimum_volume_circumscribed_ellipsoid(),
            1e-6
        )

        iris_options.iteration_limit = 1
        set_builder.set_options(iris_options)
        self.assertEqual(set_builder.options().iteration_limit, 1)

        points = np.array([
            [0,  0.25, 0],
            [0, -0.25, 0.25]
            ]
        )
        region = set_builder.BuildRegion(clique_points=points)
        self.assertGreaterEqual(region.A().shape[0], 3)
        self.assertEqual(region.A().shape[1], 2)

    def test_iris_zo_from_clique_builder(self):
        urdf = cross_cspace_urdf()
        params = dict(edge_step_size=0.125)
        builder = RobotDiagramBuilder()
        params["robot_model_instances"] = builder.parser().AddModelsFromString(
            urdf, "urdf"
        )
        params["model"] = builder.Build()
        checker = SceneGraphCollisionChecker(**params)

        iris_options = mut.IrisZoOptions()

        set_builder = mut.IrisZoFromCliqueBuilder(
            checker=checker,
            domain=None,
            options=iris_options,
            rank_tol_for_minimum_volume_circumscribed_ellipsoid=None,
        )

        self.assertTrue(
             set_builder.rank_tol_for_minimum_volume_circumscribed_ellipsoid()
             is None)

        set_builder.set_rank_tol_for_minimum_volume_circumscribed_ellipsoid(
            rank_tol_for_minimum_volume_circumscribed_ellipsoid=1e-6
        )

        self.assertEqual(
            set_builder.rank_tol_for_minimum_volume_circumscribed_ellipsoid(),
            1e-6
        )

        iris_options.max_iterations = 1
        set_builder.set_options(iris_options)
        self.assertEqual(set_builder.options().max_iterations, 1)

        points = np.array([
            [0,  0.25, 0],
            [0, -0.25, 0.25]
            ]
        )
        region = set_builder.BuildRegion(clique_points=points)
        self.assertGreaterEqual(region.A().shape[0], 3)
        self.assertEqual(region.A().shape[1], 2)

        set_builder.set_domain(region)
        numpy_compare.assert_equal(set_builder.domain().A(), region.A())

    def test_fast_clique_inflation_builder(self):
        urdf = cross_cspace_urdf()
        params = dict(edge_step_size=0.125)
        builder = RobotDiagramBuilder()
        params["robot_model_instances"] = builder.parser().AddModelsFromString(
            urdf, "urdf"
        )
        params["model"] = builder.Build()
        checker = SceneGraphCollisionChecker(**params)

        options = mut.FastCliqueInflationOptions()

        set_builder = mut.FastCliqueInflationBuilder(
            checker=checker,
            domain=None,
            options=options,
        )

        points = np.array([
            [0, 0.25, 0],
            [0, -0.25, 0.25]
        ]
        )
        region = set_builder.BuildRegion(clique_points=points)
        self.assertGreaterEqual(region.A().shape[0], 3)
        self.assertEqual(region.A().shape[1], 2)

        options.num_particles = 1
        set_builder.set_options(options)
        self.assertEqual(set_builder.options().num_particles, 1)

        set_builder.set_domain(region)
        numpy_compare.assert_equal(set_builder.domain().A(), region.A())

    def test_iris_in_configuration_space_from_clique_cover_template(self):
        # This test reimplements IrisFromCliqueCoverInConfigurationSpace using
        # Python and the template interface.

        class HPolyhedronPointSamplerPy(mut.PointSamplerBase):
            def __init__(self, domain):
                mut.PointSamplerBase.__init__(self)
                self.domain = domain
                self.last_point = self.domain.ChebyshevCenter()

            def DoSamplePoints(self, num_points, generator, parallelism):
                ret = np.zeros((self.domain.ambient_dimension(), num_points))
                for i in range(num_points):
                    ret[:, i] = self.domain.UniformSample(
                        generator, self.last_point, 10
                    )
                    self.last_point = ret[:, i]
                return ret

        class DefaultIrisSetBuilder(mut.RegionFromCliqueBase):
            def __init__(self,
                         options,
                         rank_tol_for_minimum_volume_circumscribed_ellipsoid,
                         checker):
                mut.RegionFromCliqueBase.__init__(self)
                self.options = options
                self.rank_tol_for_minimum_volume_circumscribed_ellipsoid = (
                    rank_tol_for_minimum_volume_circumscribed_ellipsoid)
                self.checker = checker

            def DoBuildRegion(self, clique_points):
                rank = self.rank_tol_for_minimum_volume_circumscribed_ellipsoid
                clique_ellipse = (
                    Hyperellipsoid.MinimumVolumeCircumscribedEllipsoid(
                        clique_points,
                        rank
                    )
                )
                if not (self.checker.
                        CheckConfigCollisionFree(clique_ellipse.center())):
                    self.options.iris_options.starting_ellipse = clique_ellipse
                else:
                    diff = (clique_points.T - clique_ellipse.center()).T
                    distances = np.linalg.norm(diff, axis=1)
                    nearest_point_col = np.argmin(distances)
                    center = clique_points[:, nearest_point_col]
                    self.options.iris_options.starting_ellipse = (
                        Hyperellipsoid(
                            clique_ellipse.A(),
                            center)
                    )
                checker.UpdatePositions(
                    self.options.iris_options.starting_ellipse.center()
                )
                return IrisInConfigurationSpace(checker.plant(),
                                                checker.plant_context(0),
                                                self.options.iris_options)

        class DefaultAdjacencyMatrixBuilder(mut.AdjacencyMatrixBuilderBase):
            def __init__(self, checker):
                mut.AdjacencyMatrixBuilderBase.__init__(self)
                self.checker = checker

            def DoBuildAdjacencyMatrix(self, points):
                visibility_graph = mut.VisibilityGraph(checker, points)
                return visibility_graph

        urdf = cross_cspace_urdf()
        params = dict(edge_step_size=0.125)
        builder = RobotDiagramBuilder()
        params["robot_model_instances"] = builder.parser().AddModelsFromString(
            urdf, "urdf"
        )
        params["model"] = builder.Build()
        checker = SceneGraphCollisionChecker(**params)

        options = mut.IrisFromCliqueCoverOptions()

        options.num_points_per_visibility_round = 100
        options.parallelism = Parallelism(1)

        options.coverage_termination_threshold = 0.9
        options.iteration_limit = 2

        generator = RandomGenerator(0)

        min_clique_cover_solver = mut.MinCliqueCoverSolverViaGreedy(
            mut.MaxCliqueSolverViaGreedy(), 5
        )
        point_sampler = HPolyhedronPointSamplerPy(
            HPolyhedron.MakeBox(checker.plant().GetPositionLowerLimits(),
                                checker.plant().GetPositionUpperLimits()))
        set_builder = DefaultIrisSetBuilder(options,
                                            1e-6,
                                            checker)

        adjacency_matrix_builder = DefaultAdjacencyMatrixBuilder(checker)
        print("Starting IrisInConfigurationSpaceFromCliqueCoverTemplate")
        sets = mut.IrisInConfigurationSpaceFromCliqueCoverTemplate(
            options=options, checker=checker, generator=generator,
            point_sampler=point_sampler,
            min_clique_cover_solver=min_clique_cover_solver,
            set_builder=set_builder, sets=[],
            adjacency_matrix_builder=adjacency_matrix_builder
        )
        self.assertGreaterEqual(len(sets), 2)

        # Now test PointsToCliqueCoverSets
        samples = point_sampler.SamplePoints(
            options.num_points_per_visibility_round, generator)
        sets = mut.PointsToCliqueCoverSets(
            points=samples,
            partition=True,
            checker=checker,
            min_clique_cover_solver=min_clique_cover_solver,
            set_builder=set_builder)
        self.assertGreaterEqual(len(sets), 2)
        sets = mut.PointsToCliqueCoverSets(
            points=samples,
            adjacency_matrix_builder=adjacency_matrix_builder,
            min_clique_cover_solver=min_clique_cover_solver,
            set_builder=set_builder,
            partition=True)
        self.assertGreaterEqual(len(sets), 2)
