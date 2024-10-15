#include "drake/bindings/pydrake/documentation_pybind.h"
#include "drake/bindings/pydrake/planning/planning_py.h"
#include "drake/bindings/pydrake/pydrake_pybind.h"
#include "drake/planning/collision_checker.h"
#include "drake/planning/graph_algorithms/max_clique_solver_base.h"
#include "drake/planning/iris/adjacency_matrix_builder_base.h"
#include "drake/planning/iris/fast_clique_inflation.h"
#include "drake/planning/iris/fast_clique_inflation_builder.h"
#include "drake/planning/iris/hpolyhedron_point_sampler.h"
#include "drake/planning/iris/iris_from_clique_cover.h"
#include "drake/planning/iris/iris_from_clique_cover_options.h"
#include "drake/planning/iris/iris_from_clique_cover_template.h"
#include "drake/planning/iris/iris_from_clique_cover_v2.h"
#include "drake/planning/iris/iris_np_from_clique_builder.h"
#include "drake/planning/iris/iris_zo.h"
#include "drake/planning/iris/iris_zo_from_clique_builder.h"
#include "drake/planning/iris/point_sampler_base.h"
#include "drake/planning/iris/region_from_clique_base.h"

namespace drake {
namespace pydrake {
namespace internal {

void DefinePlanningIrisFromCliqueCover(py::module m) {
  {
    // NOLINTNEXTLINE(build/namespaces): Emulate placement in namespace.
    using namespace drake::planning;
    constexpr auto& doc = pydrake_doc.drake.planning;
    m.def(
        "IrisInConfigurationSpaceFromCliqueCover",
        [](const CollisionChecker& checker,
            const iris::IrisFromCliqueCoverOptions& options,
            RandomGenerator generator,
            std::vector<geometry::optimization::HPolyhedron> sets,
            const planning::graph_algorithms::MaxCliqueSolverBase*
                max_clique_solver) {
          IrisInConfigurationSpaceFromCliqueCover(
              checker, options, &generator, &sets, max_clique_solver);
          return sets;
        },
        py::arg("checker"), py::arg("options"), py::arg("generator"),
        py::arg("sets"), py::arg("max_clique_solver") = nullptr,
        py::call_guard<py::gil_scoped_release>(),
        doc.IrisInConfigurationSpaceFromCliqueCover.doc);
  }
  {
    // NOLINTNEXTLINE(build/namespaces): Emulate placement in namespace.
    using namespace drake::planning::iris;
    constexpr auto& doc = pydrake_doc.drake.planning.iris;
    {
      auto cls_doc = doc.IrisFromCliqueCoverOptions;
      py::class_<IrisFromCliqueCoverOptions>(
          m, "IrisFromCliqueCoverOptions", cls_doc.doc)
          .def(py::init<>())
          .def_readwrite("iris_options",
              &IrisFromCliqueCoverOptions::iris_options,
              cls_doc.iris_options.doc)
          .def_readwrite("coverage_termination_threshold",
              &IrisFromCliqueCoverOptions::coverage_termination_threshold,
              cls_doc.coverage_termination_threshold.doc)
          .def_readwrite("iteration_limit",
              &IrisFromCliqueCoverOptions::iteration_limit,
              cls_doc.iteration_limit.doc)
          .def_readwrite("num_points_per_coverage_check",
              &IrisFromCliqueCoverOptions::num_points_per_coverage_check,
              cls_doc.num_points_per_coverage_check.doc)
          .def_readwrite("parallelism",
              &IrisFromCliqueCoverOptions::parallelism, cls_doc.parallelism.doc)
          .def_readwrite("minimum_clique_size",
              &IrisFromCliqueCoverOptions::minimum_clique_size,
              cls_doc.minimum_clique_size.doc)
          .def_readwrite("num_points_per_visibility_round",
              &IrisFromCliqueCoverOptions::num_points_per_visibility_round,
              cls_doc.num_points_per_visibility_round.doc)
          .def_readwrite("rank_tol_for_minimum_volume_circumscribed_ellipsoid",
              &IrisFromCliqueCoverOptions::
                  rank_tol_for_minimum_volume_circumscribed_ellipsoid,
              cls_doc.rank_tol_for_minimum_volume_circumscribed_ellipsoid.doc)
          .def_readwrite("point_in_set_tol",
              &IrisFromCliqueCoverOptions::point_in_set_tol,
              cls_doc.point_in_set_tol.doc)
          .def_readwrite("confidence", &IrisFromCliqueCoverOptions::confidence,
              cls_doc.confidence.doc)
          .def_readwrite("sample_outside_of_sets",
              &IrisFromCliqueCoverOptions::sample_outside_of_sets,
              cls_doc.sample_outside_of_sets.doc)
          .def_readwrite("partition", &IrisFromCliqueCoverOptions::partition,
              cls_doc.partition.doc)
          .def_readwrite("sampling_batch_size",
              &IrisFromCliqueCoverOptions::sampling_batch_size,
              cls_doc.sampling_batch_size.doc);
    }
    {
      // Trampoline class for AdjacencyMatrixBuilderBase
      class PyAdjacencyMatrixBuilderBase
          : public py::wrapper<AdjacencyMatrixBuilderBase> {
       public:
        Eigen::SparseMatrix<bool> DoBuildAdjacencyMatrix(
            const Eigen::Ref<const Eigen::MatrixXd>& points) const override {
          PYBIND11_OVERLOAD_PURE(Eigen::SparseMatrix<bool>,
              AdjacencyMatrixBuilderBase, DoBuildAdjacencyMatrix, points);
        }
      };
      const auto& cls_doc = doc.AdjacencyMatrixBuilderBase;
      py::class_<AdjacencyMatrixBuilderBase, PyAdjacencyMatrixBuilderBase>(
          m, "AdjacencyMatrixBuilderBase")
          .def(py::init<>(), cls_doc.ctor.doc)
          .def("BuildAdjacencyMatrix",
              &AdjacencyMatrixBuilderBase::BuildAdjacencyMatrix,
              py::arg("points"), cls_doc.BuildAdjacencyMatrix.doc);
    }
    {
      // Trampoline class for PointSamplerBase
      class PyPointSamplerBase : public py::wrapper<PointSamplerBase> {
       public:
        Eigen::MatrixXd DoSamplePoints(int num_points,
            RandomGenerator* generator, Parallelism parallelism) override {
          PYBIND11_OVERLOAD_PURE(Eigen::MatrixXd, PointSamplerBase,
              DoSamplePoints, num_points, generator, parallelism);
        }
      };
      const auto& cls_doc = doc.PointSamplerBase;
      py::class_<PointSamplerBase, PyPointSamplerBase>(m, "PointSamplerBase")
          .def(py::init<>(), cls_doc.ctor.doc)
          .def("SamplePoints", &PointSamplerBase::SamplePoints,
              py::arg("num_points"), py::arg("generator"),
              py::arg("parallelism") = Parallelism::Max(),
              py::call_guard<py::gil_scoped_release>(),
              cls_doc.SamplePoints.doc);
    }
    {
      const auto& cls_doc = doc.HPolyhedronPointSampler;
      py::class_<HPolyhedronPointSampler, PointSamplerBase>(
          m, "HPolyhedronPointSampler")
          .def(py::init<const geometry::optimization::HPolyhedron&, int,
                   const std::optional<Eigen::VectorXd>>(),
              py::arg("domain"), py::arg("mixing_steps") = 10,
              py::arg("first_point") = std::nullopt, cls_doc.ctor.doc)
          .def("mixing_steps", &HPolyhedronPointSampler::mixing_steps,
              cls_doc.mixing_steps.doc)
          .def("set_mixing_steps", &HPolyhedronPointSampler::set_mixing_steps,
              py::arg("mixing_steps"), cls_doc.set_mixing_steps.doc)
          .def("last_point", &HPolyhedronPointSampler::last_point,
              cls_doc.last_point.doc);
    }
    {
      // Trampoline class for RegionFromCliqueBase
      class PyRegionFromCliqueBase : public RegionFromCliqueBase {
       public:
        geometry::optimization::HPolyhedron DoBuildRegion(
            const Eigen::Ref<const Eigen::MatrixXd>& clique_points) override {
          PYBIND11_OVERLOAD_PURE(geometry::optimization::HPolyhedron,
              RegionFromCliqueBase, DoBuildRegion, clique_points);
        }
      };
      const auto& cls_doc = doc.RegionFromCliqueBase;
      py::class_<RegionFromCliqueBase, PyRegionFromCliqueBase>(
          m, "RegionFromCliqueBase")
          .def(py::init<>(), cls_doc.ctor.doc)
          .def("BuildRegion", &RegionFromCliqueBase::BuildRegion,
              py::arg("clique_points"), cls_doc.BuildRegion.doc);
    }
    {
      const auto& cls_doc = doc.IrisNpFromCliqueBuilder;
      py::class_<IrisNpFromCliqueBuilder, RegionFromCliqueBase>(
          m, "IrisNpFromCliqueBuilder")
          .def(py::init<const planning::CollisionChecker&,
                   const geometry::optimization::IrisOptions&,
                   std::optional<double>>(),
              py::arg("checker"), py::arg("options"),
              py::arg("rank_tol_for_minimum_volume_circumscribed_ellipsoid") =
                  std::nullopt,
              cls_doc.ctor.doc)
          .def(
              "options", &IrisNpFromCliqueBuilder::options, cls_doc.options.doc)
          .def("set_options", &IrisNpFromCliqueBuilder::set_options,
              cls_doc.set_options.doc)
          .def("rank_tol_for_minimum_volume_circumscribed_ellipsoid",
              &IrisNpFromCliqueBuilder::
                  rank_tol_for_minimum_volume_circumscribed_ellipsoid,
              cls_doc.rank_tol_for_minimum_volume_circumscribed_ellipsoid.doc)
          .def("set_rank_tol_for_minimum_volume_circumscribed_ellipsoid",
              &IrisNpFromCliqueBuilder::
                  set_rank_tol_for_minimum_volume_circumscribed_ellipsoid,
              py::arg("rank_tol_for_minimum_volume_circumscribed_ellipsoid"),
              cls_doc.set_rank_tol_for_minimum_volume_circumscribed_ellipsoid
                  .doc);
    }
    {
      const auto& cls_doc = doc.IrisZoFromCliqueBuilder;
      py::class_<IrisZoFromCliqueBuilder, RegionFromCliqueBase>(
          m, "IrisZoFromCliqueBuilder")
          .def(py::init<const planning::CollisionChecker&,
                   const std::optional<geometry::optimization::HPolyhedron>,
                   const planning::IrisZoOptions&, std::optional<double>>(),
              py::arg("checker"), py::arg("domain") = std::nullopt,
              py::arg("options") = planning::IrisZoOptions(),
              py::arg("rank_tol_for_minimum_volume_circumscribed_ellipsoid") =
                  std::nullopt,
              cls_doc.ctor.doc)
          .def(
              "options", &IrisZoFromCliqueBuilder::options, cls_doc.options.doc)
          .def("set_options", &IrisZoFromCliqueBuilder::set_options,
              cls_doc.set_options.doc)
          .def("domain", &IrisZoFromCliqueBuilder::domain, cls_doc.domain.doc)
          .def("set_domain", &IrisZoFromCliqueBuilder::set_domain,
              py::arg("domain"), cls_doc.set_domain.doc)
          .def("rank_tol_for_minimum_volume_circumscribed_ellipsoid",
              &IrisZoFromCliqueBuilder::
                  rank_tol_for_minimum_volume_circumscribed_ellipsoid,
              cls_doc.rank_tol_for_minimum_volume_circumscribed_ellipsoid.doc)
          .def("set_rank_tol_for_minimum_volume_circumscribed_ellipsoid",
              &IrisZoFromCliqueBuilder::
                  set_rank_tol_for_minimum_volume_circumscribed_ellipsoid,
              py::arg("rank_tol_for_minimum_volume_circumscribed_ellipsoid"),
              cls_doc.set_rank_tol_for_minimum_volume_circumscribed_ellipsoid
                  .doc);
    }
    {
      const auto& cls_doc = doc.FastCliqueInflationBuilder;
      py::class_<FastCliqueInflationBuilder, RegionFromCliqueBase>(
          m, "FastCliqueInflationBuilder")
          .def(py::init<const planning::CollisionChecker&,
                   const std::optional<geometry::optimization::HPolyhedron>,
                   const planning::FastCliqueInflationOptions&>(),
              py::arg("checker"), py::arg("domain") = std::nullopt,
              py::arg("options") = planning::FastCliqueInflationOptions(),
              cls_doc.ctor.doc)
          .def("options", &FastCliqueInflationBuilder::options,
              cls_doc.options.doc)
          .def("set_options", &FastCliqueInflationBuilder::set_options,
              cls_doc.set_options.doc)
          .def(
              "domain", &FastCliqueInflationBuilder::domain, cls_doc.domain.doc)
          .def("set_domain", &FastCliqueInflationBuilder::set_domain,
              py::arg("domain"), cls_doc.set_domain.doc);
    }
    m.def(
         "IrisInConfigurationSpaceFromCliqueCoverTemplate",
         [](const IrisFromCliqueCoverOptions& options,
             const planning::CollisionChecker& checker,
             RandomGenerator generator, PointSamplerBase* point_sampler,
             planning::graph_algorithms::MinCliqueCoverSolverBase*
                 min_clique_cover_solver,
             RegionFromCliqueBase* set_builder,
             std::vector<geometry::optimization::HPolyhedron> sets,
             AdjacencyMatrixBuilderBase* adjacency_matrix_builder) {
           IrisInConfigurationSpaceFromCliqueCoverTemplate(options, checker,
               &generator, point_sampler, min_clique_cover_solver, set_builder,
               &sets, adjacency_matrix_builder);
           return sets;
         },
         py::arg("options"), py::arg("checker"), py::arg("generator"),
         py::arg("point_sampler"), py::arg("min_clique_cover_solver"),
         py::arg("set_builder"), py::arg("sets"),
         py::arg("adjacency_matrix_builder") = nullptr,
         py::call_guard<py::gil_scoped_release>(),
         doc.IrisInConfigurationSpaceFromCliqueCoverTemplate.doc)
        .def("PointsToCliqueCoverSets",
            py::overload_cast<const Eigen::Ref<const Eigen::MatrixXd>&,
                const planning::CollisionChecker&,
                planning::graph_algorithms::MinCliqueCoverSolverBase*,
                RegionFromCliqueBase*, Parallelism, bool,
                std::shared_ptr<geometry::Meshcat>>(&PointsToCliqueCoverSets),
            py::arg("points"), py::arg("checker"),
            py::arg("min_clique_cover_solver"), py::arg("set_builder"),
            py::arg("parallelism") = Parallelism::Max(),
            py::arg("partition") = true, py::arg("meshcat") = nullptr,
            py::call_guard<py::gil_scoped_release>())
        .def("PointsToCliqueCoverSets",
            py::overload_cast<const Eigen::Ref<const Eigen::MatrixXd>&,
                AdjacencyMatrixBuilderBase*,
                planning::graph_algorithms::MinCliqueCoverSolverBase*,
                RegionFromCliqueBase*, bool,
                std::shared_ptr<geometry::Meshcat>>(&PointsToCliqueCoverSets),
            py::arg("points"), py::arg("adjacency_matrix_builder"),
            py::arg("min_clique_cover_solver"), py::arg("set_builder"),
            py::arg("partition") = true, py::arg("meshcat") = nullptr,
            py::call_guard<py::gil_scoped_release>())
        .def(
            "IrisInConfigurationSpaceFromCliqueCoverV2",
            [](const planning::CollisionChecker& checker,
                const IrisFromCliqueCoverOptions& options,
                RandomGenerator generator,
                std::vector<geometry::optimization::HPolyhedron> sets,
                const planning::graph_algorithms::MaxCliqueSolverBase*
                    max_clique_solver) {
              IrisInConfigurationSpaceFromCliqueCoverV2(
                  checker, options, &generator, &sets, max_clique_solver);
              return sets;
            },
            py::arg("checker"), py::arg("options"), py::arg("generator"),
            py::arg("sets"), py::arg("max_clique_solver") = nullptr,
            py::call_guard<py::gil_scoped_release>(),
            doc.IrisInConfigurationSpaceFromCliqueCoverTemplate.doc);
  }
}  // DefinePlanningIrisFromCliqueCover

}  // namespace internal
}  // namespace pydrake
}  // namespace drake
