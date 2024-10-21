#include "drake/planning/iris/fast_clique_inflation_builder.h"

#include "drake/geometry/optimization/hpolyhedron.h"

namespace drake {
namespace planning {
namespace iris {
using geometry::optimization::HPolyhedron;

FastCliqueInflationBuilder::FastCliqueInflationBuilder(
    const CollisionChecker& checker,
    const std::optional<geometry::optimization::HPolyhedron> domain,
    const FastCliqueInflationOptions& options)
    : RegionFromCliqueBase(),
      checker_(checker.Clone()),
      domain_(domain.value_or(
          HPolyhedron::MakeBox(checker.plant().GetPositionLowerLimits(),
                               checker.plant().GetPositionUpperLimits()))),
      options_(options) {
  DRAKE_THROW_UNLESS(domain_.ambient_dimension() ==
                     checker_->plant().num_positions());
}

HPolyhedron FastCliqueInflationBuilder::DoBuildRegion(
    const Eigen::Ref<const Eigen::MatrixXd>& clique_points) {
  return FastCliqueInflation(*checker_, clique_points, domain_, options_);
}

void FastCliqueInflationBuilder::set_domain(
    const geometry::optimization::HPolyhedron& domain) {
  DRAKE_THROW_UNLESS(domain.ambient_dimension() ==
                     checker_->plant().num_positions());
  domain_ = domain;
}

}  // namespace iris
}  // namespace planning
}  // namespace drake
