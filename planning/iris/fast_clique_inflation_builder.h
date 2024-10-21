#pragma once

#include <memory>
#include <optional>

#include "drake/geometry/optimization/hpolyhedron.h"
#include "drake/planning/collision_checker.h"
#include "drake/planning/iris/fast_clique_inflation.h"
#include "drake/planning/iris/region_from_clique_base.h"

namespace drake {
namespace planning {
namespace iris {

/**
 * Given a clique, this class constructs a region by calling the
 * FastCliqueInflation subroutine. For use with IrisFromCliqueCoverTemplate.
 *
 * TODO(Alexandre.Amice) currently I clone the collision checker. This is
 * wasteful so consider a better design when going to master.
 */
class FastCliqueInflationBuilder final : public RegionFromCliqueBase {
 public:
  FastCliqueInflationBuilder(
      const CollisionChecker& checker,
      const std::optional<geometry::optimization::HPolyhedron> domain =
          std::nullopt,
      const FastCliqueInflationOptions& options = FastCliqueInflationOptions());

  /**
   * The options used during FastCliqueInflation.
   * @return
   */
  const FastCliqueInflationOptions& options() const { return options_; }

  /**
   * Sets the options used during FastCliqueInflation.
   * @return
   */
  void set_options(const FastCliqueInflationOptions& iris_options) {
    options_ = iris_options;
  }

  /**
   * The domain used during FastCliqueInflation.
   * @return
   */
  const geometry::optimization::HPolyhedron& domain() const { return domain_; }

  /**
   * Sets the domain used during FastCliqueInflation.
   * @return
   */
  void set_domain(const geometry::optimization::HPolyhedron& domain);

 private:
  geometry::optimization::HPolyhedron DoBuildRegion(
      const Eigen::Ref<const Eigen::MatrixXd>& clique_points) final;

  const std::unique_ptr<CollisionChecker> checker_;
  geometry::optimization::HPolyhedron domain_;
  FastCliqueInflationOptions options_;
};

}  // namespace iris
}  // namespace planning
}  // namespace drake
