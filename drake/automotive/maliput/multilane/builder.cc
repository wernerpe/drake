#include "drake/automotive/maliput/multilane/builder.h"

#include <cmath>
#include <utility>

#include "drake/automotive/maliput/multilane/arc_road_curve.h"
#include "drake/automotive/maliput/multilane/branch_point.h"
#include "drake/automotive/maliput/multilane/cubic_polynomial.h"
#include "drake/automotive/maliput/multilane/junction.h"
#include "drake/automotive/maliput/multilane/line_road_curve.h"
#include "drake/automotive/maliput/multilane/road_geometry.h"
#include "drake/common/drake_assert.h"
#include "drake/common/text_logging.h"

namespace drake {
namespace maliput {
namespace multilane {

Builder::Builder(double lane_width, const api::HBounds& elevation_bounds,
                 double linear_tolerance, double angular_tolerance)
    : lane_width_(lane_width),
      elevation_bounds_(elevation_bounds),
      linear_tolerance_(linear_tolerance),
      angular_tolerance_(angular_tolerance) {
  DRAKE_DEMAND(lane_width_ >= 0.);
  DRAKE_DEMAND(linear_tolerance_ >= 0.);
  DRAKE_DEMAND(angular_tolerance_ >= 0.);
}

const Connection* Builder::Connect(const std::string& id, int num_lanes,
                                   double r0, double left_shoulder,
                                   double right_shoulder, const Endpoint& start,
                                   double length, const EndpointZ& z_end) {
  const Endpoint end(
      EndpointXy(start.xy().x() + (length * std::cos(start.xy().heading())),
                 start.xy().y() + (length * std::sin(start.xy().heading())),
                 start.xy().heading()),
      z_end);
  connections_.push_back(std::make_unique<Connection>(
      id, start, end, num_lanes, r0, left_shoulder, right_shoulder));
  return connections_.back().get();
}

const Connection* Builder::Connect(const std::string& id, int num_lanes,
                                   double r0, double left_shoulder,
                                   double right_shoulder, const Endpoint& start,
                                   const ArcOffset& arc,
                                   const EndpointZ& z_end) {
  const double alpha = start.xy().heading();
  const double theta0 = alpha - std::copysign(M_PI / 2., arc.d_theta());
  const double theta1 = theta0 + arc.d_theta();

  const double cx = start.xy().x() - (arc.radius() * std::cos(theta0));
  const double cy = start.xy().y() - (arc.radius() * std::sin(theta0));
  const Endpoint end(EndpointXy(cx + (arc.radius() * std::cos(theta1)),
                                cy + (arc.radius() * std::sin(theta1)),
                                alpha + arc.d_theta()),
                     z_end);

  connections_.push_back(std::make_unique<Connection>(
      id, start, end, num_lanes, r0, left_shoulder, right_shoulder, cx, cy,
      arc.radius(), arc.d_theta()));
  return connections_.back().get();
}

void Builder::SetDefaultBranch(const Connection* in, int in_lane_index,
                               const api::LaneEnd::Which in_end,
                               const Connection* out, int out_lane_index,
                               const api::LaneEnd::Which out_end) {
  default_branches_.push_back(
      {in, in_lane_index, in_end, out, out_lane_index, out_end});
}


Group* Builder::MakeGroup(const std::string& id) {
  groups_.push_back(std::make_unique<Group>(id));
  return groups_.back().get();
}


Group* Builder::MakeGroup(const std::string& id,
                          const std::vector<const Connection*>& connections) {
  groups_.push_back(std::make_unique<Group>(id, connections));
  return groups_.back().get();
}


namespace {
// Construct a CubicPolynomial such that:
//    f(0) = Y0 / dX           f'(0) = Ydot0
//    f(1) = (Y0 + dY) / dX    f'(1) = Ydot1
//
// This is equivalent to taking a cubic polynomial g such that:
//    g(0) = Y0          g'(0) = Ydot0
//    g(dX) = Y0 + dY    g'(1) = Ydot1
// and isotropically scaling it (scale both axes) by a factor of 1/dX
CubicPolynomial MakeCubic(double dX, double Y0, double dY,
                          double Ydot0, double Ydot1) {
  return CubicPolynomial(Y0 / dX,
                         Ydot0,
                         (3. * dY / dX) - (2. * Ydot0) - Ydot1,
                         Ydot0 + Ydot1 - (2. * dY / dX));
}

// Determine the heading (in xy-plane) along the centerline when
// travelling towards/into the lane, from the specified end.
double HeadingIntoLane(const api::Lane* const lane,
                       const api::LaneEnd::Which end) {
  switch (end) {
    case api::LaneEnd::kStart: {
      return lane->GetOrientation({0., 0., 0.}).yaw();
    }
    case api::LaneEnd::kFinish: {
      return lane->GetOrientation({lane->length(), 0., 0.}).yaw() + M_PI;
    }
    default: { DRAKE_ABORT(); }
  }
}

// Computes the location and heading of a `lane` at given `end` creating an
// Endpoint with that information. `road_curve` is used to compute z
// derivative with respect to p at the start or end of the `lane` respectively.
Endpoint ComputeEndpointForLane(const RoadCurve& road_curve, const Lane* lane,
                                const api::LaneEnd::Which end,
                                const EndpointZ& zpoint) {
  double p{}, length{};
  switch (end) {
    case api::LaneEnd::kStart:
      p = 0.;
      length = 0.;
      break;
    case api::LaneEnd::kFinish:
      p = 1.;
      length = lane->length();
      break;
    default: { DRAKE_ABORT(); }
  }
  const api::GeoPosition position = lane->ToGeoPosition({length, 0., 0.});
  const api::Rotation rotation = lane->GetOrientation({length, 0., 0.});
  const Vector3<double> w_prime =
      road_curve.W_prime_of_prh(p, lane->r0(), 0., road_curve.Rabg_of_p(p),
                                road_curve.elevation().f_dot_p(p));
  return Endpoint(
      EndpointXy(position.x(), position.y(), rotation.yaw()),
      EndpointZ(position.z(), w_prime.z(), zpoint.theta(), zpoint.theta_dot()));
}
}  // namespace



BranchPoint* Builder::FindOrCreateBranchPoint(
    const Endpoint& point,
    RoadGeometry* road_geometry,
    std::map<Endpoint, BranchPoint*, EndpointFuzzyOrder>* bp_map) const {
  auto ibp = bp_map->find(point);
  if (ibp != bp_map->end()) {
    return ibp->second;
  }
  // TODO(maddog@tri.global) Generate a more meaningful id (user-specified?)
  BranchPoint* bp = road_geometry->NewBranchPoint(
      api::BranchPointId{
        "bp:" + std::to_string(road_geometry->num_branch_points())});
  auto result = bp_map->emplace(point, bp);
  DRAKE_DEMAND(result.second);
  return bp;
}


void Builder::AttachBranchPoint(
    const Endpoint& point, Lane* const lane, const api::LaneEnd::Which end,
    RoadGeometry* road_geometry,
    std::map<Endpoint, BranchPoint*, EndpointFuzzyOrder>* bp_map) const {
  BranchPoint* bp = FindOrCreateBranchPoint(point, road_geometry, bp_map);
  // Tell the lane about its branch-point.
  switch (end) {
    case api::LaneEnd::kStart: {
      lane->SetStartBp(bp);
      break;
    }
    case api::LaneEnd::kFinish: {
      lane->SetEndBp(bp);
      break;
    }
    default: { DRAKE_ABORT(); }
  }
  // Now, tell the branch-point about the lane.
  //
  // Is this the first lane-end added to the branch-point?
  // If so, just stick it on A-Side.
  // (NB: We just test size of A-Side, since A-Side is always populated first.)
  if (bp->GetASide()->size() == 0) {
    bp->AddABranch({lane, end});
    return;
  }
  // Otherwise, assess if this new lane-end is parallel or anti-parallel to
  // the first lane-end.  Parallel: go to same, A-side; anti-parallel:
  // other, B-side.  Do this by examining the dot-product of the heading
  // vectors (rather than goofing around with cyclic angle arithmetic).
  const double new_h = HeadingIntoLane(lane, end);
  const api::LaneEnd old_le = bp->GetASide()->get(0);
  const double old_h = HeadingIntoLane(old_le.lane, old_le.end);
  if (((std::cos(new_h) * std::cos(old_h)) +
       (std::sin(new_h) * std::sin(old_h))) > 0.) {
    bp->AddABranch({lane, end});
  } else {
    bp->AddBBranch({lane, end});
  }
}

std::vector<Lane*> Builder::BuildConnection(
    const Connection* const conn, Junction* const junction,
    RoadGeometry* const road_geometry,
    std::map<Endpoint, BranchPoint*, EndpointFuzzyOrder>* const bp_map) const {
  std::unique_ptr<RoadCurve> p_road_curve;
  switch (conn->type()) {
    case Connection::kLine: {
      const V2 xy0(conn->start().xy().x(),
                   conn->start().xy().y());
      const V2 dxy(conn->end().xy().x() - xy0.x(),
                   conn->end().xy().y() - xy0.y());
      const CubicPolynomial elevation(MakeCubic(
          dxy.norm(),
          conn->start().z().z(),
          conn->end().z().z() - conn->start().z().z(),
          conn->start().z().z_dot(),
          conn->end().z().z_dot()));
      const CubicPolynomial superelevation(MakeCubic(
          dxy.norm(),
          conn->start().z().theta(),
          conn->end().z().theta() - conn->start().z().theta(),
          conn->start().z().theta_dot(),
          conn->end().z().theta_dot()));
      p_road_curve =
          std::make_unique<LineRoadCurve>(xy0, dxy, elevation, superelevation);
      break;
    }
    case Connection::kArc: {
      const V2 center(conn->cx(), conn->cy());
      const double radius = conn->radius();
      const double theta0 = std::atan2(conn->start().xy().y() - center.y(),
                                       conn->start().xy().x() - center.x());
      const double d_theta = conn->d_theta();
      const double arc_length = radius * std::abs(d_theta);
      const CubicPolynomial elevation(MakeCubic(
          arc_length,
          conn->start().z().z(),
          conn->end().z().z() - conn->start().z().z(),
          conn->start().z().z_dot(),
          conn->end().z().z_dot()));
      const CubicPolynomial superelevation(MakeCubic(
          arc_length,
          conn->start().z().theta(),
          conn->end().z().theta() - conn->start().z().theta(),
          conn->start().z().theta_dot(),
          conn->end().z().theta_dot()));
      p_road_curve = std::make_unique<ArcRoadCurve>(
          center, radius, theta0, d_theta, elevation, superelevation);
      break;
    }
    default: {
      DRAKE_ABORT();
    }
  }
  // Computes segment lateral extent.
  const double r_min = conn->r0() - lane_width_ / 2. - conn->right_shoulder();
  const double r_max =
      conn->r0() +
      lane_width_ * (static_cast<double>(conn->num_lanes() - 1) + 0.5) +
      conn->left_shoulder();
  const RoadCurve& road_curve = *p_road_curve;
  Segment* segment = junction->NewSegment(
      api::SegmentId{std::string("s:") + conn->id()}, std::move(p_road_curve),
      r_min, r_max, elevation_bounds_);
  std::vector<Lane*> lanes;
  for (int i = 0; i < conn->num_lanes(); i++) {
    Lane* lane =
        segment->NewLane(api::LaneId{std::string("l:") + conn->id() +
                                     std::string("_") + std::to_string(i)},
                         conn->r0() + lane_width_ * static_cast<double>(i),
                         {-lane_width_ / 2., lane_width_ / 2.});
    // Creates endpoints for the extents of the lane since they may not be
    // over the reference curve.
    const Endpoint start_endpoint = ComputeEndpointForLane(
        road_curve, lane, api::LaneEnd::kStart, conn->start().z());
    const Endpoint finish_endpoint = ComputeEndpointForLane(
        road_curve, lane, api::LaneEnd::kFinish, conn->end().z());
    AttachBranchPoint(start_endpoint, lane, api::LaneEnd::kStart, road_geometry,
                      bp_map);
    AttachBranchPoint(finish_endpoint, lane, api::LaneEnd::kFinish,
                      road_geometry, bp_map);
    lanes.push_back(lane);
  }

  return lanes;
}


std::unique_ptr<const api::RoadGeometry> Builder::Build(
    const api::RoadGeometryId& id) const {
  auto road_geometry = std::make_unique<RoadGeometry>(
      id, linear_tolerance_, angular_tolerance_);
  std::map<Endpoint, BranchPoint*, EndpointFuzzyOrder> bp_map(
      (EndpointFuzzyOrder(linear_tolerance_)));
  std::map<const Connection*, std::vector<Lane*>> lane_map;
  std::map<const Connection*, bool> connection_was_built;

  for (const std::unique_ptr<Connection>& connection : connections_) {
    connection_was_built.emplace(connection.get(), false);
  }

  for (const std::unique_ptr<Group>& group : groups_) {
    Junction* junction =
        road_geometry->NewJunction(
            api::JunctionId{std::string("j:") + group->id()});
    drake::log()->debug("junction: {}", junction->id().string());
    for (auto& connection : group->connections()) {
      drake::log()->debug("connection: {}", connection->id());
      DRAKE_DEMAND(!connection_was_built[connection]);
      lane_map[connection] = BuildConnection(
          connection, junction, road_geometry.get(), &bp_map);
      connection_was_built[connection] = true;
    }
  }

  for (const std::unique_ptr<Connection>& connection : connections_) {
    if (connection_was_built[connection.get()]) {
      continue;
    }
    Junction* junction =
        road_geometry->NewJunction(
            api::JunctionId{std::string("j:") + connection->id()});
    drake::log()->debug("junction: {}", junction->id().string());
    drake::log()->debug("connection: {}", connection->id());
    lane_map[connection.get()] =
        BuildConnection(connection.get(),
                        junction, road_geometry.get(), &bp_map);
  }

  for (const DefaultBranch& def : default_branches_) {
    Lane* in_lane = lane_map[def.in][def.in_lane_index];
    Lane* out_lane = lane_map[def.out][def.out_lane_index];
    DRAKE_DEMAND((def.in_end == api::LaneEnd::kStart) ||
                 (def.in_end == api::LaneEnd::kFinish));
    ((def.in_end == api::LaneEnd::kStart) ?
     in_lane->start_bp() : in_lane->end_bp())
        ->SetDefault({in_lane, def.in_end},
                     {out_lane, def.out_end});
  }

  // Make sure we didn't screw up!
  std::vector<std::string> failures = road_geometry->CheckInvariants();
  for (const auto& s : failures) {
    drake::log()->error(s);
  }
  DRAKE_DEMAND(failures.size() == 0);

  return std::move(road_geometry);
}


}  // namespace multilane
}  // namespace maliput
}  // namespace drake
