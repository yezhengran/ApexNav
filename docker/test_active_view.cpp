#include "exploration_manager/active_view.h"
#include <cassert>
#include <limits>
using apexnav_planner::ActiveView;
using Eigen::Vector2d;
int main() {
  auto yes=[](const Vector2d&) { return true; };
  auto route=[](const Vector2d& p, std::vector<Vector2d>& path) { path={p}; return true; };
  std::vector<Vector2d> path;
  ActiveView legacy;
  assert(legacy.set({1,0,0,1,1}));
  assert(!legacy.framing && legacy.framed({.1,0}));
  assert(legacy.plan({4,0},yes,yes,route,path));
  assert(std::abs(legacy.goal.norm()-1)<1e-8);
  ActiveView turn;assert(turn.set({1,0,0,1,1,1}) && turn.turn_first);
  ActiveView next;assert(next.set({2,0,0,2,.8,1.6875,2.7}));
  assert(!next.framed({1.68,0}) && next.framed({1.69,0}));
  auto outer=[](const Vector2d& p) { return p.norm()>2.30 && p.norm()<2.40; };
  assert(next.plan({4,0},outer,yes,route,path));
  assert(std::abs(next.goal.norm()-2.35)<1e-8); // Alternate farther ring.
  assert(next.plan({4,0},outer,yes,route,path)); // Frozen safe point.
  ActiveView bad;
  assert(!bad.set({1,0,0,4.6,1,3,4.7}));
  assert(!bad.set({1,0,0,2,1,2.1,2.7}));
  assert(!bad.set({1,0,0,2,1,1.5,std::numeric_limits<double>::quiet_NaN()}));
  assert(!bad.set({1,0,0,4,1})); // Legacy bound unchanged.
  assert(bad.set({1,0,0,2,1,1.5,2.7}));
  assert(!bad.plan({4,0},[](const Vector2d&) { return false; },yes,route,path));
  assert(!ActiveView::safePath({{0,0},{1,0}},[](const Vector2d& p) { return p.x()<.4 || p.x()>.6; }));
}
