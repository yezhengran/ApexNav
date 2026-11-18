#include "exploration_manager/active_view.h"
#include <cassert>
#include <limits>
#include <iostream>

using apexnav_planner::ActiveView;
using Eigen::Vector2d;
void test_active_view() {
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

void test_annular_view() {
  auto clear=[](const Vector2d&){return true;};
  auto safe=[](const Vector2d& p){return p.norm()>1.2;};
  Vector2d position(4.,0.);
  auto route=[&](const Vector2d& p,std::vector<Vector2d>& path){path={position,p};return true;};
  ActiveView view;assert(view.set({1,0,0,2.,.8,1.65,2.7,1.}));
  assert(view.annular && !view.framed({2.8,0}));
  std::vector<Vector2d> path;
  assert(view.plan(position,safe,clear,route,path));
  assert(view.samples.size()>=2 && view.samples.size()<=3);
  for(const auto& p:view.samples)assert(safe(p) && view.framed(p));
  while(true) {
    position=view.goal;
    if(!view.advanceSample())break;
    assert(view.plan(position,safe,clear,route,path));
    assert(ActiveView::safePath(path,[&](const Vector2d& p){return safe(p)&&view.framed(p);}));
  }
  ActiveView none;assert(none.set({2,0,0,2,.8,1.65,2.7,1}));
  assert(!none.plan(position,[](const Vector2d&){return false;},clear,route,path));
  ActiveView slit;assert(slit.set({3,0,0,2,.8,1.65,2.7,1}));
  // A solitary endpoint is not a traversable observation arc.
  assert(!slit.plan(position,[](const Vector2d& p){return p.x()>1.65&&std::abs(p.y())<.01;},clear,route,path));
  assert(!slit.set({4,0,0,2,.8,1.65,2.7,2}));
  assert(view.set({}));assert(!view.annular && view.samples.empty());
}

void test_crop_pose_view() {
  Vector2d position(1.8,0.);
  auto clear=[](const Vector2d&){return true;};
  auto room=[](const Vector2d& p){return p.x()>1.1 && p.x()<2.2 && std::abs(p.y())<1.1;};
  auto route=[&](const Vector2d& p,std::vector<Vector2d>& path){path={position,p};return true;};
  std::vector<Vector2d> path;
  ActiveView view;assert(view.set({1,0,0,3.1,.8,2.8,3.8,1}));
  assert(!view.plan(position,room,clear,route,path));assert(view.search.generated==72);
  assert(view.set({}));
  assert(view.set({1,0,0,1.8,.8,1.55,2.05,1,1.8,0,2}));
  assert(view.crop_pose && view.annular);assert(view.plan(position,room,clear,route,path));
  assert(view.search.generated==5 && view.samples.size()>=2 && view.samples.size()<=3);
  for (auto p:view.samples) {assert(std::abs(p.norm()-1.8)<1e-8);assert(std::abs(std::atan2(p.y(),p.x()))<=M_PI/6+1e-8);}
  while(view.advanceSample()) {position=view.goal;assert(view.plan(position,room,clear,route,path));}
  ActiveView invalid;
  assert(!invalid.set({2,0,0,1.8,.8,1.55,2.05,1,1.7,0,2})); // Wrong recorded radius.
  assert(!invalid.set({2,0,0,1.8,.8,1.50,2.05,1,1.8,0,2})); // Changed band.
  assert(!invalid.set({2,0,0,1.8,.8,1.55,2.05,1,1.8,0,1})); // Wrong wire version.
  assert(invalid.set({2,0,0,1.8,.8,1.55,2.05,1,1.8,0,2}));
  assert(!invalid.plan(position,room,[](const Vector2d&){return false;},route,path));
  assert(invalid.search.visible==0); // Visibility never bypassed.
  assert(!invalid.plan(position,room,clear,[](const Vector2d&,std::vector<Vector2d>&){return false;},path));
  assert(invalid.search.access==0);
  auto remote=[](const Vector2d& p){return p.x()<-.9;};
  assert(!invalid.plan(position,remote,clear,route,path)); // Do not search behind the object's observed side.
  auto isolated=[](const Vector2d& p){return p.x()>1.5 && std::abs(p.y())<.01;};
  assert(!invalid.plan(position,isolated,clear,route,path)); // Still requires a real short arc.
  assert(invalid.set({}));position=Vector2d(1,0);
  assert(invalid.set({3,0,0,1.,.8,.75,1.25,1,1,0,2}));
  assert(invalid.plan(position,clear,clear,route,path));
  for (size_t i=1;i<invalid.samples.size();++i)
    assert((invalid.samples[i]-invalid.samples[i-1]).norm()>.4); // Nonoverlapping arrival discs.
  assert(invalid.set({}));position=Vector2d(.6,0);
  assert(invalid.set({3,0,0,.6,.8,.6,.85,1,.6,0,2}));
  assert(!invalid.plan(position,clear,clear,route,path)); // Too little translation within a short arc.
  // Same isolated position remains forbidden in v2; v3 permits one sample,
  // with the identical free-space/visibility/path checks.
  assert(invalid.set({}));position=Vector2d(1.8,0);
  assert(invalid.set({4,0,0,1.8,.8,1.55,2.05,1,1.8,0,3}));
  assert(invalid.plan(position,isolated,clear,route,path));
  assert(invalid.samples.size()==1 && !invalid.advanceSample());
  assert(!invalid.plan(position,isolated,[](const Vector2d&){return false;},route,path));
  assert(invalid.set({}));assert(!invalid.crop_pose && !invalid.single_view);
  std::cout<<"crop-pose arc, source radius, sector and safety checks passed\n";
}

int main() { test_active_view(); test_annular_view(); test_crop_pose_view(); }
