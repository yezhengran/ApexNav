#include "exploration_manager/active_view.h"
#include <cassert>
#include <iostream>
using apexnav_planner::ActiveView;
using Eigen::Vector2d;
int main() {
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
  assert(invalid.set({}));assert(!invalid.crop_pose);
  std::cout<<"crop-pose arc, source radius, sector and safety checks passed\n";
}
