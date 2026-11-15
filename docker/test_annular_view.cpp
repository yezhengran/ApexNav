#include "exploration_manager/active_view.h"
#include <cassert>
using apexnav_planner::ActiveView;
using Eigen::Vector2d;
int main() {
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
