#include <exploration_manager/frontier_view.h>
#include <cassert>
#include <iostream>
using apexnav_planner::FrontierView;
using Eigen::Vector2d;
using Points=FrontierView::Points;
constexpr double pi=FrontierView::pi;
double error(double a,double b) {return std::abs(std::atan2(std::sin(a-b),std::cos(a-b)));}
Points arc(double first,double last) {
  Points result;
  for(double a=first;a<=last;a+=2.) result.push_back({std::cos(a*pi/180.),std::sin(a*pi/180.)});
  return result;
}
int main() {
  auto radial=[](const Vector2d& p){return p.norm()>=.999 ? 1 : 0;};
  auto cells=arc(90,150), small=arc(-8,8);
  cells.insert(cells.end(),small.begin(),small.end());
  auto h=FrontierView::heading({0,0},cells,.05,3.,radial);
  assert(h.valid && !h.normal && error(h.yaw,120*pi/180.)<.02); // Widest contiguous span, not all-cell centroid.
  h=FrontierView::heading({0,0},arc(160,200),.05,3.,radial);
  assert(h.valid && error(h.yaw,pi)<.02); // Wrap across -pi/pi.
  auto wall=[&](const Vector2d& p){return p.y()>.4 && p.y()<.6 ? -1 : radial(p);};
  h=FrontierView::heading({0,0},cells,.05,3.,wall);
  assert(h.valid && error(h.yaw,0)<.02); // Obstructed wide span cannot win.
  assert(!FrontierView::heading({0,0},arc(90,150),.05,3.,wall).valid);
  assert(!FrontierView::heading({0,0},cells,.05,.3,radial).valid);
  assert(!FrontierView::heading({0,0},{},.05,3.,radial).valid);
  assert(!FrontierView::heading({0,0},cells,0.,3.,radial).valid);
  assert(!FrontierView::heading({0,0},arc(0,358),.05,3.,radial).valid); // No unique direction in a symmetric ring.
  auto north=[](const Vector2d& p){return p.y()>=1.-1e-8?1:0;};
  Points line;for(int x=-10;x<=10;++x)line.push_back({x*.05,1.});
  h=FrontierView::heading({0,1},line,.05,3.,north);
  assert(h.valid && h.normal && error(h.yaw,pi/2)<.01); // Mean lies on boundary: use unknown-side normal.
  auto known=[](const Vector2d&){return 0;};
  assert(!FrontierView::heading({0,1},line,.05,3.,known).valid);
  FrontierView look;
  assert(look.bind({0,0},{0,1},line,.05,3.,north));
  assert(look.next({0,0},0,.05,3.,north)==0); // No en-route turns.
  assert(!look.bind({0,.6},{.01,1},line,.05,3.,north)); // Stored orientation survives small goal jitter.
  int turns=0;double yaw=0.;
  for(int i=0;i<20;++i){int a=look.next({0,.6},yaw,.05,3.,north);turns+=std::abs(a);yaw+=a*pi/6.;}
  assert(turns==3 && error(yaw,pi/2)<.01);
  assert(!look.bind({0,.6},{.3,1},line,.05,3.,north)); // No repeated nearby scan.
  assert(look.bind({0,-1},{0,1},line,.05,3.,north)); // A later separate visit may bind again.
  look.cancel();assert(look.next({0,.6},0,.05,3.,north)==0); // Object/inspection preemption.
  FrontierView stale;
  stale.bind({0,0},{0,1},line,.05,3.,north);
  assert(stale.next({0,.6},0,.05,3.,known)==0); // Retired boundary must cancel.
  assert(!stale.bind({0,.6},{0,1},line,.05,3.,north));
  FrontierView retired_span;
  retired_span.bind({-2,0},{0,0},cells,.05,3.,radial);
  auto east_only=[&](const Vector2d& p){return p.x()>.5 ? radial(p) : 0;};
  // Another part of the same cluster may remain unknown; it must not keep a
  // now-observed stored direction alive or silently redirect the arrival turn.
  assert(retired_span.next({0,0},0,.05,3.,east_only)==0);
  FrontierView bounded;
  bounded.bind({0,0},{0,1},line,.05,3.,north);
  turns=0;
  for(int i=0;i<20;++i)turns+=std::abs(bounded.next({0,.6},-pi/2,.05,3.,north));
  assert(turns==FrontierView::max_turns); // Broken yaw feedback still has a hard bound.
  FrontierView aligned;
  aligned.bind({0,0},{0,1},line,.05,3.,north);
  assert(aligned.next({0,.6},pi/2,.05,3.,north)==0);
  std::cout << "frontier geometry and lifecycle checks passed\n";
}
