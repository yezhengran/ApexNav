#include <exploration_manager/frontier_view.h>
#include <cassert>
using apexnav_planner::FrontierView;
using Eigen::Vector2d;
int main() {
  auto address=[](const Vector2d& p){return int(std::floor(p.x()*10))+1000*int(std::floor(p.y()*10));};
  auto north=[](const Vector2d& p){return p.y()>1.?1:0;};
  auto occluded=[](const Vector2d& p){return p.y()>.5?-1:0;};
  auto score=[&](double yaw){return FrontierView::gain({0,0},yaw,.4,.4,3.,.1,north,address);};
  assert(score(M_PI_2)>score(0));
  assert(FrontierView::gain({0,0},M_PI_2,.4,.4,3.,.1,occluded,address)==0);
  FrontierView look;
  assert(look.next({0,0},0,false,score)==0);
  double yaw=0.;int turns=0;
  for(int i=0;i<20;++i){int action=look.next({0,0},yaw,true,score);if(action){++turns;yaw+=action*M_PI/6.;}}
  assert(turns>0 && turns<=3);
  assert(look.next({0,0},yaw,true,score)==0); // No repeated scan at the same place.
  FrontierView empty;
  assert(empty.next({0,0},0,true,[](double){return 0;})==0);
  FrontierView changed;
  assert(changed.next({0,0},0,true,score)==1);
  assert(changed.next({0,0},M_PI/6.,true,[](double){return 0;})==0);
}
