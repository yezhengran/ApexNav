#include "exploration_manager/active_view.h"
#include <cassert>
using apexnav_planner::ActiveView;using Eigen::Vector2d;
int main() {
 auto clear=[](const Vector2d&){return true;};
 auto route=[](const Vector2d& p,std::vector<Vector2d>& path){path={p};return true;};
 auto small_room=[](const Vector2d& p){return p.norm()>1.6 && p.norm()<2.6;};
 std::vector<Vector2d> path;
 ActiveView old;assert(old.set({1,0,0,3.8,.8,3.4875,4.5}));
 assert(!old.plan({2.5,0},small_room,clear,route,path));
 ActiveView adaptive;assert(adaptive.set({1,0,0,3.8,.8,1.7,4.5,1}));
 assert(adaptive.plan({2.5,0},small_room,clear,route,path));assert(adaptive.goal.norm()>=2.0125 && adaptive.goal.norm()<2.6);
 assert(adaptive.framed({1.71,0}) && !adaptive.framed({1.69,0}));
 ActiveView blocked;assert(blocked.set({1,0,0,3.8,.8,1.7,4.5,1}));
 assert(!blocked.plan({2.5,0},[](const Vector2d&){return false;},clear,route,path));
 ActiveView long_route;assert(long_route.set({1,0,0,3.8,.8,1.7,4.5,1}));
 auto detour=[](const Vector2d& p,std::vector<Vector2d>& path){path={{20,0},p};return true;};
 assert(!long_route.plan({2.5,0},clear,clear,detour,path));
 ActiveView replan;assert(replan.set({1,0,0,3.8,.8,1.7,4.5,1}));
 assert(replan.plan({2.5,0},clear,clear,route,path));
 std::vector<Vector2d> invalid;
 auto dynamic_safe=[&invalid](const Vector2d& p){for(const auto& q:invalid)if((p-q).norm()<.05)return false;return true;};
 for(int count=1;count<=2;++count) {
  invalid.push_back(replan.goal);
  assert(replan.plan({2.5,0},dynamic_safe,clear,route,path));
  assert(replan.replans==count);
 }
 invalid.push_back(replan.goal);
 assert(!replan.plan({2.5,0},dynamic_safe,clear,route,path));assert(replan.replans==2);
 assert(replan.set({}));assert(replan.replans==0 && replan.id==0);
 ActiveView bad;assert(!bad.set({1,0,0,3.8,.8,1.7,4.5,0}));assert(!bad.set({1,0,0,3.8,.8,3.9,4.5,1}));
 assert(!ActiveView::safePath({{0,0},{1,0}},[](const Vector2d& p){return p.x()<.4 || p.x()>.6;}));
}
