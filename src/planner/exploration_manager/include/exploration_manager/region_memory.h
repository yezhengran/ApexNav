#ifndef APEX_REGION_MEMORY_H
#define APEX_REGION_MEMORY_H
#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <utility>
#include <vector>
namespace apexnav_planner {
// Coarse regions are visitation bins, not assumed rooms or traversable shortcuts.
class RegionMemory {
public:
 using Point=Eigen::Vector2d;using Key=std::pair<int,int>;
 static Key key(const Point& p) {return {int(std::floor(p.x()/2.5)),int(std::floor(p.y()/2.5))};}
 int last_step=-1,hold_until=-1,reconsider_after=-1,scans=0;bool valid=false;
 Point last=Point::Zero(),goal=Point::Zero(),frontier=Point::Zero();Key selected={0,0};
 std::map<Key,int> visits;std::set<std::pair<Key,Key>> traversed_edges;
 struct Failed {Point point;int until;};std::vector<Failed> failed;
 void observe(int step,const Point& p) {
  if(step<=last_step)return;
  if(last_step>=0 && (p-last).norm()>.01 && (p-last).norm()<.6 && key(p)!=key(last))
   traversed_edges.insert({key(last),key(p)});
  ++visits[key(p)];last=p;last_step=step;
 }
 void block(const Point& p,int step) {failed.push_back({p,step+80});valid=false;}
 bool cooling(const Point& p,int step) const {
  for(const auto& f:failed)if(step<f.until && (p-f.point).norm()<.75)return true;
  return false;
 }
 static double utility(double information,double semantic,double path_length,int visits) {
  return 1.5*std::clamp(information,0.,1.)+.8*std::clamp(semantic,0.,1.)
    +.6/(1.+visits/12.)-.25*std::log1p(std::max(0.,path_length));
 }
 static bool keep(int step,int until,double current,double challenger) {
  return step<until || challenger<current+.20;
 }
};
}
#endif
