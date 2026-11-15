#ifndef APEX_NAVIGATION_MEMORY_H
#define APEX_NAVIGATION_MEMORY_H
#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <deque>
#include <map>
#include <set>
#include <utility>
#include <vector>

namespace apexnav_planner {
// Episode-local decisions only. No ground truth and no mutation of occupancy/fusion.
class NavigationMemory {
public:
  using Point = Eigen::Vector2d;
  int selected = -1, selected_at = -1000;
  Point goal = Point::Zero();
  bool has_goal = false;
  std::map<int, int> cooldown_until;
  struct FailedEdge { Point start, direction; int expires; };
  struct Sample { int step; Point position, target; bool novel, blocked; };
  std::vector<FailedEdge> failed_edges;
  std::vector<Point> breadcrumbs;
  std::deque<Sample> window;
  std::set<std::pair<int,int>> visited;
  int last_step = -1, recovery_count = 0;

  bool cooling(int id, int step) const {
    auto it = cooldown_until.find(id);
    return id >= 0 && it != cooldown_until.end() && step < it->second;
  }
  static bool keepTarget(int age, double current_score, double challenger_score,
      bool current_confirmed, bool challenger_confirmed) {
    if (current_confirmed != challenger_confirmed) return current_confirmed;
    return age < 24 || challenger_score < current_score + 0.08;
  }
  void commit(int id, const Point& point, int step) {
    if (selected != id) selected_at = step;
    selected = id; goal = point; has_goal = true;
  }
  void abandon(int step) {
    if (selected >= 0) cooldown_until[selected] = step + 60;
    selected = -1; has_goal = false; window.clear();
  }
  void failEdge(const Point& start, const Point& direction, int step) {
    if (!start.allFinite() || !direction.allFinite() || direction.norm() < 1e-6) return;
    failed_edges.erase(std::remove_if(failed_edges.begin(), failed_edges.end(),
        [step](const FailedEdge& x){return x.expires <= step;}),failed_edges.end());
    failed_edges.push_back({start,direction.normalized(),step+40});
  }
  bool blockedEdge(const Point& start, const Point& move, int step) const {
    if (move.norm() < 1e-6) return false;
    for (const auto& edge : failed_edges)
      if (step < edge.expires && (start-edge.start).norm() < 0.22 &&
          move.normalized().dot(edge.direction) > 0.90) return true;
    return false;
  }
  // Observe once per real action, never once per planning callback or API handshake.
  int observe(int step, const Point& position, const Point& target, bool blocked) {
    if (step <= last_step) return 0;
    last_step = step;
    const auto cell = std::make_pair(static_cast<int>(std::floor(position.x()/0.25)),
                                    static_cast<int>(std::floor(position.y()/0.25)));
    const bool novel = visited.insert(cell).second;
    if (breadcrumbs.empty() || (position-breadcrumbs.back()).norm() >= 0.3)
      breadcrumbs.push_back(position);
    if (breadcrumbs.size()>160) breadcrumbs.erase(breadcrumbs.begin());
    window.push_back({step,position,target,novel,blocked});
    while (!window.empty() && step-window.front().step >= 24) window.pop_front();
    if (window.size()<24) return 0;
    int blocks=0,new_cells=0;double motion=0.;
    for (size_t i=0;i<window.size();++i) {
      blocks+=window[i].blocked;new_cells+=window[i].novel;
      if(i) motion+=(window[i].position-window[i-1].position).norm();
    }
    double early_best=1e9,recent_best=1e9;
    for(size_t i=0;i<6;++i) {
      early_best=std::min(early_best,(window[i].position-target).norm());
      recent_best=std::min(recent_best,(window[window.size()-1-i].position-target).norm());
    }
    const double progress=early_best-recent_best;
    if (blocks>=4 && progress<0.25 && new_cells<=2) return 1;
    if (motion>=1.0 && new_cells<=1 && progress<0.25) return 2;
    return 0;
  }
};
} // namespace apexnav_planner
#endif
