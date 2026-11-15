#ifndef APEX_ACTIVE_VIEW_H
#define APEX_ACTIVE_VIEW_H

#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <functional>
#include <tuple>
#include <vector>

namespace apexnav_planner {

// A deliberately small adapter: map queries and A* stay owned by ApexNav.
class ActiveView {
public:
  int id = 0;
  Eigen::Vector2d center = Eigen::Vector2d::Zero();
  double radius = 0.0, extent = 0.0;
  double minimum_radius = 0.0, max_radius = 0.0;
  bool framing = false, adaptive = false;
  int replans = 0;
  bool locked = false;
  bool turn_first = false, oriented = false;
  int orientation_steps = 0;
  Eigen::Vector2d goal = Eigen::Vector2d::Zero();
  using Safe = std::function<bool(const Eigen::Vector2d&)>;
  using Route = std::function<bool(const Eigen::Vector2d&, std::vector<Eigen::Vector2d>&)>;

  bool set(const std::vector<double>& data)
  {
    if (data.empty()) { *this = ActiveView(); return true; }
    if (data.size() != 5 && data.size() != 6 && data.size() != 7 && data.size() != 8) return false;
    for (double x : data) if (!std::isfinite(x)) return false;
    if (data[0] < 1 || data[0] > 1000000 || std::floor(data[0]) != data[0] ||
        data[3] < 0.6 || data[3] > (data.size() >= 7 ? 4.5 : 3.5) || data[4] < 0.15 || data[4] > 3.0)
      return false;
    if (data.size() == 6 && data[5] != 1.0) return false;
    if (data.size() >= 7 && (data[5] < .6 || data[5] > data[3] ||
        data[6] < data[3] || data[6] > 4.5)) return false;
    if(data.size()==8 && data[7]!=1.)return false;
    if (id == static_cast<int>(data[0])) return true;  // Freeze this instance.
    id = static_cast<int>(data[0]);
    center = Eigen::Vector2d(data[1], data[2]);
    radius = data[3];
    extent = data[4];
    framing = data.size() >= 7;adaptive=data.size()==8;replans=0;
    minimum_radius = framing ? data[5] : 0.;
    max_radius = framing ? data[6] : radius;
    locked = false;
    turn_first = data.size() == 6;
    oriented = false;
    orientation_steps = 0;
    return true;
  }

  bool framed(const Eigen::Vector2d& point) const
  {
    return !framing || (point - center).norm() + 1e-5 >= minimum_radius;
  }

  static bool safePath(const std::vector<Eigen::Vector2d>& path, const Safe& safe)
  {
    if (path.empty()) return false;
    // A* relaxes checks near the start and appends the exact goal. Validate
    // every segment here, including both of those otherwise unchecked parts.
    for (size_t i = 0; i < path.size(); ++i) {
      if (!safe(path[i])) return false;
      if (!i) continue;
      const auto delta = (path[i] - path[i-1]).eval();
      int n = std::max(1, static_cast<int>(std::ceil(delta.norm() / 0.025)));
      for (int j = 1; j < n; ++j)
        if (!safe(path[i-1] + delta * (static_cast<double>(j) / n))) return false;
    }
    return true;
  }

  bool visible(const Eigen::Vector2d& point, const Safe& clear) const
  {
    auto delta = (center - point).eval();
    double distance = delta.norm();
    if (distance < 1e-6) return false;
    // Do not ray-test the estimated object interior as if it were free space.
    double stop = std::max(0.0, distance - extent / 2.0);
    for (double d = 0.0; d < stop; d += 0.05)
      if (!clear(point + delta * (d / distance))) return false;
    return true;
  }

  bool planAdaptive(const Eigen::Vector2d& start,const Safe& safe,const Safe& clear,
      const Route& route,std::vector<Eigen::Vector2d>& path)
  {
    struct Option {Eigen::Vector2d point;double score;};
    std::vector<Option> options;
    std::vector<double> rings={radius};
    for(double r=minimum_radius+.3125;r<=max_radius+1e-8;r+=.35)
      if(std::abs(r-radius)>.10)rings.push_back(r);
    constexpr double pi=3.14159265358979323846;
    for(double r:rings)for(int k=0;k<32;++k) {
      Eigen::Vector2d point=center+r*Eigen::Vector2d(std::cos(k*pi/16.),std::sin(k*pi/16.));
      if((point-start).norm()<.65 || !safe(point) || !visible(point,clear))continue;
      int free=0;
      for(int x=-3;x<=3;++x)for(int y=-3;y<=3;++y)
        if(x*x+y*y<=9 && safe(point+Eigen::Vector2d(x*.1,y*.1)))++free;
      // Larger framing deficit costs more; a shorter viable partial view can
      // beat an excessively long detour. Safety is always a hard gate.
      double deficit=4.*std::max(0.,1.-r/radius);
      options.push_back({point,deficit+.25*(point-start).norm()-.01*free});
    }
    std::stable_sort(options.begin(),options.end(),[](const Option& a,const Option& b){return a.score<b.score;});
    double best=1e9;int checked=0;std::vector<Eigen::Vector2d> candidate_path,best_path;
    for(const auto& option:options) {
      if(++checked>64 || option.score>=best)break;
      if(!route(option.point,candidate_path) || !safePath(candidate_path,safe))continue;
      double length=0.;for(size_t i=1;i<candidate_path.size();++i)length+=(candidate_path[i]-candidate_path[i-1]).norm();
      if(length>8.0)continue; // Reserve the 48-action inspection budget for translation and turns.
      double score=option.score+.25*(length-(option.point-start).norm());
      if(score<best){best=score;goal=option.point;best_path=candidate_path;}

    }
    if(best_path.empty()){path.clear();return false;}
    locked=true;path=best_path;return true;
  }

  bool plan(const Eigen::Vector2d& start, const Safe& safe, const Safe& clear,
      const Route& route, std::vector<Eigen::Vector2d>& path)
  {
    path.clear();
    if (locked) {
      if(safe(goal) && visible(goal,clear) && route(goal,path) && safePath(path,safe))return true;
      if(!adaptive || replans>=2)return false;
      ++replans;locked=false;
    }
    if(adaptive)return planAdaptive(start,safe,clear,route,path);

    struct Candidate { Eigen::Vector2d point; int ring, free; double distance; };
    std::vector<Candidate> candidates;
    constexpr double pi = 3.14159265358979323846;
    for (int ring = 0; ring < (framing ? 3 : 1); ++ring) {
      const double search_radius = std::min(max_radius, radius + ring * .35);
      if (ring && search_radius <= radius + (ring-1)*.35 + 1e-8) continue;
      for (int i = 0; i < 24; ++i) {
      double angle = 2 * pi * i / 24;
      Eigen::Vector2d point = center + search_radius * Eigen::Vector2d(std::cos(angle), std::sin(angle));
      // Require translation beyond the 0.3125 m arrival tolerance.
      if ((point - start).norm() < 0.65 || !safe(point) || !visible(point, clear)) continue;
      int free = 0;
      for (int x = -5; x <= 5; ++x)
        for (int y = -5; y <= 5; ++y)
          if (x*x + y*y <= 25 && safe(point + Eigen::Vector2d(x*0.1, y*0.1))) ++free;
      candidates.push_back({point, ring, free, (point - start).norm()});
      }
    }
    std::stable_sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
      return std::make_tuple(a.ring, -a.free, a.distance) < std::make_tuple(b.ring, -b.free, b.distance);
    });
    for (const auto& candidate : candidates) {
      if (route(candidate.point, path) && safePath(path, safe)) {
        goal = candidate.point;
        locked = true;
        return true;
      }
    }
    path.clear();
    return false;
  }
};
}  // namespace apexnav_planner
#endif
