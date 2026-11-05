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
  bool locked = false;
  Eigen::Vector2d goal = Eigen::Vector2d::Zero();
  using Safe = std::function<bool(const Eigen::Vector2d&)>;
  using Route = std::function<bool(const Eigen::Vector2d&, std::vector<Eigen::Vector2d>&)>;

  bool set(const std::vector<double>& data)
  {
    if (data.empty()) { *this = ActiveView(); return true; }
    if (data.size() != 5) return false;
    for (double x : data) if (!std::isfinite(x)) return false;
    if (data[0] < 1 || data[0] > 1000000 || std::floor(data[0]) != data[0] ||
        data[3] < 0.6 || data[3] > 3.5 || data[4] < 0.15 || data[4] > 3.0)
      return false;
    if (id == static_cast<int>(data[0])) return true;  // Freeze this instance.
    id = static_cast<int>(data[0]);
    center = Eigen::Vector2d(data[1], data[2]);
    radius = data[3];
    extent = data[4];
    locked = false;
    return true;
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

  bool plan(const Eigen::Vector2d& start, const Safe& safe, const Safe& clear,
      const Route& route, std::vector<Eigen::Vector2d>& path)
  {
    path.clear();
    if (locked)
      return safe(goal) && visible(goal, clear) && route(goal, path) && safePath(path, safe);

    struct Candidate { Eigen::Vector2d point; int free; double distance; };
    std::vector<Candidate> candidates;
    constexpr double pi = 3.14159265358979323846;
    for (int i = 0; i < 24; ++i) {
      double angle = 2 * pi * i / 24;
      Eigen::Vector2d point = center + radius * Eigen::Vector2d(std::cos(angle), std::sin(angle));
      if ((point - start).norm() < 0.30 || !safe(point) || !visible(point, clear)) continue;
      int free = 0;
      for (int x = -5; x <= 5; ++x)
        for (int y = -5; y <= 5; ++y)
          if (x*x + y*y <= 25 && safe(point + Eigen::Vector2d(x*0.1, y*0.1))) ++free;
      candidates.push_back({point, free, (point - start).norm()});
    }
    std::stable_sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
      return std::make_tuple(-a.free, a.distance) < std::make_tuple(-b.free, b.distance);
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
