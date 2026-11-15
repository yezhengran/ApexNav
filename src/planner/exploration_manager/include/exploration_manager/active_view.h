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
  bool framing = false;
  bool annular = false;
  std::vector<Eigen::Vector2d> samples;
  size_t sample_index = 0;
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
    if (data.size() == 8 && data[7] != 1.) return false;
    if (id == static_cast<int>(data[0])) return true;  // Freeze this instance.
    id = static_cast<int>(data[0]);
    center = Eigen::Vector2d(data[1], data[2]);
    radius = data[3];
    extent = data[4];
    framing = data.size() >= 7;
    annular = data.size() == 8;
    samples.clear(); sample_index = 0;
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
    const double d = (point - center).norm();
    return !framing || (d + 1e-5 >= minimum_radius && (!annular || d <= max_radius + 1e-5));
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
    if (annular) return planAnnulus(start, safe, clear, route, path);
    if (locked)
      return safe(goal) && visible(goal, clear) && route(goal, path) && safePath(path, safe);

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

  bool advanceSample()
  {
    if (!annular || sample_index + 1 >= samples.size()) return false;
    goal = samples[++sample_index];
    return true;
  }

  bool planAnnulus(const Eigen::Vector2d& start, const Safe& safe, const Safe& clear,
      const Route& route, std::vector<Eigen::Vector2d>& path)
  {
    auto bandSafe = [&](const Eigen::Vector2d& p) { return safe(p) && framed(p); };
    if (locked) {
      // Access to the band may start outside it; subsequent arc motion must
      // stay in observed, collision-free band space, including A* segments.
      return safe(goal) && framed(goal) && visible(goal, clear) && route(goal, path) &&
          safePath(path, sample_index ? Safe(bandSafe) : safe);
    }
    constexpr double pi = 3.14159265358979323846;
    double best_cost = INFINITY;
    std::vector<Eigen::Vector2d> best_path;
    for (int ring = 0; ring < 3; ++ring) {
      const double r = std::min(max_radius, radius + .35 * ring);
      if (ring && r <= std::min(max_radius, radius + .35*(ring-1)) + 1e-8) continue;
      for (int k = 0; k < 24; ++k) {
        const double angle = k * pi / 12.;
        const Eigen::Vector2d entry = center + r * Eigen::Vector2d(std::cos(angle), std::sin(angle));
        if (!bandSafe(entry) || !visible(entry, clear)) continue;
        std::vector<Eigen::Vector2d> access;
        if (!route(entry, access) || !safePath(access, safe)) continue;
        double access_cost = 0.;
        for (size_t i=1;i<access.size();++i) access_cost += (access[i]-access[i-1]).norm();
        for (int sign : {-1, 1}) {
          std::vector<Eigen::Vector2d> candidate{entry};
          double cost = access_cost;
          // A short 30-degree arc, with a center-facing sample every 15 degrees.
          // Never demand a complete circle through walls or unknown space.
          for (int j=1;j<=2;++j) {
            const double a = angle + sign*j*pi/12.;
            const Eigen::Vector2d p = center + r * Eigen::Vector2d(std::cos(a), std::sin(a));
            if (!bandSafe(p) || !visible(p, clear) || !safePath({candidate.back(), p}, bandSafe)) break;
            cost += (p-candidate.back()).norm(); candidate.push_back(p);
          }
          // At least two distinct observation positions. Prefer the complete
          // short arc; a one-segment arc remains useful in narrow free space.
          if (candidate.size() < 2) continue;
          cost += candidate.size() == 2 ? .5 : 0.;
          if (cost < best_cost) { best_cost=cost; samples=candidate; best_path=access; }
        }
      }
    }
    if (samples.empty()) { path.clear(); return false; }
    sample_index=0; goal=samples.front(); locked=true; path=best_path;
    return true;
  }
};
}  // namespace apexnav_planner
#endif
