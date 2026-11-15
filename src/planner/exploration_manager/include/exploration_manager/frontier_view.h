#ifndef APEX_FRONTIER_VIEW_H
#define APEX_FRONTIER_VIEW_H
#include <Eigen/Core>
#include <cmath>
#include <functional>
#include <set>

namespace apexnav_planner {
// Optional arrival-only look. No frontier creation, route or collision changes.
// Map callback: occupied/outside=-1, known free=0, unknown=1.
class FrontierView {
public:
  using Cell = std::function<int(const Eigen::Vector2d&)>;
  using Address = std::function<int(const Eigen::Vector2d&)>;
  static int gain(const Eigen::Vector2d& p, double yaw, double left, double right,
      double range, double resolution, const Cell& cell, const Address& address) {
    std::set<int> boundary;
    const int rays = std::max(1, static_cast<int>(std::ceil((left+right)*range/resolution)));
    for (int i=0;i<=rays;++i) {
      const double a=yaw-right+(left+right)*i/rays;
      const Eigen::Vector2d direction(std::cos(a),std::sin(a));
      for (double d=resolution;d<=range;d+=resolution*.5) {
        const Eigen::Vector2d q=p+d*direction;
        const int state=cell(q);
        if (state < 0) break;
        if (state > 0) { boundary.insert(address(q)); break; }
      }
    }
    return static_cast<int>(boundary.size());
  }

  int next(const Eigen::Vector2d& p, double yaw, bool arrived,
           const std::function<int(double)>& score) {
    if (used_ && (p-anchor_).norm() > .75) { used_=false; remaining_=0; }
    if (!used_) {
      if (!arrived) return 0;
      used_=true; anchor_=p;
      const int baseline=score(yaw);
      double best=0.; int chosen=0;
      for (int turns=1;turns<=3;++turns) for (int sign : {1,-1}) {
        const double value=static_cast<double>(score(yaw+sign*turns*step)-baseline)/turns;
        if (value>best) { best=value; chosen=sign*turns; }
      }
      if (!chosen) return 0;
      target_=yaw+chosen*step; remaining_=std::abs(chosen);
    }
    if (!remaining_) return 0;
    const double delta=std::atan2(std::sin(target_-yaw),std::cos(target_-yaw));
    // Stop as soon as the chosen direction has no remaining map gain.
    if (std::abs(delta)<step*.5 || score(target_)<=score(yaw)) { remaining_=0; return 0; }
    --remaining_;
    return delta>0 ? 1 : -1;
  }
private:
  static constexpr double step=3.14159265358979323846/6.;
  bool used_=false;
  int remaining_=0;
  double target_=0.;
  Eigen::Vector2d anchor_=Eigen::Vector2d::Zero();
};
}
#endif
