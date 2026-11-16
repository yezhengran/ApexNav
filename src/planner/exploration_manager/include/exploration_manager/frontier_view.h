#ifndef APEX_FRONTIER_VIEW_H
#define APEX_FRONTIER_VIEW_H
#include <Eigen/Core>
#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <vector>

namespace apexnav_planner {
// Optional exploration-goal heading. Never changes a goal position or route.
// Map callback: occupied/outside=-1, known free=0, unknown=1.
class FrontierView {
public:
  using Cell = std::function<int(const Eigen::Vector2d&)>;
  using Points = std::vector<Eigen::Vector2d>;
  struct Heading { bool valid=false; double yaw=0., width=0.; bool normal=false; };
  static constexpr double pi=3.14159265358979323846;
  static constexpr double arrival=.5;
  static constexpr double separation=.75;
  static constexpr int max_turns=6;

  static bool boundary(const Eigen::Vector2d& p, double res, const Cell& cell) {
    if (cell(p)!=1) return false;
    for (const auto& d : offsets()) if (cell(p+res*d)==0) return true;
    return false;
  }
  static bool visible(const Eigen::Vector2d& from, const Eigen::Vector2d& to,
                      double res, double range, const Cell& cell) {
    const double distance=(to-from).norm();
    if (distance>range) return false;
    // The mean frontier itself can occupy an unknown boundary cell. Ignore
    // only that initial cell; intervening occupied/unknown cells occlude.
    for (double d=res; d<distance-res*.75; d+=res*.5)
      if (cell(from+(to-from)*(d/distance))!=0) return false;
    return true;
  }
  static Heading heading(const Eigen::Vector2d& goal, const Points& cells,
                         double res, double range, const Cell& cell) {
    Heading result;
    if (!(res>0. && range>0.) || !goal.allFinite()) return result;
    constexpr int bins=720;
    constexpr double angle=2.*pi/bins;
    std::array<bool,bins> occupied{};
    Eigen::Vector2d normal=Eigen::Vector2d::Zero();
    bool on_boundary=false;
    for (const auto& p : cells) {
      if (!p.allFinite() || !boundary(p,res,cell) || !visible(goal,p,res,range,cell)) continue;
      const Eigen::Vector2d delta=p-goal;
      const double distance=delta.norm();
      on_boundary |= distance<=res;
      // Points away from the known free neighbors, into the unknown side.
      for (const auto& d : offsets())
        if (cell(p+res*d)==0) normal-=d/(res+distance);
      if (distance<=res*.5) continue;
      const double a=std::atan2(delta.y(),delta.x());
      const double half=std::asin(std::min(1.,res*.71/distance));
      for (int b=0;b<bins;++b)
        if (std::abs(wrap(b*angle-a))<=half+angle*.5) occupied[b]=true;
    }
    int best_start=0,best_length=0,length=0;
    for (int b=0;b<2*bins;++b) {
      length=occupied[b%bins] ? std::min(bins,length+1) : 0;
      if (length>best_length) {best_length=length;best_start=b-length+1;}
    }
    // Frontier goals are normally cluster means. If the mean lies on the
    // boundary, facing the mean is undefined (and facing its tangent is wrong).
    // Use its free-to-unknown normal in that degenerate case.
    if (on_boundary || best_length>=bins/2) {
      if (normal.norm()>1e-6 && cell(goal+res*normal.normalized())==1)
        result={true,std::atan2(normal.y(),normal.x()),0.,true};
    } else if (best_length>0) {
      result={true,wrap((best_start+(best_length-1)*.5)*angle),best_length*angle,false};
    }
    return result;
  }

  // Bind after the planner's normal path-stability selection. A small shift
  // of the same goal must not continuously change its stored heading.
  bool bind(const Eigen::Vector2d& position, const Eigen::Vector2d& goal,
            const Points& cells, double res, double range, const Cell& cell) {
    if (used_ && (position-anchor_).norm()<=separation) return false;
    if (used_) used_=false;
    if (pending_ && (goal-goal_).norm()<=res*1.5) return false;
    if (pending_ && arrived(position)) return false;
    pending_=false;
    const auto h=heading(goal,cells,res,range,cell);
    if (!h.valid) return false;
    goal_=goal; cells_=cells; heading_=h; remaining_=max_turns; pending_=true;
    return true;
  }
  bool arrived(const Eigen::Vector2d& position) const {
    return pending_ && (position-goal_).norm()<arrival;
  }
  int next(const Eigen::Vector2d& position, double yaw, double res,
           double range, const Cell& cell) {
    if (!arrived(position)) return 0;
    bool valid=false;
    const Eigen::Vector2d normal(std::cos(heading_.yaw),std::sin(heading_.yaw));
    for (const auto& p : cells_) {
      const Eigen::Vector2d delta=p-goal_;
      const bool in_span=heading_.normal ? cell(goal_+res*normal)==1 :
          std::abs(wrap(std::atan2(delta.y(),delta.x())-heading_.yaw))<=
              heading_.width*.5+std::atan2(res,delta.norm());
      if (in_span && boundary(p,res,cell) && visible(position,p,res,range,cell)) {
        valid=true; break;
      }
    }
    const double delta=wrap(heading_.yaw-yaw);
    if (!valid || !remaining_ || std::abs(delta)<=pi/12.+1e-6) {
      finish(position); return 0;
    }
    used_=true; anchor_=position; --remaining_;
    return delta>0. ? 1 : -1;
  }
  void cancel() { pending_=false; remaining_=0; }
  const Heading& target() const { return heading_; }
  const Eigen::Vector2d& goal() const { return goal_; }
private:
  static double wrap(double a) { return std::atan2(std::sin(a),std::cos(a)); }
  static const std::array<Eigen::Vector2d,4>& offsets() {
    static const std::array<Eigen::Vector2d,4> values={
      Eigen::Vector2d(1,0),Eigen::Vector2d(-1,0),Eigen::Vector2d(0,1),Eigen::Vector2d(0,-1)};
    return values;
  }
  void finish(const Eigen::Vector2d& position) { cancel(); used_=true; anchor_=position; }
  bool pending_=false,used_=false;
  int remaining_=0;
  Heading heading_;
  Points cells_;
  Eigen::Vector2d goal_=Eigen::Vector2d::Zero(),anchor_=Eigen::Vector2d::Zero();
};
}
#endif
