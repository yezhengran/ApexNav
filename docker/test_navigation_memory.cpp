#include <exploration_manager/navigation_memory.h>
#include <cassert>
#include <iostream>
using apexnav_planner::NavigationMemory;
using P=Eigen::Vector2d;
int main(){
  assert(NavigationMemory::keepTarget(50,.8079,.8080,false,false));
  assert(!NavigationMemory::keepTarget(50,.70,.81,false,false));
  assert(NavigationMemory::keepTarget(8,.70,.81,false,false));
  assert(!NavigationMemory::keepTarget(8,.95,.60,false,true));
  assert(NavigationMemory::keepTarget(100,.60,.95,true,false));
  NavigationMemory m;m.commit(8,P(1,2),10);m.commit(8,P(1,2),20);
  assert(m.selected_at==10);m.abandon(40);assert(m.cooling(8,99)&&!m.cooling(8,100));
  assert(!m.has_goal&&m.selected==-1);
  m.failEdge(P(0,0),P(1,0),50);
  assert(m.blockedEdge(P(0,0),P(1,0),89));
  assert(!m.blockedEdge(P(0,0),P(1,0),90));
  assert(!m.blockedEdge(P(0,0),P(-1,0),60));
  assert(!m.blockedEdge(P(1,0),P(1,0),60));
  NavigationMemory blocked;
  for(int step=0;step<23;++step) assert(blocked.observe(step,P(0,0),P(3,0),true)==0);
  assert(blocked.observe(23,P(0,0),P(3,0),true)==1);
  auto size=blocked.window.size();assert(blocked.observe(23,P(0,0),P(3,0),true)==0);
  assert(blocked.window.size()==size);
  NavigationMemory walk;
  for(int step=0;step<60;++step) assert(walk.observe(step,P(step*.25,0),P(20,0),false)==0);
  NavigationMemory turning;
  for(int step=0;step<40;++step) assert(turning.observe(step,P(0,0),P(3,0),false)==0);
  NavigationMemory loop;int last=0;
  for(int step=0;step<60;++step) last=loop.observe(step,P(step%2,0),P(3,0),false);
  assert(last==2);
  // A new object must not inherit a no-progress verdict from exploration.
  NavigationMemory scoped;scoped.scoped_progress=true;scoped.beginGoal(-2,P(8,0));
  for(int step=0;step<23;++step) scoped.observe(step,P(step%2,0),P(8,0),false);
  scoped.commit(6,P(3,0),23);
  assert(scoped.window.empty());
  assert(scoped.observe(23,P(0,0),P(3,0),false)==0);
  assert(scoped.visited.size()==2); // exploration/collision memory is retained
  scoped.failEdge(P(0,0),P(1,0),23);
  scoped.beginGoal(10,P(3,0));assert(scoped.window.empty());
  assert(scoped.blockedEdge(P(0,0),P(1,0),24));
  for(int step=24;step<47;++step)assert(scoped.observe(step,P(0,0),P(3,0),true)==0);
  assert(scoped.observe(47,P(0,0),P(3,0),true)==1); // genuine new-goal stall still detected
  auto count=scoped.window.size();scoped.beginGoal(10,P(3.2,0));
  assert(scoped.window.size()==count); // small geometry jitter cannot erase the window
  scoped.beginGoal(10,P(4,0));assert(scoped.window.empty());
  scoped.beginGoal(-2,P(4,0));assert(scoped.progress_scope==-2);
  NavigationMemory old;old.observe(0,P(0,0),P(1,0),false);old.commit(8,P(5,0),1);
  assert(old.window.size()==1); // legacy A/B behavior stays opt-in
  assert(!NavigationMemory::keepPriority(.5,.65,false,false));
  assert(NavigationMemory::keepPriority(.5,.55,false,false));
  assert(NavigationMemory::keepPriority(.5,.95,true,false));
  assert(!NavigationMemory::keepPriority(.95,.5,false,true));
  NavigationMemory reset;assert(reset.visited.empty()&&reset.failed_edges.empty()&&reset.cooldown_until.empty());
  std::cout<<"Navigation scenarios passed: hysteresis, priority, cooldown, edge expiry, blocked motion, duplicate frame, advancing, rotation, loop, episode reset\n";
}
