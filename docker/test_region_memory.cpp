#include "exploration_manager/region_memory.h"
#include <cassert>
using apexnav_planner::RegionMemory;
int main() {
 RegionMemory m;m.observe(1,{2.4,0});m.observe(1,{2.4,0});assert(m.visits[RegionMemory::key({2.4,0})]==1);
 m.observe(2,{2.6,0});assert(m.traversed_edges.size()==1);
 m.observe(3,{9,9});assert(m.traversed_edges.size()==1); // A teleport is not evidence of a portal.
 assert(RegionMemory::utility(.5,.5,2,0)>RegionMemory::utility(.5,.5,2,48));
 assert(RegionMemory::utility(.5,.5,2,0)>RegionMemory::utility(.5,.5,20,0));
 assert(RegionMemory::utility(.6,.5,2,0)>RegionMemory::utility(.1,.5,2,0));
 assert(RegionMemory::keep(10,32,.5,.6));assert(!RegionMemory::keep(40,32,.5,.8));
 m.block({1,1},10);assert(m.cooling({1.1,1.1},20));assert(!m.cooling({1.1,1.1},90));assert(!m.cooling({2,2},20));
 RegionMemory fresh;assert(fresh.visits.empty() && fresh.failed.empty() && fresh.scans==0);
}
