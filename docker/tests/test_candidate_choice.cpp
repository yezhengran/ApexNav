#include <exploration_manager/candidate_choice.h>
#include <cassert>
using namespace apexnav_planner;
int main() {
  std::vector<ChoiceCandidate> pool={{1,true,false,true},{2,true,false,true},{3,false,false,true}};
  assert((comparisonPool(399,pool)==std::vector<int>{1,2}));
  assert(comparisonPool(400,pool).empty());
  pool[0].confirmed=pool[1].confirmed=false;
  assert(comparisonPool(399,pool).empty());
  assert((comparisonPool(400,pool)==std::vector<int>{1,2,3}));
  pool[1].rejected=true; pool[2].reachable=false;
  assert(comparisonPool(400,pool).empty());
  pool[1].rejected=false;
  assert((comparisonPool(499,pool)==std::vector<int>{1,2}));
  pool[2].confirmed=true;
  assert(comparisonPool(400,pool).empty()); // unreachable confirmation still counts
  std::vector<int32_t> ids={2,4};
  assert(validComparisonReceipt(true,7,7,400,400,1,1,ids,ids,4));
  assert(!validComparisonReceipt(false,7,7,400,400,1,1,ids,ids,4));
  assert(!validComparisonReceipt(true,8,7,400,400,1,1,ids,ids,4));
  assert(!validComparisonReceipt(true,7,7,399,400,1,1,ids,ids,4));
  assert(!validComparisonReceipt(true,7,7,400,400,2,1,ids,ids,4));
  assert(!validComparisonReceipt(true,7,7,400,400,1,1,ids,ids,3));
  assert(!validComparisonReceipt(true,7,7,400,400,1,1,ids,{2,5},4));
}
