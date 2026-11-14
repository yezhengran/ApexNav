#include <plan_env/review_rank.h>
#include <cassert>
#include <limits>
using apexnav_planner::reviewRank;
int main() {
  // Late unconfirmed ranking reverses fusion-only order.
  assert(reviewRank(.8,true,true,2,true,.1) < reviewRank(.6,true,true,2,true,.9));
  assert(reviewRank(.8,true,false,2,true,.1) == .8);
  assert(reviewRank(.8,false,true,2,true,.1) == .8);
  assert(reviewRank(.8,true,true,0,false,0) == .8);
  assert(reviewRank(.8,true,true,1,true,.1) == .8);
  assert(reviewRank(.8,true,true,3,true,.1) == .8);
  assert(reviewRank(.8,true,true,2,true,0) == .4);
  bool caught=false;try {reviewRank(.8,true,true,2,true,std::numeric_limits<double>::quiet_NaN());} catch (...) {caught=true;}
  assert(caught);
  using apexnav_planner::validReviewReceipt;
  assert(validReviewReceipt(true,4,5,42,42,10,10,2,.5));
  assert(!validReviewReceipt(true,4,5,41,42,10,10,2,.5));
  assert(!validReviewReceipt(true,4,5,42,42,9,10,2,.5));
  assert(!validReviewReceipt(true,4,6,42,42,10,10,2,.5));
  assert(!validReviewReceipt(true,-1,0,42,42,10,10,2,.5));
  assert(!validReviewReceipt(false,4,5,42,42,10,10,2,.5));
  assert(!validReviewReceipt(true,4,5,42,42,10,10,0,.5));
  assert(!validReviewReceipt(true,4,5,42,42,10,10,2,1.1));
}
