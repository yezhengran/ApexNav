#ifndef APEX_REVIEW_RANK_H
#define APEX_REVIEW_RANK_H
#include <cmath>
#include <cstdint>
#include <stdexcept>
namespace apexnav_planner {
inline bool validReviewReceipt(bool enabled, int pending, int candidate,
    uint64_t token, uint64_t expected_token, uint32_t observation, uint32_t step,
    int decision, double confidence) {
  return enabled && pending >= 0 && candidate == pending + 1 && token != 0 &&
      token == expected_token && observation == step && decision >= 1 && decision <= 4 &&
      std::isfinite(confidence) && confidence >= 0 && confidence <= 1;
}
inline double reviewRank(double fusion, bool enabled, bool late, int state,
                         bool has_qwen, double qwen) {
  if (!enabled || !late || state == 1 || state == 3 || !has_qwen) return fusion;
  if (!std::isfinite(qwen) || qwen < 0 || qwen > 1)
    throw std::invalid_argument("Invalid target confidence");
  return 0.5 * fusion + 0.5 * qwen;
}
inline bool annularCandidateBefore(int aid, double af, int as, bool aq, double av,
    int bid, double bf, int bs, bool bq, double bv, bool late) {
  if ((aq && (!std::isfinite(av) || av<0 || av>1)) ||
      (bq && (!std::isfinite(bv) || bv<0 || bv>1)))
    throw std::invalid_argument("Invalid annular target confidence");
  if ((as==1)!=(bs==1)) return as==1; // Confirmed objects precede uncertain ones.
  if (late && aq!=bq) return aq;      // Unreviewed fusion-only fallback comes last.
  const double a=late&&aq?av:af, b=late&&bq?bv:bf;
  return a==b ? aid<bid : a>b;
}
}
#endif
