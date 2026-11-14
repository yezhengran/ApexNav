#pragma once
#include <cstdint>

namespace apexnav_planner {
// A reset opens a new monotonically identified episode, never an old packet.
// No automatic retransmission: rejected/gapped frames stop the evaluator.
class ObservationSequence {
 public:
  void reset() { active_ = false; }
  bool accepts(uint64_t token, uint32_t step) const {
    return token && (active_ ? token == token_ && step == step_ + 1
                            : token > token_ && step == 0);
  }
  void commit(uint64_t token, uint32_t step) {
    token_ = token; step_ = step; active_ = true;
  }
 private:
  uint64_t token_ = 0;
  uint32_t step_ = 0;
  bool active_ = false;
};
}  // namespace apexnav_planner
