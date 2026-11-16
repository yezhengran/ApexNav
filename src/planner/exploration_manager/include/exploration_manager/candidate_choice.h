#ifndef APEXNAV_CANDIDATE_CHOICE_H
#define APEXNAV_CANDIDATE_CHOICE_H
#include <algorithm>
#include <cstdint>
#include <vector>
namespace apexnav_planner {
struct ChoiceCandidate { int id; bool confirmed; bool rejected; bool reachable; bool reviewed = false; };
// IDs are zero-based inside the planner. Preserve the explicit user gates;
// late selection cannot bypass an existing confirmation, even if unreachable.
inline std::vector<int> comparisonPool(int step, const std::vector<ChoiceCandidate>& candidates) {
  std::vector<int> ids;
  for (const auto& c : candidates) {
    if (step >= 400 && c.confirmed) return {};
    if (!c.rejected && c.reachable && (step >= 400 || c.confirmed)) ids.push_back(c.id);
  }
  std::sort(ids.begin(), ids.end());
  ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
  return ids.size() >= 2 ? ids : std::vector<int>{};
}
// No-frontier approach never lets an unreviewed fusion leader displace an
// available Qwen-reviewed object. A single candidate needs no comparative call.
inline std::vector<int> noFrontierPool(const std::vector<ChoiceCandidate>& candidates) {
  bool confirmed=false;
  for (const auto& c:candidates)
    confirmed |= c.reviewed && c.reachable && !c.rejected && c.confirmed;
  std::vector<int> ids;
  for (const auto& c:candidates)
    if (c.reviewed && c.reachable && !c.rejected && (!confirmed || c.confirmed)) ids.push_back(c.id);
  std::sort(ids.begin(),ids.end()); ids.erase(std::unique(ids.begin(),ids.end()),ids.end());
  return ids;
}
inline bool validComparisonReceipt(bool pending, uint64_t token, uint64_t expected_token,
    uint32_t observation, uint32_t expected_observation, uint32_t request, uint32_t expected_request,
    const std::vector<int32_t>& ids, const std::vector<int32_t>& expected_ids, int selected) {
  return pending && token && token == expected_token && observation == expected_observation &&
      request == expected_request && ids == expected_ids && ids.size() >= 2 &&
      selected > 0 && std::find(ids.begin(), ids.end(), selected) != ids.end();
}
}
#endif
