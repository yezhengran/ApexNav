#include <cassert>
#include <exploration_manager/observation_sequence.h>
using apexnav_planner::ObservationSequence;
int main() {
  ObservationSequence sequence;
  assert(!sequence.accepts(0, 0));
  assert(!sequence.accepts(10, 1));
  assert(sequence.accepts(10, 0)); sequence.commit(10, 0);
  assert(!sequence.accepts(10, 0));  // Repeated callback cannot integrate twice.
  assert(!sequence.accepts(10, 2));  // No dropped frame may silently advance.
  assert(!sequence.accepts(11, 1)); // No cross-episode contamination.
  assert(sequence.accepts(10, 1)); sequence.commit(10, 1);
  sequence.reset();
  assert(!sequence.accepts(10, 0));
  assert(!sequence.accepts(10, 2));
  assert(!sequence.accepts(9, 0));
  assert(sequence.accepts(11, 0)); sequence.commit(11, 0);
  assert(sequence.accepts(11, 1));
}
