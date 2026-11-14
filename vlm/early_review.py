"""Opt-in automatic review of detector-qualified, mapper-linked candidate images.

Only raw single-frame target confidence gates this mode. Map confidence never
supplies an image score. Uncertain reviews wait for natural new views, without
requesting an inspection detour or changing the arrival condition.
"""

import math
import shutil

from .active_view import SelectedTargetReviewer
from .linked_review import LinkedTargetReviewer


class EarlyTargetReviewer(LinkedTargetReviewer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.image_threshold = float(self.cfg.early_review_threshold)
        if not math.isfinite(self.image_threshold) or not 0 <= self.image_threshold <= 1:
            raise ValueError('Early image threshold must be finite and in [0, 1]')
        if not getattr(self.cfg, 'early_review', False) or getattr(self.cfg, 'stop_review', False):
            raise ValueError('Early review requires its separate profile, without STOP review')

    @staticmethod
    def _rank(e):
        # Within usable target frames, confidence wins over size/completeness.
        return (e['label'] != 0, -e['score'], not e['complete'], -e['pixels'], -e['step'])

    def _eligible(self, e):
        return (self._usable(e) and e['label'] == 0 and
                math.isfinite(e['score']) and self.image_threshold <= e['score'] <= 1)

    def _retain(self, e, packet):
        # Fail closed on invalid detector output, rather than poisoning ranking.
        if not math.isfinite(e['score']) or not 0 <= e['score'] <= 1:
            raise ValueError('Detector confidence must be finite and in [0, 1]')
        super()._retain(e, packet)
        self.event('candidate_image', candidate=e['candidate'], observation=e['step'],
                   mask_index=e['key'][1], detector_score=e['score'], label=e['label'],
                   pixels=e['pixels'], complete=e['complete'],
                   eligible=self._eligible(e), threshold=self.image_threshold,
                   retained=any(row is e for row in self.frames[e['candidate']]))

    def _available(self, ident, history):
        if history['calls'] >= self.max_calls or history.get('decision') in (self.CONFIRMED, self.REJECTED):
            return []
        return [e for e in self.frames.get(ident, [])
                if self._eligible(e) and self._novel(e, history)]

    def readiness(self):
        ready = []
        for ident in self.frames:
            history = self.histories.get(ident, SelectedTargetReviewer._target_history(self, ident))
            if self.selected is not None and self.selected['id'] == ident:
                history = self.selected
            available = self._available(ident, history)
            if available:
                ready.append((ident, max(e['step'] for e in available)))
        return sorted(ready)

    def _evidence(self, unused=False):
        self.evidence_filter_counts = {'threshold': self.image_threshold}
        available = self._available(self.selected['id'], self.selected)
        return min(available, key=self._rank) if available else None

    def begin(self, data):
        if len(data) != 11 or data[-1] != 2.0:
            raise ValueError('Early review requires automatic-review metadata')
        self.event('automatic_review_started', candidate=int(data[0]), threshold=self.image_threshold)
        # Deliberately bypass only the old selected/STOP trigger, not its geometry,
        # identity, budget, decision parsing, or transport checks.
        return SelectedTargetReviewer.begin(self, data[:10])

    def _allow_high_confidence(self):
        return True

    def _may_inspect(self):
        return False

    def _verify(self, phase, evidence=None):
        chosen = evidence if evidence is not None else self._evidence(unused=True)
        if chosen is not None:
            # Include mask index: two masks in one frame must never overwrite an
            # already-used source, even if they belong to the same map candidate.
            path = self.directory / (f'selected-{self.selected["id"]:03d}-'
                                     f's{chosen["step"]:04d}-m{chosen["key"][1]:02d}.npz')
            shutil.copyfile(chosen['source_file'], path)
            self.event('verification_source', candidate=self.selected['id'],
                       observation=chosen['step'], mask_index=chosen['key'][1],
                       source_file=str(path), detector_score=chosen['score'],
                       trigger_threshold=self.image_threshold, selection='highest_qualified_detector_score')
        return SelectedTargetReviewer._verify(self, phase, chosen)
