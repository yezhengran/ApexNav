"""Q8-C: detector OR fusion review trigger and relative multi-object selection.

Relative comparison never changes per-object decisions or category scores.
Images are one best retained crop per exact mapper identity, not a joint view
of one object. Every request and source is retained independently for auditing.
"""
import shutil
import time

from PIL import Image

from .annular_review import AnnularTargetReviewer
from .qwen_review import QwenTransportError


class ChoiceTargetReviewer(AnnularTargetReviewer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not getattr(self.cfg, 'choice_review', False) or not callable(
                getattr(self.multiview_backend, 'compare_candidates', None)):
            raise ValueError('Choice review requires its explicit profile and backend')
        self.comparison_count = 0
        self.last_comparison_request = 0

    def crop_ready(self):
        return sorted(ident for ident, rows in self.frames.items()
                      if rows and min(rows, key=self._rank)['score'] > .7)

    def comparison_ready(self):
        return sorted(ident for ident, rows in self.frames.items() if rows and
                      self.histories.get(ident, {}).get('decision') != self.REJECTED)

    def _first_review_allowed(self, ident, score, threshold):
        crop = ident in self.crop_ready()
        allowed = score >= threshold or crop
        self.event('choice_trigger_check', candidate=ident, fused_confidence=score,
                   threshold=threshold, crop_threshold=.7, crop_passed=crop, allowed=allowed)
        return allowed

    def compare(self, candidate_ids, request_id, episode_token, observation_id, reason=""):
        if (episode_token != self.episode_token or observation_id != self.step or
                request_id <= self.last_comparison_request or self.selected is not None or
                self.active is not None or len(candidate_ids) < 2 or
                candidate_ids != sorted(set(candidate_ids)) or
                not set(candidate_ids).issubset(self.comparison_ready())):
            raise RuntimeError('Invalid or stale candidate comparison request')
        if reason not in ('', 'multiple_confirmed', 'late_unconfirmed', 'no_frontier'):
            raise RuntimeError('Unknown candidate comparison phase')
        if reason == 'no_frontier' and (not getattr(self.cfg, 'choice_no_frontier', False) or
                any(i not in self.qwen_scores or self.histories.get(i, {}).get('decision') not in
                    (self.CONFIRMED, self.UNCERTAIN, self.PENDING_EVIDENCE) for i in candidate_ids)):
            raise RuntimeError('No-frontier comparison requires reviewed nonrejected candidates')
        self.last_comparison_request = request_id
        self.event('comparison_requested', request_id=request_id, candidate_ids=candidate_ids, reason=reason)
        remaining = list(candidate_ids)
        round_index = 0
        # Groups have <=5 images (same conservative per-HTTP reservation). All
        # candidates are considered. Each group returns exactly one relative winner.
        while len(remaining) > 1:
            winners = []
            for offset in range(0, len(remaining), 5):
                group = remaining[offset:offset + 5]
                if len(group) == 1:
                    winners.extend(group)
                    continue
                rows, sources, crops = [], [], []
                for ident in group:
                    e = self._load(min(self.frames[ident], key=self._rank), ident)
                    stem = f'comparison-{request_id:03d}-{round_index:02d}-{offset:03d}-{ident:03d}'
                    source = self.directory / (stem + '.npz')
                    shutil.copyfile(e['source_file'], source)
                    crop = self.directory / (stem + '.png')
                    Image.fromarray(e['crop']).save(crop)
                    rows.append(e); crops.append(str(crop))
                    sources.append(dict(candidate=ident, observation=e['step'],
                        mask_index=e['key'][1], source_file=str(source), crop=str(crop),
                        detector_score=e['score'], pose=e['pose'], bounds=e['bounds'],
                        mask_pixels=e['pixels'], complete=e['complete'], detection_label=e['label']))
                self.event('comparison_sources', request_id=request_id, round_index=round_index,
                           candidate_ids=group, sources=sources, crops=crops)
                started = time.monotonic()
                try:
                    reply = self.multiview_backend.compare_candidates(
                        [e['crop'] for e in rows], self.label, group)
                except QwenTransportError as error:
                    self.event('verification_transport_error', phase='candidate_comparison',
                               candidate=0, request_id=request_id, call_id=error.call_id,
                               retryable=error.retryable, crop=crops[0], crops=crops)
                    raise
                answer = reply['answer']
                ident, outcome = answer['candidate_id'], answer['outcome']
                if (outcome != 'selected' or type(ident) is not int or ident not in group):
                    raise RuntimeError('Comparison returned an invalid candidate')
                seconds = time.monotonic() - started
                self.stats['vlm_calls'] += 1
                self.stats['vlm_seconds'] += seconds
                self.comparison_count += 1
                self.event('candidate_comparison', request_id=request_id, round_index=round_index,
                           candidate_ids=group, selected_candidate=ident, outcome=outcome,
                           model_reply=reply, sources=sources, crops=crops, seconds=seconds)
                winners.append(ident)
            remaining = winners
            outcome = 'selected' if remaining else 'none'
            round_index += 1
        selected = remaining[0] if remaining else 0
        self.event('comparison_finished', request_id=request_id, candidate_ids=candidate_ids,
                   selected_candidate=selected, outcome=outcome)
        return selected
