"""Fused-map trigger, per-object best crop, and one conditional active view.

The planner owns fusion eligibility and alternative-target reachability. This
reviewer owns exact image lineage, two completed calls and one inspection budget.
Images live on disk; per-candidate metadata survives the old 48-object hot cache.
"""
import math
import shutil

import numpy as np
from PIL import Image

from .active_view import SelectedTargetReviewer, crop_target, observation_radius
from .linked_review import LinkedTargetReviewer


class FusedTargetReviewer(LinkedTargetReviewer):
    FIRST, INSPECTION = 3.0, 4.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not getattr(self.cfg, 'fused_review', False) or any(
                getattr(self.cfg, name, False) for name in ('early_review', 'stop_review')):
            raise ValueError('Fused review requires its independent profile')
        self.inspected = set()
        self.first_steps = {}
        self.first_decisions = {}

    @staticmethod
    def _rank(e):
        return (-e['score'], not e['complete'], -e['pixels'], -e['step'], e['key'][1])

    def _eligible(self, e):
        return self._usable(e) and e['label'] == 0

    def _retain(self, e, packet):
        if not math.isfinite(e['score']) or not 0 <= e['score'] <= 1:
            raise ValueError('Invalid detector confidence')
        ident = e['candidate']
        existing = self.frames.get(ident, [])
        pool = existing + ([e] if self._eligible(e) else [])
        history = self.histories.get(ident)
        if self.selected is not None and self.selected['id'] == ident:
            history = self.selected
        kept = []
        if pool:
            best = min(pool, key=self._rank)
            kept = [best]
            if history and history['calls'] == 1:
                since = max(self.first_steps[ident],
                            self.active['started'] if self.active is not None and
                            self.active['id'] == ident else -1)
                novel = [row for row in pool if row['step'] > since and self._novel(row, history)]
                second = min(novel, key=self._rank) if novel else None
                if second is not None and second is not best:
                    kept.append(second)
        retained = any(row is e for row in kept)
        if retained:
            path = self.trace_dir / f'candidate-{ident:03d}-s{e["step"]:04d}-m{e["key"][1]:02d}.npz'
            np.savez_compressed(path, rgb=packet['rgb'], depth=packet['depth'],
                mask_bits=e['mask_bits'], mask_shape=e['mask_shape'], pose=e['pose'],
                pitch=packet['pitch'], bounds=e['bounds'], observation=e['step'],
                mask_index=e['key'][1], candidate=ident)
            e['source_file'] = str(path)
        # No RGB/depth/mask arrays in the persistent candidate index.
        self.frames[ident] = [{k: v for k, v in row.items() if k not in
                              ('crop', 'mask_bits', 'mask_shape')} for row in kept]
        retained_keys = {row['key'] for row in kept}
        for old in existing:
            if old['key'] not in retained_keys:
                self._remove_source(old)
        self.event('candidate_image', candidate=ident, observation=e['step'],
            mask_index=e['key'][1], detector_score=e['score'], label=e['label'],
            pixels=e['pixels'], short_side=e['short_side'], pose=e['pose'],
            complete=e['complete'], eligible=self._eligible(e),
            retained=retained, trigger='map_fusion_not_image_score',
            best_key=list(kept[0]['key']) if kept else None)

    def readiness(self):
        # Only first calls. Failed first calls are scheduled separately by the
        # planner after checking all first-call candidates and confirmed routes.
        return sorted((ident, max(e['step'] for e in rows))
            for ident, rows in self.frames.items() if rows and
            self.histories.get(ident, {}).get('calls', 0) == 0 and
            (self.selected is None or self.selected['id'] != ident))

    def inspection_readiness(self):
        return sorted(ident for ident, h in self.histories.items()
            if h['calls'] == 1 and h['decision'] != self.CONFIRMED and
            ident not in self.inspected)

    def _evidence(self, unused=False):
        target = self.selected
        self.evidence_filter_counts = {'trigger': 'map_fusion', 'second_view': bool(target['calls'])}
        if target['calls'] >= self.max_calls:
            return None
        rows = self.frames.get(target['id'], [])
        if target['calls']:
            since = max(self.first_steps[target['id']], target.get('started', -1))
            rows = [e for e in rows if e['step'] > since and self._novel(e, target)]
        if not rows:
            return None
        chosen = min(rows, key=self._rank).copy()
        with np.load(chosen['source_file'], allow_pickle=False) as packet:
            shape = tuple(packet['mask_shape'])
            mask = np.unpackbits(packet['mask_bits'])[:math.prod(shape)].reshape(shape).astype(bool)
            crop = crop_target(packet['rgb'], mask)
            if crop is None:
                raise RuntimeError('Retained crop is missing')
            thumbnail = Image.fromarray(crop[0])
            thumbnail.thumbnail((256, 256))
            chosen['crop'] = np.asarray(thumbnail).copy()
        return chosen

    def begin(self, data):
        if len(data) != 11 or not np.isfinite(data).all() or data[-1] not in (self.FIRST, self.INSPECTION):
            raise ValueError('Fused review requires explicit first/inspection metadata')
        if self.pose is None or self.selected is not None:
            raise RuntimeError('Missing observation or another review is active')
        ident, score, threshold, cx, cy, extent, x0, y0, x1, y1, phase = data
        if ident < 1 or int(ident) != ident or x0 > x1 or y0 > y1 or extent < 0:
            raise ValueError('Invalid candidate identity/geometry')
        ident = int(ident)
        history = self._target_history(ident)
        if phase == self.FIRST and (score < threshold or history['calls']):
            raise ValueError('First review requires fused threshold and no previous call')
        if phase == self.INSPECTION and ident not in self.inspection_readiness():
            raise ValueError('No pending inspection budget')
        self.selected = {'id': ident, 'center': np.array([cx, cy]), 'low': np.array([x0, y0]),
            'high': np.array([x1, y1]), 'extent': float(np.clip(extent, .15, 3.)),
            'blocked_forward': 0, **history}
        self.stats['selected_targets'] += 1
        self.event('selected_target', candidate=ident, fused_confidence=score,
            original_threshold=threshold, center=[cx, cy], extent=extent, bounds=[x0, y0, x1, y1])
        if phase == self.FIRST:
            self.event('fused_review_started', candidate=ident, fused_confidence=score, threshold=threshold)
            decision = self._verify('fused_first')
            if self.selected['calls'] != 1:
                raise RuntimeError('Planner selected first review without a usable image')
            self.first_steps[ident] = self.step
            self.first_decisions[ident] = decision
            if decision != self.CONFIRMED:
                self.event('additional_view_pending', candidate=ident, initial_decision=decision)
                # Provisional navigation state, not a rewritten raw VLM answer.
                return self._complete(self.UNCERTAIN, 'awaiting_conditional_observation')
            return self._complete(decision, 'first_view_confirmed')
        self.inspected.add(ident)
        self.active = self.selected
        self.active.update(radius=observation_radius(extent), started=self.step,
            last_motion=self.step, last_position=np.array(self.pose[:2]))
        self.stats['inspections'] += 1
        self.event('inspection_requested', candidate=ident, radius=self.active['radius'],
            center=[cx, cy], extent=extent, reason='no_reachable_confirmed_alternative',
            initial_decision=self.first_decisions[ident])
        return None

    def _verify(self, phase, evidence=None):
        chosen = evidence if evidence is not None else self._evidence(unused=True)
        if chosen is not None:
            path = self.directory / (f'selected-{self.selected["id"]:03d}-'
                                    f's{chosen["step"]:04d}-m{chosen["key"][1]:02d}.npz')
            shutil.copyfile(chosen['source_file'], path)
            self.event('verification_source', candidate=self.selected['id'], observation=chosen['step'],
                mask_index=chosen['key'][1], source_file=str(path), detector_score=chosen['score'],
                selection='best_new_view' if self.selected['calls'] else 'best_object_history')
        return SelectedTargetReviewer._verify(self, phase, chosen)
