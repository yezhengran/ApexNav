"""Selected-target evidence with explicit mapper provenance, never task truth.

Two keyframes per candidate survive detection gaps. The map remains authoritative
for identity; geometry is logged only as a shadow comparison with the old matcher.
"""

from collections import OrderedDict
import math

import numpy as np
from PIL import Image

from .active_view import SelectedTargetReviewer, crop_target, target_geometry


class LinkedTargetReviewer(SelectedTargetReviewer):
    MAX_CANDIDATES = 48
    MAX_PENDING = 4

    def __init__(self, *args, episode_token, **kwargs):
        super().__init__(*args, **kwargs)
        self.episode_token = int(episode_token)
        self.recent_evidence = True
        self.turn_first = False
        self.pending = OrderedDict()
        self.frames = OrderedDict()
        self.histories = {}
        self.last_association = 0
        self.stop_check = False
        self.detector_threshold = 1.0
        self.latest_rows = []
        self.stats['pending_evidence'] = 0
        self.trace_dir = self.directory / 'sources'
        self.trace_dir.mkdir()

    @staticmethod
    def _usable(evidence):
        # Border completeness is logged separately; it cannot prove no occlusion.
        return evidence['pixels'] >= 64 and evidence['short_side'] >= 8

    @staticmethod
    def _rank(e):
        return (e['label'] != 0, not e['complete'], -e['pixels'], -e['score'], -e['step'])

    @staticmethod
    def _novel(e, history):
        if e['key'] in history['used']:
            return False
        previous = history['last_evidence']
        if previous is None:
            return True
        distance = np.linalg.norm(np.array(e['pose'][:2]) - previous['pose'][:2])
        yaw = abs(math.atan2(math.sin(e['pose'][2] - previous['pose'][2]),
                             math.cos(e['pose'][2] - previous['pose'][2])))
        better = e['pixels'] >= 1.25 * previous['pixels'] or (e['complete'] and not previous['complete'])
        return bool(distance >= 0.15 or yaw >= math.radians(15) or better)

    def observe(self, rgb, observations, masks, scores, labels, pitch, step, action=None):
        previous_pose = self.pose
        self.step = step
        self.pose = [-float(observations['gps'][2]), -float(observations['gps'][0]),
                     float(observations['compass'][0])]
        rows = []
        for index, (mask, score, label) in enumerate(zip(masks, scores, labels)):
            crop = crop_target(rgb, mask)
            geometry = target_geometry(mask, observations, self.sensor, pitch)
            if crop is None:
                rows.append(None)  # Preserve ORIGINAL index even if unusable.
                continue
            v, u = np.where(mask)
            thumbnail = Image.fromarray(crop[0])
            thumbnail.thumbnail((256, 256))
            rows.append({'center': geometry[0] if geometry else None,
                         'crop': np.asarray(thumbnail).copy(), 'score': float(score),
                         'label': int(label), 'step': step, 'pose': list(self.pose),
                         'key': (step, index), 'pixels': crop[1], 'complete': crop[2],
                         'short_side': int(min(u.max()-u.min()+1, v.max()-v.min()+1)),
                         'mask_bits': np.packbits(mask), 'mask_shape': mask.shape})
        self.current = [e for e in rows if e is not None]
        self.pending[step] = {'rows': rows, 'rgb': rgb.copy(),
                              'depth': np.asarray(observations['depth']).copy(), 'pitch': pitch}
        while len(self.pending) > self.MAX_PENDING:
            expired, _ = self.pending.popitem(last=False)
            self.event('association_missing', observation=expired, reason='pending_window_exceeded')
        return self._advance_inspection(previous_pose, action)

    def associate(self, token, observation, ids, reasons, bounds):
        if int(token) != self.episode_token:
            self.event('association_ignored', observation=observation, reason='other_episode')
            return
        packet = self.pending.get(observation)
        if packet is None or observation <= self.last_association:
            self.event('association_ignored', observation=observation, reason='duplicate_or_out_of_order')
            return
        rows = packet['rows']
        if len(ids) != len(rows) or len(reasons) != len(rows) or len(bounds) != 4 * len(rows):
            raise RuntimeError('Mapper association does not preserve original mask indices')
        self.last_association = observation
        self.latest_rows = []
        for index, (e, ident, reason) in enumerate(zip(rows, ids, reasons)):
            box = np.asarray(bounds[4*index:4*index+4], dtype=float)
            if ident < 0 or (ident and (reason != 'assigned' or not np.isfinite(box).all())):
                raise RuntimeError('Invalid mapper candidate association')
            old_match = None
            if e is not None and e['center'] is not None and ident:
                outside = np.maximum(np.maximum(box[:2] - e['center'], e['center'] - box[2:]), 0)
                old_match = bool(np.linalg.norm(outside) <= 0.20)
            self.event('source_association', observation=observation, mask_index=index,
                       candidate=int(ident), reason=reason, legacy_match=old_match,
                       usable=bool(e is not None and self._usable(e)), bounds=box.tolist())
            if not ident or e is None:
                continue
            e.update(candidate=int(ident), bounds=box.tolist())
            self.latest_rows.append(e)
            self._retain(e, packet)
        self.pending.pop(observation)
        self.cache = [e for entries in self.frames.values() for e in entries]

    def stop_quality(self):
        reliable, conflicts = [], []
        for ident in {e['candidate'] for e in self.latest_rows}:
            rows = [e for e in self.latest_rows if e['candidate'] == ident]
            target = max((e['score'] for e in rows if e['label'] == 0), default=0.)
            other = max((e['score'] for e in rows if e['label'] != 0), default=0.)
            conflict = other > 0 and other >= target
            if conflict:
                conflicts.append(ident)
            elif any(e['label'] == 0 and self._usable(e) and e['complete'] and
                     e['score'] >= self.detector_threshold for e in rows):
                reliable.append(ident)
        return sorted(reliable), sorted(conflicts)

    def begin(self, data):
        self.stop_check = len(data) == 11 and data[-1] == 1.0
        if len(data) == 11 and (not self.stop_check or not getattr(self.cfg, 'stop_review', False)):
            raise ValueError('STOP review metadata requires the separate enabled B profile')
        if self.stop_check:
            self.event('stop_review_started', candidate=int(data[0]))
        return super().begin(data[:10] if self.stop_check else data)

    def _allow_high_confidence(self):
        return self.stop_check and getattr(self.cfg, 'stop_review', False)

    def _retain(self, e, packet):
        ident = e['candidate']
        existing = self.frames.pop(ident, [])
        pool = existing + [e]
        # Keep the best identity frame and the newest distinct usable view.
        best = min(pool, key=lambda row: (not self._usable(row), self._rank(row)))
        others = [row for row in pool if row['key'] != best['key']]
        latest = max(others, key=lambda row: (self._usable(row), row['step']), default=None)
        kept = [best] + ([latest] if latest is not None else [])
        self.frames[ident] = kept
        if any(row is e for row in kept):
            # Exact local observation for offline association/quality audit only.
            path = self.trace_dir / f'candidate-{ident:03d}-s{e["step"]:04d}-m{e["key"][1]:02d}.npz'
            np.savez_compressed(path, rgb=packet['rgb'], depth=packet['depth'],
                                mask_bits=e['mask_bits'], mask_shape=e['mask_shape'],
                                pose=e['pose'], pitch=packet['pitch'], bounds=e['bounds'],
                                observation=e['step'], mask_index=e['key'][1], candidate=ident)
            e['source_file'] = str(path)
        for old in existing:
            if all(old is not row for row in kept):
                self._remove_source(old)
        while len(self.frames) > self.MAX_CANDIDATES:
            _, removed = self.frames.popitem(last=False)
            for old in removed:
                self._remove_source(old)

    @staticmethod
    def _remove_source(e):
        from pathlib import Path
        if e.get('source_file'):
            Path(e['source_file']).unlink(missing_ok=True)

    def _target_history(self, ident):
        self.retrying = ident in self.histories
        return self.histories.get(ident, super()._target_history(ident)).copy()

    def _may_inspect(self):
        # Initial review keeps the existing bounded observation. A retry consumes
        # only newly available evidence, never another full observation detour.
        return not self.retrying

    def _evidence(self, unused=False):
        target = self.selected
        self.evidence_filter_counts = {'small': 0, 'used_or_same_view': 0,
                                       'other_cluster': 0, 'not_current_for_stop': 0}
        if target['calls'] >= self.max_calls:
            return None
        matches = []
        for e in self.frames.get(target['id'], []):
            if self.stop_check and self.step - e['step'] > 2:
                self.evidence_filter_counts['not_current_for_stop'] += 1
            elif not self._usable(e):
                self.evidence_filter_counts['small'] += 1
            elif not self._novel(e, target):
                self.evidence_filter_counts['used_or_same_view'] += 1
            else:
                matches.append(e)
        return min(matches, key=self._rank) if matches else None

    def readiness(self):
        ready = []
        for ident, entries in self.frames.items():
            history = self.histories.get(ident, super()._target_history(ident))
            if self.selected is not None and self.selected['id'] == ident:
                history = self.selected
            if history['calls'] >= self.max_calls or history.get('decision') in (self.CONFIRMED, self.REJECTED):
                continue
            usable = [e for e in entries if self._usable(e) and self._novel(e, history)]
            if usable:
                ready.append((ident, max(e['step'] for e in usable)))
        return ready

    def _verify(self, phase, evidence=None):
        chosen = evidence if evidence is not None else self._evidence(unused=True)
        if chosen is not None:
            # Copy selected source before bounded keyframe replacement removes it.
            import shutil
            if chosen.get('source_file'):
                shutil.copyfile(chosen['source_file'], self.directory /
                                f'selected-{self.selected["id"]:03d}-s{chosen["step"]:04d}.npz')
            self.event('verification_source', candidate=self.selected['id'],
                       observation=chosen['step'], mask_index=chosen['key'][1],
                       source_file=chosen.get('source_file'))
        return super()._verify(phase, chosen)

    def _complete(self, decision, reason):
        target = self.selected
        if decision == self.UNCERTAIN and target['calls'] == 0:
            decision = self.PENDING_EVIDENCE
        self.histories[target['id']] = {key: target[key] for key in ('used', 'calls', 'last_evidence')}
        if target['last_evidence'] is not None:
            # Novelty checks need no retained RGB/mask/depth after a decision.
            self.histories[target['id']]['last_evidence'] = {
                key: target['last_evidence'][key] for key in ('pose', 'pixels', 'complete')}
        self.histories[target['id']]['decision'] = decision
        return super()._complete(decision, reason)
