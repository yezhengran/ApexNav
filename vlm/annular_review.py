"""Strict per-object top-five evidence and one bounded annular observation.

Map IDs own image association. Detector scores select images; the latest joint
Qwen score ranks unresolved objects. No ground-truth geometry is used.
"""
import math
import shutil
import time

import numpy as np
from PIL import Image

from .active_view import crop_target
from .framing_review import FramingTargetReviewer, object_envelope, framing_distance
from .qwen_review import QwenTransportError


def effective_decision(reply, target):
    answer = reply['answer']
    if answer['quality'] != 'adequate':
        return 'uncertain'
    if answer['decision'] == 'rejected':
        return 'rejected'
    if target == 'potted plant':
        return 'confirmed' if answer['target_confidence'] >= .60 else 'uncertain'
    return answer['decision']


class AnnularTargetReviewer(FramingTargetReviewer):
    def __init__(self, *args, **kwargs):
        backend = kwargs.get('verifier', args[3] if len(args) > 3 else None)
        super().__init__(*args, **kwargs)
        if not getattr(self.cfg, 'annular_review', False) or not callable(getattr(backend, 'review_images', None)):
            raise ValueError('Annular review requires the multi-image Qwen backend')
        self.multiview_backend = backend

    def _retain(self, e, packet):
        if not math.isfinite(e['score']) or not 0 <= e['score'] <= 1:
            raise ValueError('Invalid detector confidence')
        ident = e['candidate']
        existing = self.frames.get(ident, [])
        pool = [row for row in existing if row['key'] != e['key']]
        if self._eligible(e):
            pool.append(e)
        kept = sorted(pool, key=self._rank)[:5]
        retained = any(row is e for row in kept)
        if retained:
            path = self.trace_dir / f'candidate-{ident:03d}-s{e["step"]:04d}-m{e["key"][1]:02d}.npz'
            np.savez_compressed(path, rgb=packet['rgb'], depth=packet['depth'],
                mask_bits=e['mask_bits'], mask_shape=e['mask_shape'], pose=e['pose'],
                pitch=packet['pitch'], bounds=e['bounds'], observation=e['step'],
                mask_index=e['key'][1], candidate=ident, **self._source_metadata(packet))
            e['source_file'] = str(path)
        self.frames[ident] = [{k: v for k, v in row.items() if k not in
                              ('crop', 'mask_bits', 'mask_shape')} for row in kept]
        keys = {row['key'] for row in kept}
        for old in existing:
            if old['key'] not in keys:
                self._remove_source(old)
        self.event('candidate_image', candidate=ident, observation=e['step'],
            mask_index=e['key'][1], detector_score=e['score'], label=e['label'],
            pixels=e['pixels'], short_side=e['short_side'], pose=e['pose'],
            complete=e['complete'], eligible=self._eligible(e), retained=retained,
            trigger='map_fusion_or_detector_gt_0.7' if getattr(self.cfg, 'choice_review', False) else 'map_fusion_not_image_score',
            selection='strict_detector_top5',
            retained_keys=[list(row['key']) for row in kept],
            retained_scores=[row['score'] for row in kept])

    def inspection_readiness(self):
        if self.step >= 400:
            return []
        return sorted(ident for ident, h in self.histories.items()
                      if h['calls'] == 1 and h['decision'] == self.UNCERTAIN and ident not in self.inspected)

    def _inspection_after_first(self, decision):
        return decision == self.UNCERTAIN

    def _advance_inspection(self, previous_pose, action):
        result = super()._advance_inspection(previous_pose, action)
        if self.active is not None and self.step >= 400:
            return self._finish('late_approach_budget')
        return result

    def _evidence(self, unused=False):
        rows = self.frames.get(self.selected['id'], [])
        return self._load(min(rows, key=self._rank)) if rows else None

    def _load(self, row, candidate_id=None):
        candidate_id = self.selected['id'] if candidate_id is None else candidate_id
        chosen = row.copy()
        with np.load(chosen['source_file'], allow_pickle=False) as packet:
            if int(packet['candidate']) != candidate_id or tuple(chosen['key']) != (
                    int(packet['observation']), int(packet['mask_index'])):
                raise RuntimeError('Candidate image lineage mismatch')
            if not np.array_equal(packet['pose'], chosen['pose']) or not np.array_equal(packet['bounds'], chosen['bounds']):
                raise RuntimeError('Candidate image pose/bounds mismatch')
            shape = tuple(packet['mask_shape'])
            mask = np.unpackbits(packet['mask_bits'])[:math.prod(shape)].reshape(shape).astype(bool)
            value = crop_target(packet['rgb'], mask)
            if value is None:
                raise RuntimeError('Missing retained crop')
            if value[1] != chosen['pixels'] or value[2] != chosen['complete']:
                raise RuntimeError('Candidate image mask mismatch')
            thumb = Image.fromarray(value[0]); thumb.thumbnail((256, 256))
            chosen['crop'] = np.asarray(thumb).copy()
        return chosen

    def _verify(self, phase, evidence=None):
        target = self.selected
        if target['calls'] >= 2:
            raise RuntimeError('Per-object review limit exhausted')
        second = target['calls'] == 1
        if second and not target.get('facing_ready', False):
            raise RuntimeError('Joint review requires completed center-facing route')
        rows = self.frames.get(target['id'], [])
        if not 1 <= len(rows) <= 5:
            raise RuntimeError('Missing or unbounded candidate top-five images')
        rows = sorted(rows, key=self._rank)
        selected = [self._load(row) for row in (rows if second else rows[:1])]
        crops, sources = [], []
        for index, e in enumerate(selected):
            stem = f'candidate-{target["id"]:03d}-{phase}-{target["calls"]:02d}-{index:02d}'
            source = self.directory / (stem + '.npz')
            shutil.copyfile(e['source_file'], source)
            crop = self.directory / (stem + '.png')
            Image.fromarray(e['crop']).save(crop)
            crops.append(str(crop))
            row = dict(observation=e['step'], mask_index=e['key'][1], source_file=str(source),
                       crop=str(crop), detector_score=e['score'], pose=e['pose'], complete=e['complete'],
                       bounds=e['bounds'], mask_pixels=e['pixels'], detection_label=e['label'])
            sources.append(row)
            self.event('verification_source', candidate=target['id'], phase=phase, image_index=index,
                selection='strict_detector_top5' if second else 'best_object_history',
                reused=e['key'] in target['used'], **row)
        started = time.monotonic()
        try:
            reply = (self.multiview_backend.review_images([e['crop'] for e in selected], self.label)
                     if second else self.multiview_backend(selected[0]['crop'], self.label))
        except QwenTransportError as error:
            self.event('verification_transport_error', candidate=target['id'], phase=phase,
                       call_id=error.call_id, retryable=error.retryable, crop=crops[0], crops=crops)
            raise
        confidence = reply['answer']['target_confidence']
        if type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError('Invalid joint Qwen target confidence')
        name = effective_decision(reply, self.label)
        decision = {'confirmed': self.CONFIRMED, 'rejected': self.REJECTED, 'uncertain': self.UNCERTAIN}[name]
        self.qwen_scores[target['id']] = float(confidence)
        seconds = time.monotonic() - started
        self.stats['vlm_calls'] += 1; self.stats['vlm_seconds'] += seconds
        target['calls'] += 1
        target['used'].update(e['key'] for e in selected)
        target['last_evidence'] = selected[0]
        self.event('verification', candidate=target['id'], phase=phase, backend='qwen', score=None,
            decision=decision, effective_decision=name, decision_policy='adequate_top5_plant060_v1',
            model_reply=reply, seconds=seconds, crop=crops[0], crops=crops, sources=sources,
            evidence_step=selected[0]['step'], evidence_age=self.step-selected[0]['step'],
            pose=selected[0]['pose'], complete=selected[0]['complete'],
            mask_pixels=selected[0]['pixels'], detection_label=selected[0]['label'],
            image_count=len(crops), new_view_images=sum(e['step'] > self.first_steps.get(target['id'], self.step)
                                                       for e in selected))
        return decision

    def _inspection_geometry(self, extent):
        envelopes, shape = [], None
        for row in self.frames[self.selected['id']]:
            source = self.directory / f'geometry-{self.selected["id"]:03d}-s{row["step"]:04d}-m{row["key"][1]:02d}.npz'
            shutil.copyfile(row['source_file'], source)
            with np.load(source, allow_pickle=False) as packet:
                envelope = object_envelope(packet, self.sensor)
                shape = tuple(packet['mask_shape'])
            self.event('annular_geometry_source', candidate=self.selected['id'], source_file=str(source),
                       observation=row['step'], mask_index=row['key'][1], envelope=envelope)
            if envelope:
                envelopes.append(envelope)
        envelope = None
        if envelopes:
            envelope = dict(low=np.quantile([e['low'] for e in envelopes], .2, axis=0).tolist(),
                            high=np.quantile([e['high'] for e in envelopes], .8, axis=0).tolist(),
                            border_clipped=any(e['border_clipped'] for e in envelopes))
        geometry = framing_distance(extent, envelope, self.camera_height, self.pitch, self.sensor.hfov, shape)
        self.event('inspection_geometry', candidate=self.selected['id'], envelope=envelope,
                   extent=extent, source_count=len(self.frames[self.selected['id']]), **geometry)
        return {**geometry, 'envelope': envelope, 'image_shape': shape, 'geometry_extent': extent}

    def request(self):
        value = super().request()
        return value + [1.] if value else []  # Explicit eight-field annular contract.

    def finish_inspection(self, reachable=True):
        if reachable:
            if self.active is None:
                raise RuntimeError('Missing annular route receipt')
            distance = float(np.linalg.norm(self.active['center']-np.array(self.pose[:2])))
            if distance > self.active['max_radius'] + 1e-5:
                raise RuntimeError('Annular receipt is outside the observation band')
        return super().finish_inspection(reachable)
