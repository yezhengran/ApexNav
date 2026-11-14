"""Opt-in center-facing review with a singleton best image per map object."""
import math
import shutil

import numpy as np
from PIL import Image

from .active_view import SelectedTargetReviewer, crop_target
from .fused_review import FusedTargetReviewer


class ConfidenceTargetReviewer(FusedTargetReviewer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not getattr(self.cfg, 'confidence_review', False) or self.cfg.backend != 'qwen':
            raise ValueError('Confidence review requires its Qwen profile')
        self.qwen_scores = {}
        verifier = self.verifier

        def scored(image, target):
            reply = verifier(image, target)
            value = reply['answer']['target_confidence']
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('Invalid target confidence')
            # Latest completed answer, not the maximum over repeated queries.
            self.qwen_scores[self.selected['id']] = float(value)
            return reply
        self.verifier = scored

    def _retain(self, e, packet):
        if not math.isfinite(e['score']) or not 0 <= e['score'] <= 1:
            raise ValueError('Invalid detector confidence')
        ident = e['candidate']
        existing = self.frames.get(ident, [])
        pool = existing + ([e] if self._eligible(e) else [])
        best = min(pool, key=self._rank) if pool else None
        retained = best is e
        if retained:
            path = self.trace_dir / f'candidate-{ident:03d}-s{e["step"]:04d}-m{e["key"][1]:02d}.npz'
            np.savez_compressed(path, rgb=packet['rgb'], depth=packet['depth'],
                mask_bits=e['mask_bits'], mask_shape=e['mask_shape'], pose=e['pose'],
                pitch=packet['pitch'], bounds=e['bounds'], observation=e['step'],
                mask_index=e['key'][1], candidate=ident, **self._source_metadata(packet))
            e['source_file'] = str(path)
            # Historical thumbnails are audit/presentation artifacts. Only one
            # NPZ is live in the selection index for each candidate.
            archive = self.directory / 'best-history'
            archive.mkdir(exist_ok=True)
            thumb = Image.fromarray(e['crop']); thumb.thumbnail((256, 256))
            png = archive / (path.stem + '.png'); thumb.save(png)
            self.event('best_crop_updated', candidate=ident, observation=e['step'],
                mask_index=e['key'][1], detector_score=e['score'], crop=str(png))
        self.frames[ident] = [{k: v for k, v in best.items() if k not in
                              ('crop', 'mask_bits', 'mask_shape')}] if best else []
        for old in existing:
            if best is None or old['key'] != best['key']:
                self._remove_source(old)
        self.event('candidate_image', candidate=ident, observation=e['step'],
            mask_index=e['key'][1], detector_score=e['score'], label=e['label'],
            pixels=e['pixels'], short_side=e['short_side'], pose=e['pose'],
            complete=e['complete'], eligible=self._eligible(e), retained=retained,
            trigger='map_fusion_not_image_score', best_key=list(best['key']) if best else None)

    def _source_metadata(self, packet):
        return {}

    def _evidence(self, unused=False):
        self.evidence_filter_counts = {'selection': 'singleton_best_history'}
        if self.selected['calls'] >= self.max_calls:
            return None
        rows = self.frames.get(self.selected['id'], [])
        if not rows:
            return None
        assert len(rows) == 1
        chosen = rows[0].copy()
        with np.load(chosen['source_file'], allow_pickle=False) as packet:
            shape = tuple(packet['mask_shape'])
            mask = np.unpackbits(packet['mask_bits'])[:math.prod(shape)].reshape(shape).astype(bool)
            crop = crop_target(packet['rgb'], mask)
            if crop is None:
                raise RuntimeError('Retained crop is missing')
            thumb = Image.fromarray(crop[0]); thumb.thumbnail((256, 256))
            chosen['crop'] = np.asarray(thumb).copy()
        return chosen

    def _verify(self, phase, evidence=None):
        if self.selected['calls'] and not self.selected.get('facing_ready', False):
            raise RuntimeError('Second review requires center-facing observation')
        chosen = evidence if evidence is not None else self._evidence()
        if chosen is not None:
            path = self.directory / (f'selected-{self.selected["id"]:03d}-'
                                    f's{chosen["step"]:04d}-m{chosen["key"][1]:02d}.npz')
            shutil.copyfile(chosen['source_file'], path)
            self.event('verification_source', candidate=self.selected['id'], observation=chosen['step'],
                mask_index=chosen['key'][1], source_file=str(path), detector_score=chosen['score'],
                selection='best_object_history', reused=chosen['key'] in self.selected['used'])
        return SelectedTargetReviewer._verify(self, phase, chosen)

    def _advance_inspection(self, previous_pose, action):
        if self.active is None:
            return None
        target = self.active
        self.stats['inspection_actions'] += 1
        motion = float(np.linalg.norm(np.array(self.pose[:2]) - previous_pose[:2]))
        self.event('inspection_evidence', candidate=target['id'], action=action,
            displacement=motion, review_deferred='awaiting_viewpoint_and_center_facing')
        if np.linalg.norm(np.array(self.pose[:2]) - target['last_position']) >= .05:
            target['last_motion'] = self.step
            target['last_position'] = np.array(self.pose[:2])
        if self.step - target['started'] >= self.cfg.max_inspection_steps:
            return self._finish('inspection_action_budget')
        if self.step - target['last_motion'] >= self.cfg.max_stationary_steps:
            return self._finish('no_translation')
        return None

    def _finish(self, reason):
        # No final API request on route failure or budget exhaustion: the
        # center-facing gate has not been met. Keep the prior numeric score.
        return self._complete(self.UNCERTAIN, reason)

    def finish_inspection(self, reachable=True):
        if not reachable:
            return self._finish('no_safe_viewpoint')
        target = self.active
        if target is None or self.step <= target['started'] or self.last_association != self.step:
            raise RuntimeError('Missing post-motion mapped observation')
        delta = target['center'] - np.array(self.pose[:2])
        yaw = math.atan2(delta[1], delta[0])
        error = abs(math.atan2(math.sin(yaw-self.pose[2]), math.cos(yaw-self.pose[2])))
        if error > math.radians(18) + 1e-6:
            raise RuntimeError('Planner inspection receipt is not facing the candidate center')
        target['facing_ready'] = True
        self.event('inspection_facing_ready', candidate=target['id'], center=target['center'].tolist(),
            pose=self.pose, yaw_error_degrees=math.degrees(error), observation=self.step)
        return self._complete(self._verify('center_facing'), 'center_facing_second_review')
