"""Bounded best-view verification; geometry uses observations, never task goals.

The size-to-distance rule and two-stage inspection follow DynZSON f6c2bf0.
BLIP2 ITM is deliberately reused instead of introducing another model.
"""

import json
import math
from pathlib import Path
import time

import numpy as np


def observation_radius(extent, adaptive=True):
    extent = float(np.clip(extent, 0.15, 3.0))
    edge = float(np.clip(0.75 * extent, 0.60, 2.0)) if adaptive else 0.90
    return edge + extent / 2


def target_geometry(mask, observations, sensor, pitch):
    """Robust horizontal bbox in ApexNav world coordinates, with camera pitch."""
    depth = np.asarray(observations["depth"]).squeeze()
    valid = np.asarray(mask, dtype=bool) & np.isfinite(depth)
    if sensor.normalize_depth:
        depth = depth * (sensor.max_depth - sensor.min_depth) + sensor.min_depth
    valid &= (depth > max(0.05, sensor.min_depth)) & (depth < sensor.max_depth * 0.95)
    v, u = np.where(valid)
    if len(u) < 16:
        return None
    # Bound per-detection work without consuming the simulator RNG.
    stride = max(1, len(u) // 2048)
    u, v = u[::stride], v[::stride]
    z = depth[v, u]
    h, w = depth.shape
    fx = w / (2 * math.tan(math.radians(sensor.hfov) / 2))
    # Habitat uses square pixels: vertical FOV follows the aspect ratio.
    forward = z * math.cos(pitch) + (v - h / 2) * z / fx * math.sin(pitch)
    left = -(u - w / 2) * z / fx
    yaw = float(observations["compass"][0])
    gps = observations["gps"]
    xy = np.column_stack((
        forward * math.cos(yaw) - left * math.sin(yaw) - gps[2],
        forward * math.sin(yaw) + left * math.cos(yaw) - gps[0],
    ))
    low, high = np.percentile(xy, [5, 95], axis=0)
    return (low + high) / 2, float(np.max(high - low))


def crop_target(rgb, mask):
    v, u = np.where(mask)
    if len(u) < 16:
        return None
    x0, x1, y0, y1 = int(u.min()), int(u.max()) + 1, int(v.min()), int(v.max()) + 1
    dx, dy = max(2, (x1 - x0) // 5), max(2, (y1 - y0) // 5)
    h, w = rgb.shape[:2]
    crop = rgb[max(0, y0-dy):min(h, y1+dy), max(0, x0-dx):min(w, x1+dx)].copy()
    # Prefer a complete crop at comparable detector confidence.
    complete = x0 > 2 and y0 > 2 and x1 < w-2 and y1 < h-2
    return crop, len(u), complete


class SelectedTargetReviewer:
    """Review only a low-confidence object selected by the original planner.

    Decisions match ObjectMap2D::ReviewState: confirmed=1, uncertain=2,
    rejected=3. Observations never filter or alter the original map input.
    """

    CONFIRMED, UNCERTAIN, REJECTED = 1, 2, 3

    def __init__(self, cfg, sensor, label, verifier, directory, episode):
        self.cfg, self.sensor, self.label, self.verifier = cfg, sensor, label, verifier
        if cfg.mode != 'selected_review':
            raise ValueError('SelectedTargetReviewer requires selected_review mode')
        if not (0 <= cfg.itm_reject_threshold < cfg.itm_confirm_threshold <= 1):
            raise ValueError('Require 0 <= reject < confirm <= 1')
        if min(cfg.max_inspection_steps, cfg.max_stationary_steps) < 1:
            raise ValueError('Inspection budgets must be positive')
        parent = Path(directory) / 'verification' / f'episode-{episode:04d}'
        self.directory = parent / f'attempt-{time.time_ns()}'
        self.directory.mkdir(parents=True)
        self.step, self.current, self.pose = 0, [], None
        self.selected, self.active = None, None
        self.stats = {'selected_targets': 0, 'vlm_calls': 0, 'vlm_seconds': 0.0,
                      'confirmed': 0, 'uncertain': 0, 'rejected': 0,
                      'inspections': 0, 'inspection_actions': 0}

    def event(self, event, **values):
        with (self.directory / 'events.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'step': self.step, 'event': event, **values},
                                    ensure_ascii=False, allow_nan=False) + '\n')

    def observe(self, rgb, observations, masks, scores, labels, pitch, step):
        """Collect current evidence only; return a decision on inspection timeout."""
        self.step = step
        self.pose = [-float(observations['gps'][2]), -float(observations['gps'][0]),
                     float(observations['compass'][0])]
        self.current = []
        for mask, score, label in zip(masks, scores, labels):
            geometry = target_geometry(mask, observations, self.sensor, pitch)
            crop = crop_target(rgb, mask)
            if geometry is None or crop is None:
                continue
            self.current.append({'center': geometry[0], 'extent': geometry[1],
                                 'crop': crop[0], 'score': float(score),
                                 'label': int(label), 'step': step, 'pose': self.pose})
        if self.active is not None:
            target = self.active
            self.stats['inspection_actions'] += 1
            if np.linalg.norm(np.array(self.pose[:2]) - target['last_position']) >= 0.05:
                target['last_motion'] = step
                target['last_position'] = np.array(self.pose[:2])
            if step - target['started'] >= self.cfg.max_inspection_steps:
                return self._complete(self.UNCERTAIN, 'inspection_action_budget')
            if step - target['last_motion'] >= self.cfg.max_stationary_steps:
                return self._complete(self.UNCERTAIN, 'no_translation')
        return None

    def _evidence(self):
        target = self.selected
        matches = []
        for evidence in self.current:
            # Match the current mask to the selected map cluster's footprint.
            # No nearest-centre candidate registry can change target selection.
            center = evidence['center']
            outside = np.maximum(np.maximum(target['low'] - center,
                                            center - target['high']), 0)
            if np.linalg.norm(outside) > 0.20:
                continue
            distance = float(np.linalg.norm(center - target['center']))
            matches.append(((evidence['label'] != 0, distance, -evidence['score']), evidence))
        return min(matches, key=lambda pair: pair[0])[1] if matches else None

    def _verify(self, phase):
        from PIL import Image

        target, evidence = self.selected, self._evidence()
        if evidence is None:
            self.event('verification_missing', candidate=target['id'], phase=phase)
            return self.UNCERTAIN
        crop_path = self.directory / f"candidate-{target['id']:03d}-{phase}.png"
        Image.fromarray(evidence['crop']).save(crop_path)
        started = time.monotonic()
        score = float(self.verifier(evidence['crop'], f'A photo of a {self.label}.'))
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('VLM ITM score must be finite and in [0, 1]')
        seconds = time.monotonic() - started
        self.stats['vlm_calls'] += 1
        self.stats['vlm_seconds'] += seconds
        if score >= self.cfg.itm_confirm_threshold:
            decision = self.CONFIRMED
        elif score <= self.cfg.itm_reject_threshold:
            decision = self.REJECTED
        else:
            decision = self.UNCERTAIN
        self.event('verification', candidate=target['id'], phase=phase, score=score,
                   decision=decision, confirm_threshold=self.cfg.itm_confirm_threshold,
                   reject_threshold=self.cfg.itm_reject_threshold, seconds=seconds,
                   crop=str(crop_path), evidence_step=evidence['step'], pose=evidence['pose'],
                   detection_label=evidence['label'], detection_center=evidence['center'].tolist())
        return decision

    def _complete(self, decision, reason):
        target = self.selected
        name = {self.CONFIRMED: 'confirmed', self.UNCERTAIN: 'uncertain',
                self.REJECTED: 'rejected'}[decision]
        self.stats[name] += 1
        self.event('review_finished', candidate=target['id'], decision=name, reason=reason)
        self.selected = self.active = None
        return target['id'], decision

    def begin(self, data):
        """Metadata: id+1, fused score, threshold, cx, cy, extent, xmin,ymin,xmax,ymax."""
        if self.pose is None:
            raise RuntimeError("A real observation is required before selected-target review")
        if self.selected is not None:
            raise RuntimeError('Another selected target is still being reviewed')
        if len(data) != 10 or not np.isfinite(data).all():
            raise ValueError('Invalid selected-target metadata')
        ident, confidence, threshold, cx, cy, extent, x0, y0, x1, y1 = data
        if ident < 1 or int(ident) != ident or x0 > x1 or y0 > y1 or extent < 0:
            raise ValueError('Invalid selected-target geometry or ID')
        if confidence >= threshold:
            raise ValueError('High-confidence objects must retain the original planner path')
        self.selected = {'id': int(ident), 'center': np.array([cx, cy]),
                         'low': np.array([x0, y0]), 'high': np.array([x1, y1]),
                         'extent': float(np.clip(extent, 0.15, 3.0))}
        self.stats['selected_targets'] += 1
        self.event('selected_target', candidate=int(ident), fused_confidence=confidence,
                   original_threshold=threshold, center=[cx, cy], extent=extent,
                   bounds=[x0, y0, x1, y1])
        decision = self._verify('selected')
        if decision != self.UNCERTAIN:
            return self._complete(decision, 'selected_view_decisive')
        self.active = self.selected
        self.active.update(radius=observation_radius(extent), started=self.step,
                           last_motion=self.step, last_position=np.array(self.pose[:2]))
        self.stats['inspections'] += 1
        self.event('inspection_requested', candidate=int(ident),
                   radius=self.active['radius'], center=[cx, cy], extent=extent)
        return None

    def request(self):
        if self.active is None:
            return []
        target = self.active
        return [float(target['id']), *target['center'].tolist(), target['radius'], target['extent']]

    def finish_inspection(self, reachable=True):
        if self.active is None:
            raise RuntimeError('No selected target is under inspection')
        if not reachable:
            return self._complete(self.UNCERTAIN, 'no_safe_viewpoint_or_path')
        decision = self._verify('active')
        return self._complete(decision, 'active_view_completed')

    def finish_episode(self):
        if self.selected is not None:
            self._complete(self.UNCERTAIN, 'episode_ended_during_inspection')
        self.event('episode_end', **self.stats)
        (self.directory / 'summary.json').write_text(json.dumps(self.stats, indent=2) + '\n')
