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


class ActiveViewVerifier:
    """One historical and at most one fresh active-view call per candidate.

    request() is [instance_id, center_x, center_y, center_radius, extent].
    An empty request cancels navigation. Candidate association is a lightweight
    spatial approximation, not a replacement for ApexNav's semantic map.
    """

    def __init__(self, cfg, sensor, label, verifier, directory, episode):
        self.cfg, self.sensor, self.label = cfg, sensor, label
        if cfg.mode not in ("vlm", "fixed_view", "adaptive_view"):
            raise ValueError("ActiveViewVerifier requires an enabled mode")
        if not math.isfinite(cfg.itm_threshold) or not 0 <= cfg.itm_threshold <= 1:
            raise ValueError("itm_threshold must be finite and in [0, 1]")
        if (not math.isfinite(cfg.association_distance) or cfg.association_distance <= 0 or
                min(cfg.max_candidates, cfg.max_inspection_steps, cfg.max_stationary_steps) < 1):
            raise ValueError("Candidate association and inspection budgets must be positive")
        self.verifier = verifier
        # Keep failed/interrupted attempts separate when the batch resumes.
        episode_dir = Path(directory) / "verification" / f"episode-{episode:04d}"
        episode_dir.mkdir(parents=True, exist_ok=True)
        self.directory = episode_dir / f"attempt-{time.time_ns()}"
        self.directory.mkdir()
        self.candidates = []
        self.active = None
        self.step = 0
        self.current = {}
        self.matched = {}
        self.labels = []
        self.stats = {"vlm_calls": 0, "vlm_seconds": 0.0, "accepted": 0,
                      "rejected": 0, "inspections": 0, "inspection_actions": 0}

    def event(self, event, **values):
        with (self.directory / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"step": self.step, "event": event, **values},
                                    ensure_ascii=False, allow_nan=False) + "\n")

    def _associate(self, center, extent):
        matches = [(np.linalg.norm(c["center"] - center), c) for c in self.candidates]
        if matches:
            distance, candidate = min(matches, key=lambda pair: pair[0])
            if distance <= self.cfg.association_distance:
                return candidate
        if len(self.candidates) >= self.cfg.max_candidates:
            return None
        candidate = {"id": len(self.candidates) + 1, "center": center, "extent": extent,
                     "status": "new", "best": None, "seen": 0}
        self.candidates.append(candidate)
        return candidate

    def _verify(self, candidate, evidence, phase):
        from PIL import Image

        crop_path = self.directory / f"candidate-{candidate['id']:03d}-{phase}.png"
        Image.fromarray(evidence["crop"]).save(crop_path)
        started = time.monotonic()
        # Do not catch service failures: a failed VLM run is not an ablation.
        score = float(self.verifier(evidence["crop"], f"A photo of a {self.label}."))
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("VLM ITM score must be finite and in [0, 1]")
        elapsed = time.monotonic() - started
        self.stats["vlm_calls"] += 1
        self.stats["vlm_seconds"] += elapsed
        accepted = score >= self.cfg.itm_threshold
        self.event("verification", candidate=candidate["id"], phase=phase, score=score,
                   accepted=accepted, seconds=elapsed, crop=str(crop_path),
                   evidence_step=evidence["step"], pose=evidence["pose"],
                   center=candidate["center"].tolist(), extent=candidate["extent"])
        if accepted:
            candidate["status"] = "accepted"
            self.stats["accepted"] += 1
            # Freeze acceptance evidence; latest detections cannot overwrite it.
            candidate["verified"] = evidence
        return accepted

    def _reject(self, candidate, reason):
        candidate["status"] = "rejected"
        candidate["best"] = None
        self.stats["rejected"] += 1
        self.event("rejected", candidate=candidate["id"], reason=reason)
        self.active = None

    def update(self, rgb, observations, masks, scores, labels, pitch, step):
        self.step = step
        self.current = {}
        self.matched = {}
        self.labels = list(labels)
        pose = [-float(observations["gps"][2]), -float(observations["gps"][0]),
                float(observations["compass"][0])]
        for index, (mask, score, label) in enumerate(zip(masks, scores, labels)):
            if label != 0:
                continue
            geometry = target_geometry(mask, observations, self.sensor, pitch)
            crop = crop_target(rgb, mask)
            if geometry is None or crop is None:
                continue
            candidate = self._associate(*geometry)
            if candidate is None:
                continue
            self.matched[index] = candidate
            if candidate["status"] == "rejected":
                continue
            image, area, complete = crop
            evidence = {"crop": image, "score": float(score), "area": area,
                        "complete": complete, "pose": pose, "step": step}
            key = (float(score), area)
            previous = self.current.get(candidate["id"])
            if previous is None or key > (previous["score"], previous["area"]):
                self.current[candidate["id"]] = evidence
            best = candidate["best"]
            if (best is None or score > best["score"] or
                    (complete and not best["complete"] and score >= best["score"] - 0.08)):
                candidate["best"] = evidence
            if candidate.get("last_seen") != step:
                candidate["seen"] += 1
                candidate["last_seen"] = step

        if self.active is not None:
            candidate = self.active
            self.stats["inspection_actions"] += 1
            if np.linalg.norm(np.array(pose[:2]) - candidate["last_position"]) >= 0.05:
                candidate["last_motion"] = step
                candidate["last_position"] = np.array(pose[:2])
            if step - candidate["started"] >= self.cfg.max_inspection_steps:
                self._reject(candidate, "inspection_action_budget")
            elif step - candidate["last_motion"] >= self.cfg.max_stationary_steps:
                self._reject(candidate, "no_translation")

        # Preserve the baseline's initial 26-action panorama. Only one active
        # target is pursued, and at most one new VLM query starts on each step.
        if self.active is None and step >= self.cfg.start_step:
            for candidate in self.candidates:
                if (candidate["status"] != "new" or candidate["seen"] < 2 or
                        candidate["id"] not in self.current):
                    continue
                if not self._verify(candidate, candidate["best"], "historical"):
                    if self.cfg.mode == "vlm":
                        self._reject(candidate, "historical_not_accepted")
                    else:
                        candidate["status"] = "inspecting"
                        candidate["radius"] = observation_radius(
                            candidate["extent"], self.cfg.mode == "adaptive_view")
                        candidate["started"] = candidate["last_motion"] = step
                        candidate["last_position"] = np.array(pose[:2])
                        self.active = candidate
                        self.stats["inspections"] += 1
                        self.event("inspection_requested", candidate=candidate["id"],
                                   center=candidate["center"].tolist(),
                                   radius=candidate["radius"], extent=candidate["extent"])
                break

        return self.accepted_indices()

    def request(self):
        if self.active is None:
            return []
        c = self.active
        return [float(c["id"]), *c["center"].tolist(), c["radius"],
                float(np.clip(c["extent"], 0.15, 3.0))]

    def finish_inspection(self, reachable=True):
        """Only current-frame evidence can confirm arrival; never old best crop."""
        candidate = self.active
        if candidate is None:
            return False
        evidence = self.current.get(candidate["id"])
        accepted = False
        if not reachable:
            reason = "no_safe_viewpoint_or_path"
        elif evidence is None or evidence["step"] != self.step:
            reason = "target_missing_at_viewpoint"
        else:
            accepted = self._verify(candidate, evidence, "active")
            reason = "active_not_accepted"
        if not accepted:
            self._reject(candidate, reason)
        self.active = None
        return accepted

    def accepted_indices(self):
        return [i for i, label in enumerate(self.labels)
                if label != 0 or (i in self.matched and self.matched[i]["status"] == "accepted")]

    def finish_episode(self):
        self.event("episode_end", **self.stats, candidates=len(self.candidates))
        (self.directory / "summary.json").write_text(
            json.dumps(self.stats, indent=2) + "\n", encoding="utf-8")
