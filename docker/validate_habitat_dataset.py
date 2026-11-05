#!/usr/bin/env python3
"""Load the actual ObjectNav config and exercise a scene with real semantics."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import threading
import time

import habitat
import numpy as np
from hydra import compose, initialize_config_dir
from habitat.config.default import patch_config
from habitat.config.default_structured_configs import (
    HabitatConfigPlugin, register_hydra_plugin,
)
from habitat.sims.habitat_simulator.actions import HabitatSimActions

ROOT = Path(__file__).resolve().parents[1]


class GpuMemorySampler:
    """Sample whole-GPU usage, including other apps and inference transients."""

    def __init__(self):
        self.samples = []
        self.stop = threading.Event()

    def sample(self):
        memory = subprocess.check_output([
            "nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
        ], text=True, timeout=5)
        self.samples.append(int(memory.splitlines()[0]))

    def __enter__(self):
        self.sample()

        def monitor():
            while not self.stop.wait(0.2):
                try:
                    self.sample()
                except (OSError, ValueError, subprocess.SubprocessError):
                    pass

        self.thread = threading.Thread(target=monitor, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(timeout=6)


def check_local_vlm(image):
    """Only send images to loopback, while the Habitat environment stays open."""
    import requests
    from vlm.server_wrapper import api_headers, image_to_str, str_to_bool_arr

    payloads = {
        "blip2": (12182, "blip2itm", {"txt": "a bed in a bedroom"}),
        "dino": (12181, "gdino", {
            "caption": "bed . chair . potted plant . toilet . tv . couch .", "box_threshold": 0.4,
            "text_threshold": 0.25,
        }),
        "sam": (12183, "mobile_sam", {"bbox": [160, 120, 480, 400]}),
        "yolo": (12184, "yolov7", {
            "agnostic_nms": True, "conf_thres": 0.3, "iou_thres": 0.5,
        }),
    }
    session = requests.Session()
    session.trust_env = False
    session.headers.update(api_headers())
    encoded = image_to_str(image)
    results = {}
    for name, (port, route, fields) in payloads.items():
        started = time.monotonic()
        response = session.post(
            f"http://127.0.0.1:{port}/{route}",
            json={"image": encoded, **fields}, timeout=(5, 120),
        )
        response.raise_for_status()
        body = response.json()
        if name == "blip2":
            assert np.isfinite([body["response"], body["itm score"]]).all()
        elif name in ("dino", "yolo"):
            assert len(body["boxes"]) == len(body["logits"]) == len(body["phrases"])
            assert np.asarray(body["boxes"]).size == len(body["boxes"]) * 4
            assert np.isfinite(body["boxes"]).all()
            assert np.isfinite(body["logits"]).all()
        else:
            mask = str_to_bool_arr(body["cropped_mask"], image.shape[:2])
            assert mask.shape == image.shape[:2]
        memory = subprocess.check_output([
            "nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits",
        ], text=True, timeout=5)
        results[name] = {"seconds": round(time.monotonic() - started, 3),
                         "gpu_total_used_mib_after_call": int(memory.splitlines()[0])}
        print(f"[OK] Local {name} actual scene inference while Habitat is open", flush=True)
    session.close()
    return results


def check_camera_tilt(environment, config):
    def rotation(pose):
        q = pose.rotation
        coefficients = np.array([q.real, *q.imag], dtype=float)
        return coefficients / np.linalg.norm(coefficients)

    start = environment.sim.get_agent_state()
    start_rotations = {name: rotation(pose) for name, pose in start.sensor_states.items()}
    down = environment.step(HabitatSimActions.look_down)
    tilted = environment.sim.get_agent_state()
    expected = float(config.habitat.task.actions.look_down.tilt_angle)
    for name, pose in tilted.sensor_states.items():
        dot = np.clip(abs(np.dot(rotation(pose), start_rotations[name])), 0, 1)
        angle = np.degrees(2 * np.arccos(dot))
        if not np.isclose(angle, expected, atol=1e-2):
            raise RuntimeError(f"{name} camera tilt was {angle}, expected {expected}")
    up = environment.step(HabitatSimActions.look_up)
    restored = environment.sim.get_agent_state()
    for name, pose in restored.sensor_states.items():
        if not np.isclose(abs(np.dot(rotation(pose), start_rotations[name])), 1, atol=1e-6):
            raise RuntimeError(f"{name} camera did not restore after look_up")
    np.testing.assert_allclose(restored.position, start.position, atol=1e-6)
    for observation in (down, up):
        if observation["rgb"].shape != (480, 640, 3) or not np.isfinite(observation["depth"]).all():
            raise RuntimeError("Camera tilt returned invalid RGB/depth")
    print(f"[OK] look_down/look_up: {expected} degrees, sensors={list(start_rotations)}", flush=True)
    return {"actions": ["look_down", "look_up"], "degrees": expected,
            "sensors": list(start_rotations), "body_position_unchanged": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-vlm", action="store_true",
                        help="Also exercise all four local model servers with the scene open")
    parser.add_argument("--views", type=int, default=1,
                        help="Number of 30-degree turns/images to test (1-12)")
    args = parser.parse_args()
    if not 1 <= args.views <= 12:
        parser.error("--views must be between 1 and 12")
    os.chdir(ROOT)
    register_hydra_plugin(HabitatConfigPlugin)
    with initialize_config_dir(config_dir=str(ROOT / "config"), version_base=None):
        config = compose(config_name="habitat_eval_hm3dv2")
    with GpuMemorySampler() as memory_sampler, habitat.Env(config=patch_config(config)) as environment:
        observations = environment.reset()
        if observations["rgb"].shape != (480, 640, 3):
            raise RuntimeError("Unexpected RGB sensor shape")
        if not np.isfinite(observations["depth"]).all():
            raise RuntimeError("Invalid depth frame")
        objects = [obj for obj in environment.sim.semantic_annotations().objects if obj is not None]
        if not objects:
            raise RuntimeError("No semantic objects were loaded")
        camera_tilt = check_camera_tilt(environment, config)
        observations = environment.step(HabitatSimActions.turn_left)
        distance = float(environment.get_metrics()["distance_to_goal"])
        if not np.isfinite(distance):
            raise RuntimeError("Goal geodesic distance is not finite")
        summary = {
            "episodes": environment.number_of_episodes,
            "scene": environment.current_episode.scene_id,
            "target": environment.current_episode.object_category,
            "semantic_objects": len(objects),
            "rgb_shape": list(observations["rgb"].shape),
            "goal_distance_after_turn_m": distance,
            "camera_tilt": camera_tilt,
            "purpose": "dataset/simulator validation; not a navigation evaluation score",
        }
        if args.local_vlm:
            views = []
            for index in range(args.views):
                if index:
                    observations = environment.step(HabitatSimActions.turn_left)
                views.append(check_local_vlm(observations["rgb"]))
            summary["local_vlm_views"] = views
            summary["gpu_total_used_mib_max_after_calls"] = max(
                result["gpu_total_used_mib_after_call"]
                for view in views for result in view.values()
            )
            summary["memory_measurement_note"] = (
                "Whole GPU after each HTTP call, including desktop and Habitat; "
                "not an instantaneous peak or worst-case guarantee"
            )
        memory_sampler.sample()
        summary["gpu_total_sampled_peak_mib"] = max(memory_sampler.samples)
        summary["gpu_memory_samples"] = len(memory_sampler.samples)
        summary["gpu_sampling_note"] = "200ms whole-GPU samples; shorter peaks may be missed"
        filename = "dataset-all-local-vlm-validation.json" if args.local_vlm else "dataset-validation.json"
        output = ROOT / "outputs/runtime" / filename
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))
        print("HABITAT_OBJECTNAV_DATA_VALIDATED")


if __name__ == "__main__":
    main()
