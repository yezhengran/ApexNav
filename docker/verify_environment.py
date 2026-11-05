#!/usr/bin/env python3
"""Verify the ApexNav Jazzy container without starting long-running servers."""

from __future__ import annotations

import argparse
import importlib
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(os.environ.get("APEXNAV_ROOT", "/workspace/ApexNav"))
DEPENDENCY_COMMITS = {
    "/opt/apexnav_deps/habitat-sim": "acbe6f49",
    "/opt/apexnav_deps/GroundingDINO": "856dde20",
    "/opt/apexnav_deps/yolov7": "a207844b",
    "/opt/apexnav_deps/MobileSAM": "f706ad9c",
}
WEIGHTS = {
    "mobile_sam.pt": 30_000_000,
    "groundingdino_swint_ogc.pth": 600_000_000,
    "yolov7-e6e.pt": 250_000_000,
}


def report(label: str, value: object) -> None:
    print(f"[OK] {label}: {value}")


def verify_imports() -> None:
    modules = (
        "rclpy",
        "cv_bridge",
        "tf_transformations",
        "habitat",
        "habitat_sim",
        "torch",
        "torchvision",
        "groundingdino._C",
        "mobile_sam",
        "lavis.models",
        "vlm.detector.grounding_dino",
        "vlm.detector.yolov7",
        "vlm.segmentor.sam",
        "vlm.itm.blip2itm",
        "habitat2ros.habitat_publisher",
        "llm.answer",
    )
    for name in modules:
        importlib.import_module(name)
    report("Python/ROS imports", len(modules))


def verify_gpu() -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() is false")
    left = torch.rand((128, 128), device="cuda")
    right = torch.rand((128, 128), device="cuda")
    result = left @ right
    torch.cuda.synchronize()
    report(
        "PyTorch CUDA",
        f"torch={torch.__version__}, device={torch.cuda.get_device_name(0)}, "
        f"mean={result.mean().item():.4f}",
    )


def verify_files() -> None:
    for directory, expected in DEPENDENCY_COMMITS.items():
        actual = subprocess.check_output(
            ["git", "-C", directory, "rev-parse", "--short=8", "HEAD"],
            text=True,
        ).strip()
        if actual != expected:
            raise RuntimeError(f"{directory}: expected {expected}, got {actual}")
        report(Path(directory).name, actual)

    for name, minimum_size in WEIGHTS.items():
        path = PROJECT_ROOT / "data" / name
        size = path.stat().st_size
        if size < minimum_size:
            raise RuntimeError(f"{path}: file is too small ({size} bytes)")
        report(name, f"{size / 1_000_000:.1f} MB")


def verify_egl_render() -> None:
    import habitat_sim
    import numpy as np

    scene = Path(
        "/opt/apexnav_deps/habitat-sim/"
        "data/test_assets/scenes/simple_room.glb"
    )
    if not scene.is_file():
        raise FileNotFoundError(scene)

    simulator_config = habitat_sim.SimulatorConfiguration()
    simulator_config.scene_id = str(scene)
    simulator_config.gpu_device_id = 0
    sensor = habitat_sim.CameraSensorSpec()
    sensor.uuid = "color"
    sensor.sensor_type = habitat_sim.SensorType.COLOR
    sensor.resolution = [64, 64]
    agent = habitat_sim.agent.AgentConfiguration()
    agent.sensor_specifications = [sensor]
    simulator = habitat_sim.Simulator(
        habitat_sim.Configuration(simulator_config, [agent])
    )
    try:
        frame = simulator.get_sensor_observations()["color"]
        if frame.shape != (64, 64, 4) or frame.dtype != np.uint8:
            raise RuntimeError(
                f"unexpected Habitat render: shape={frame.shape}, dtype={frame.dtype}"
            )
        report("Habitat EGL render", f"shape={frame.shape}, dtype={frame.dtype}")
    finally:
        simulator.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-render",
        action="store_true",
        help="skip the real headless Habitat/EGL render",
    )
    args = parser.parse_args()

    report("Python", sys.version.split()[0])
    verify_imports()
    verify_gpu()
    verify_files()
    if not args.skip_render:
        verify_egl_render()
    print("APEXNAV_ENVIRONMENT_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
