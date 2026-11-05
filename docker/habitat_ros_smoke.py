#!/usr/bin/env python3
"""Render real HM3D geometry and check the ROS planner handshake without VLMs."""

import gzip
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import cv2
import habitat_sim
import magnum as mn
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import Image
from std_msgs.msg import Float64, Int32

from habitat2ros.habitat_publisher import ROSPublisherNonNode, SENSOR_QOS

ROOT = Path(__file__).resolve().parents[1]


def main():
    os.environ["ROS_DOMAIN_ID"] = os.environ.get("APEXNAV_ROS_DOMAIN_ID", "79")
    os.environ["ROS_LOCALHOST_ONLY"] = "1"
    content = ROOT / "data/datasets/objectnav/hm3d/v2/val/content"
    files = sorted(content.glob("*.json.gz"))
    if not files:
        raise FileNotFoundError(f"No HM3Dv2 validation task files: {content}")
    with gzip.open(files[0], "rt") as stream:
        episode = json.load(stream)["episodes"][0]
    scene = ROOT / "data/scene_datasets" / episode["scene_id"]
    if not scene.is_file():
        raise FileNotFoundError(scene)

    output = ROOT / "outputs/runtime"
    output.mkdir(parents=True, exist_ok=True)
    config = habitat_sim.SimulatorConfiguration()
    config.scene_id = str(scene)
    # Use official semantic metadata once installed, retaining the geometry-only
    # fallback for diagnosing a machine that has not downloaded annotations yet.
    dataset_config = ROOT / episode.get("scene_dataset_config", "__missing_config__")
    if dataset_config.is_file():
        config.scene_dataset_config_file = str(dataset_config)
    config.gpu_device_id = 0
    specs = []
    for name, kind in [("rgb", habitat_sim.SensorType.COLOR), ("depth", habitat_sim.SensorType.DEPTH)]:
        spec = habitat_sim.CameraSensorSpec()
        spec.uuid, spec.sensor_type = name, kind
        spec.resolution = [480, 640]
        spec.position = mn.Vector3(0.0, 0.88, 0.0)
        spec.hfov = mn.Deg(79.0)
        specs.append(spec)
    agent = habitat_sim.agent.AgentConfiguration()
    agent.sensor_specifications = specs
    simulator = habitat_sim.Simulator(habitat_sim.Configuration(config, [agent]))
    node = None
    planner = None
    try:
        state = habitat_sim.AgentState()
        state.position = episode["start_position"]
        state.rotation = habitat_sim.utils.common.quat_from_coeffs(episode["start_rotation"])
        simulator.get_agent(0).set_state(state)
        observation = simulator.get_sensor_observations()
        rgb = observation["rgb"][:, :, :3]
        depth = np.clip(observation["depth"] / 5.0, 0, 1).astype(np.float32)
        if rgb.shape != (480, 640, 3) or not np.isfinite(depth).all():
            raise RuntimeError("Invalid RGB/depth frame")
        cv2.imwrite(str(output / "smoke-rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        print(f"[OK] HM3D scene rendered: {scene.name}, navmesh={simulator.pathfinder.is_loaded}", flush=True)

        rclpy.init()
        node = rclpy.create_node("apexnav_geometry_smoke")
        publisher = ROSPublisherNonNode(node)
        confidence = node.create_publisher(Float64, "/detector/confidence_threshold", 10)
        trigger = node.create_publisher(PoseStamped, "/move_base_simple/goal", 10)
        states, actions, images = [], [], []
        node.create_subscription(Int32, "/ros/state", lambda msg: states.append(msg.data), 10)
        node.create_subscription(Int32, "/habitat/plan_action", lambda msg: actions.append(msg.data), 10)
        node.create_subscription(Image, "/habitat/camera_depth", lambda msg: images.append(msg.width), SENSOR_QOS)
        with (output / "smoke-planner.log").open("w") as log:
            planner = subprocess.Popen(
                ["ros2", "launch", "exploration_manager", "exploration.launch.py"],
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
        observations = {"rgb": rgb, "depth": depth, "gps": np.zeros(3), "compass": 0.0, "camera_pitch": 0.0}
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            if planner.poll() is not None:
                raise RuntimeError(f"Planner exited ({planner.returncode}); see {output / 'smoke-planner.log'}")
            publisher.habitat_publish_ros_topic(observations)
            confidence.publish(Float64(data=0.5))
            trigger.publish(PoseStamped())
            rclpy.spin_once(node, timeout_sec=0.1)
            if actions and images and any(state > 0 for state in states):
                print(f"[OK] ROS depth messages={len(images)}, states={sorted(set(states))}, action={actions[0]}", flush=True)
                print("HABITAT_ROS_SMOKE_OK (geometry/communication only; no ObjectNav score)")
                return 0
            time.sleep(0.1)
        raise RuntimeError(f"ROS handshake timeout: states={set(states)}, actions={actions}; see {output / 'smoke-planner.log'}")
    finally:
        if planner and planner.poll() is None:
            os.killpg(planner.pid, signal.SIGINT)
            try:
                planner.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(planner.pid, signal.SIGTERM)
                planner.wait(timeout=5)
        if node:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        simulator.close()


if __name__ == "__main__":
    raise SystemExit(main())
