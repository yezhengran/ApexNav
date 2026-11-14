"""Run in the built ROS/venv: exercises the real mapper/FSM, no model or API.

Use a separate ROS_DOMAIN_ID for this test. The launch child is always stopped.
"""
import copy
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
import unittest

import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped
from plan_env.msg import NavigationObservation
from plan_env.srv import ApplyObservation
from std_msgs.msg import Int32


class ObservationBarrierTest(unittest.TestCase):
    def test_real_mapper_receipt_order_and_legacy_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            with open(Path(directory)/'planner.log', 'w') as log:
                child = subprocess.Popen(['ros2', 'launch', 'exploration_manager',
                    'exploration.launch.py', 'step_sync:=true'], stdout=log,
                    stderr=subprocess.STDOUT, start_new_session=True)
                rclpy.init()
                node = rclpy.create_node('test_observation_barrier')
                try:
                    client = node.create_client(ApplyObservation, '/habitat/apply_observation')
                    self.assertTrue(client.wait_for_service(timeout_sec=30.))
                    request = ApplyObservation.Request(observation=NavigationObservation())
                    f = request.observation
                    f.episode_token = time.time_ns()
                    f.itm_score = -1.
                    f.confidence_threshold = .45
                    f.depth = CvBridge().cv2_to_imgmsg(np.full((480, 640), 3., np.float32), '32FC1')
                    f.depth.header.stamp = node.get_clock().now().to_msg()
                    f.robot_pose.header = copy.deepcopy(f.depth.header)
                    f.camera_pose.header = copy.deepcopy(f.depth.header)
                    f.robot_pose.pose.pose.orientation.w = 1.
                    f.camera_pose.pose.pose.orientation.x = 1.
                    f.camera_pose.pose.pose.position.z = .88
                    def call(packet):
                        future = client.call_async(packet)
                        rclpy.spin_until_future_complete(node, future, timeout_sec=10.)
                        self.assertTrue(future.done())
                        return future.result()
                    self.assertTrue(call(request).accepted)
                    self.assertFalse(call(request).accepted)  # Duplicate initial image.
                    future_frame = copy.deepcopy(request)
                    future_frame.observation.observation_id = 1
                    self.assertFalse(call(future_frame).accepted)  # Before any planned action.

                    actions = []
                    node.create_subscription(Int32, '/habitat/plan_action', lambda m: actions.append(m.data), 10)
                    trigger = node.create_publisher(PoseStamped, '/move_base_simple/goal', 10)
                    deadline = time.monotonic() + 10
                    while not actions and time.monotonic() < deadline:
                        trigger.publish(PoseStamped())
                        rclpy.spin_once(node, timeout_sec=.05)
                    self.assertTrue(actions)
                    f.observation_id = 1
                    f.itm_score = .25
                    f.robot_pose.pose.pose.position.x = .25
                    f.camera_pose.pose.pose.position.x = .25
                    state = node.create_publisher(Int32, '/habitat/state', 10)
                    state.publish(Int32(data=1))  # ACTION_EXEC
                    self.assertTrue(call(request).accepted)
                    self.assertFalse(call(request).accepted)
                    bad = copy.deepcopy(request)
                    bad.observation.observation_id = 3
                    self.assertFalse(call(bad).accepted)
                    bad.observation.observation_id = 2
                    bad.observation.camera_pose.header.stamp.sec += 1
                    self.assertFalse(call(bad).accepted)
                    bad = copy.deepcopy(request)
                    bad.observation.observation_id = 2
                    bad.observation.episode_token -= 1
                    self.assertFalse(call(bad).accepted)
                    # Legacy depth/odom publishers cannot update this synchronized mapper.
                    from nav_msgs.msg import Odometry
                    stale = node.create_publisher(Odometry, '/habitat/odom', 10)
                    stale.publish(Odometry())
                    rclpy.spin_once(node, timeout_sec=.1)
                finally:
                    node.destroy_node()
                    rclpy.shutdown()
                    os.killpg(child.pid, signal.SIGINT)
                    try:
                        child.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
            text = (Path(directory)/'planner.log').read_text()
            self.assertEqual(text.count('[Observation applied]'), 2)
            self.assertIn('step=1 position=(0.250000,0.000000)', text)


if __name__ == '__main__':
    unittest.main()
