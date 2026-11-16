"""Check both CLI/ROS switch states on the installed planner; no Habitat/models."""
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
import unittest
import rclpy
from rcl_interfaces.srv import GetParameters


class FrontierLaunchTest(unittest.TestCase):
    def test_off_and_on_reach_the_real_node(self):
        for enabled in (False, True):
            with self.subTest(enabled=enabled), tempfile.TemporaryDirectory() as directory:
                with open(Path(directory)/'planner.log', 'w') as log:
                    child = subprocess.Popen(['ros2', 'launch', 'exploration_manager',
                        'exploration.launch.py', 'step_sync:=true',
                        f'frontier_look:={str(enabled).lower()}'], stdout=log,
                        stderr=subprocess.STDOUT, start_new_session=True)
                    rclpy.init()
                    node = rclpy.create_node('test_frontier_launch')
                    try:
                        client = node.create_client(GetParameters, '/exploration_node/get_parameters')
                        self.assertTrue(client.wait_for_service(timeout_sec=20.))
                        future = client.call_async(GetParameters.Request(names=['navigation.frontier_look']))
                        rclpy.spin_until_future_complete(node, future, timeout_sec=10.)
                        self.assertTrue(future.done())
                        self.assertEqual(future.result().values[0].bool_value, enabled)
                        self.assertIsNone(child.poll())
                    finally:
                        node.destroy_node()
                        rclpy.shutdown()
                        os.killpg(child.pid, signal.SIGINT)
                        try:
                            child.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL)
                            child.wait()
                # Old ROS services must disappear before the next switch state.
                time.sleep(.3)
