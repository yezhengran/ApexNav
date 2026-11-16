"""Installed ROS switch and comparison message serialization, without navigation/API."""
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
import unittest
import rclpy
from rcl_interfaces.srv import GetParameters
from plan_env.msg import CandidateComparison, ReviewEvidence

class ChoiceLaunchTest(unittest.TestCase):
    def test_real_launch_and_message_roundtrip(self):
        for enabled in (False,True):
            with self.subTest(enabled=enabled),tempfile.TemporaryDirectory() as directory:
                with open(Path(directory)/'planner.log','w') as log:
                    child=subprocess.Popen(['ros2','launch','exploration_manager','exploration.launch.py',
                        'step_sync:=true','review_enabled:=true','review_lineage:=true','review_automatic:=true',
                        'review_fused:=true','review_confidence:=true','review_annular:=true','review_late_step:=400',
                        f'review_choice:={str(enabled).lower()}',f'review_choice_no_frontier:={str(enabled).lower()}'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    rclpy.init();node=rclpy.create_node('test_choice_launch')
                    try:
                        client=node.create_client(GetParameters,'/exploration_node/get_parameters')
                        self.assertTrue(client.wait_for_service(timeout_sec=30.))
                        future=client.call_async(GetParameters.Request(names=['review.choice','review.annular','navigation.frontier_look','review.choice_no_frontier']))
                        rclpy.spin_until_future_complete(node,future,timeout_sec=10.)
                        self.assertEqual([v.bool_value for v in future.result().values],[enabled,True,False,enabled])
                        received=[]
                        node.create_subscription(CandidateComparison,'/test/comparison',lambda msg:received.append(msg),10)
                        pub=node.create_publisher(CandidateComparison,'/test/comparison',10)
                        msg=CandidateComparison(episode_token=2**62+17,observation_id=400,request_id=1,candidate_ids=[2,4],selected_candidate=4,reason="no_frontier")
                        deadline=time.monotonic()+5
                        while not received and time.monotonic()<deadline:
                            pub.publish(msg);rclpy.spin_once(node,timeout_sec=.05)
                        self.assertEqual(received[-1],msg)
                        self.assertEqual(list(ReviewEvidence(crop_candidate_ids=[2],comparison_candidate_ids=[2,4]).crop_candidate_ids),[2])
                        self.assertIsNone(child.poll())
                    finally:
                        node.destroy_node();rclpy.shutdown();os.killpg(child.pid,signal.SIGINT)
                        try:child.wait(timeout=15)
                        except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                time.sleep(.3)
