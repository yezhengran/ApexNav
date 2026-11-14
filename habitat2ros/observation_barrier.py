"""Optional simulator-only, one-frame/one-map-update transport.

This does not select navigation targets or inspect Habitat task truth. A missing
or rejected receipt is an infrastructure error, never a navigation failure.
"""
import hashlib
import json
import time
from pathlib import Path

import rclpy
from plan_env.msg import NavigationObservation, MultipleMasksWithConfidence
from plan_env.srv import ApplyObservation


class ObservationBarrier:
    def __init__(self, node, directory):
        self.node = node
        self.client = node.create_client(ApplyObservation, '/habitat/apply_observation')
        self.log = Path(directory) / 'navigation-observations.jsonl'
        self.token = 0
        self.next_step = 0

    def begin_episode(self, token):
        self.token = int(token)
        self.next_step = 0

    def apply(self, observations, step, itm=-1., detections=None, action=None, rgb=None):
        if step != self.next_step or not self.token:
            raise RuntimeError('Observation sequence mismatch before publication')
        if not self.client.wait_for_service(timeout_sec=10.):
            raise RuntimeError('Observation barrier unavailable; check matching planner build/options')
        depth, camera, robot = self.node.ros_pub.habitat_publish_ros_topic(observations)
        packet = NavigationObservation(
            episode_token=self.token, observation_id=step, itm_score=float(itm),
            confidence_threshold=float(self.node.fusion_threshold), depth=depth,
            camera_pose=camera, robot_pose=robot,
            detections=detections if detections is not None else MultipleMasksWithConfidence())
        started = time.monotonic()
        future = self.client.call_async(ApplyObservation.Request(observation=packet))
        rclpy.spin_until_future_complete(self.node, future, timeout_sec=10.)
        if not future.done():
            raise RuntimeError('Observation barrier timed out; do not advance or retransmit')
        response = future.result()
        if not response.accepted or (response.episode_token, response.observation_id) != (self.token, step):
            raise RuntimeError('Observation barrier rejected frame: ' + response.reason)
        self.node.wait_for_associations()
        # Hash actual inputs, without wall-clock stamps or task/goal information.
        digest = hashlib.sha256()
        for array in (observations['depth'], rgb if rgb is not None else observations['rgb']):
            digest.update(array.tobytes())
        pose = [float(v) for v in observations['gps']] + [float(observations['compass'][0]),
                                                        float(observations['camera_pitch'])]
        digest.update(json.dumps([pose, float(itm), list(packet.detections.confidence_scores),
                                  list(packet.detections.label_indices)]).encode())
        for cloud in packet.detections.point_clouds:
            digest.update(bytes(cloud.data))
        with self.log.open('a') as stream:
            stream.write(json.dumps({'episode_token': self.token, 'step': step, 'action': action,
                                     'pose': pose, 'itm': float(itm), 'input_sha256': digest.hexdigest(),
                                     'barrier_seconds': time.monotonic()-started})+'\n')
        self.next_step += 1
