#!/usr/bin/env python3
"""Regression for Lab camera pitch with both Habitat-Sim sensor suite APIs."""

from types import SimpleNamespace
import unittest

import magnum as mn
import numpy as np
from habitat.tasks.nav.nav import NavigationMovementAgentAction


class CameraTiltTest(unittest.TestCase):
    def check_suite(self, name):
        nodes = [SimpleNamespace(rotation=mn.Quaternion()) for _ in range(2)]
        suite = {key: SimpleNamespace(node=node) for key, node in zip(("rgb", "depth"), nodes)}
        agent = SimpleNamespace(**{name: suite})
        action = object.__new__(NavigationMovementAgentAction)
        action._sim = SimpleNamespace(agents=[agent])
        action._move_camera_vertical(-30)
        for node in nodes:
            self.assertGreater(np.linalg.norm(np.asarray(node.rotation.vector)), 0.1)
        action._move_camera_vertical(30)
        for node in nodes:
            np.testing.assert_allclose(np.asarray(node.rotation.vector), 0, atol=1e-6)
            self.assertAlmostEqual(float(node.rotation.scalar), 1.0, places=6)

    def test_current_private_suite(self):
        self.check_suite("_sensors")

    def test_public_suite(self):
        self.check_suite("sensors")


if __name__ == "__main__":
    unittest.main(verbosity=2)
