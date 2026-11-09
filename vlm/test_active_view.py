"""Small observation-chain regressions; real projection/crops, no simulator GT."""

import json
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from vlm.active_view import SelectedTargetReviewer


class ReviewerTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.rgb = np.full((64, 64, 3), 100, dtype=np.uint8)
        self.mask = np.zeros((64, 64), dtype=bool)
        self.mask[24:40, 24:40] = True

    def reviewer(self, recent=True, turn=False, score=0.8, **budgets):
        cfg = SimpleNamespace(mode='selected_review', recent_evidence=recent,
                              turn_first=turn, itm_confirm_threshold=0.65,
                              itm_reject_threshold=0.20, max_inspection_steps=48,
                              max_stationary_steps=24)
        for key, value in budgets.items():
            setattr(cfg, key, value)
        return SelectedTargetReviewer(cfg, SimpleNamespace(normalize_depth=False,
                                      min_depth=0.1, max_depth=10, hfov=90),
                                      'chair', lambda image, text: score,
                                      self.directory.name, 1)

    def observe(self, view, step, detected=True, action=2, x=0.0, partial=False):
        mask = self.mask.copy()
        if partial:
            mask[:, :] = False
            mask[24:40, :16] = True
        observations = {'gps': np.array([0, 0, -x]), 'compass': np.array([0.0]),
                        'depth': np.full((64, 64, 1), 2.0)}
        return view.observe(self.rgb, observations, [mask] if detected else [],
                            [0.4] if detected else [], [0] if detected else [],
                            0.0, step, action)

    def target(self, view):
        center = view.current[0]['center'] if view.current else np.array([2.0, 0.0])
        return [1, 0.3, 0.5, *center, 1.0, *(center - 0.5), *(center + 0.5)]

    def events(self, view):
        return [json.loads(line) for line in (view.directory / 'events.jsonl').read_text().splitlines()]

    def test_recent_evidence_survives_detection_gap_legacy_does_not(self):
        for recent in (False, True):
            view = self.reviewer(recent=recent)
            self.observe(view, 1)
            target = self.target(view)
            self.observe(view, 2, detected=False)
            result = view.begin(target)
            self.assertEqual(result, (1, view.CONFIRMED) if recent else None)
            self.assertEqual(view.stats['vlm_calls'], int(recent))
            if not recent:
                self.assertEqual(view.finish_inspection(False), (1, view.UNCERTAIN))

    def test_stale_or_other_cluster_never_supplies_evidence(self):
        for stale in (False, True):
            view = self.reviewer()
            self.observe(view, 1)
            target = self.target(view)
            if not stale:
                target[3:5] = [20, 20]
                target[6:10] = [19, 19, 21, 21]
            self.observe(view, 22 if stale else 2, detected=False)
            self.assertIsNone(view.begin(target))
            self.assertEqual(view.finish_inspection(False), (1, view.UNCERTAIN))
            self.assertEqual(view.stats['vlm_calls'], 0)

    def test_during_view_decision_precedes_action_budget(self):
        view = self.reviewer(max_inspection_steps=1)
        self.observe(view, 1, detected=False)
        view.begin(self.target(view))
        self.assertEqual(self.observe(view, 2), (1, view.CONFIRMED))
        self.assertEqual(view.stats['inspection_actions'], 1)
        self.assertEqual(self.events(view)[-1]['reason'], 'during_view_decisive')

    def test_partial_negative_is_uncertain_complete_negative_rejects(self):
        for partial in (False, True):
            view = self.reviewer(score=0.1)
            self.observe(view, 1, partial=partial)
            result = view.begin(self.target(view))
            self.assertEqual(result, None if partial else (1, view.REJECTED))
            event = next(e for e in self.events(view) if e['event'] == 'verification')
            self.assertEqual(event['partial_negative_withheld'], partial)

    def test_duplicate_view_cannot_spend_second_call(self):
        view = self.reviewer(score=0.4)
        self.observe(view, 1)
        view.begin(self.target(view))
        for step in range(2, 6):
            self.assertIsNone(self.observe(view, step))
        self.assertEqual(view.stats['vlm_calls'], 1)
        self.assertEqual(self.observe(view, 6, x=0.2), (1, view.UNCERTAIN))
        self.assertEqual(view.stats['vlm_calls'], 2)
        self.assertEqual(self.events(view)[-1]['reason'], 'verification_budget')

    def test_turns_do_not_count_as_blocked_forward_and_motion_resets_count(self):
        view = self.reviewer(turn=True)
        self.observe(view, 1, detected=False)
        view.begin(self.target(view))
        self.assertEqual(view.request()[-1], 1.0)
        for step in (2, 3, 4):
            self.assertIsNone(self.observe(view, step, detected=False, action=2))
        for step in (5, 6):
            self.assertIsNone(self.observe(view, step, detected=False, action=1))
        self.assertIsNone(self.observe(view, 7, detected=False, action=1, x=0.25))
        for step in (8, 9):
            self.assertIsNone(self.observe(view, step, detected=False, action=1, x=0.25))
        self.assertEqual(self.observe(view, 10, detected=False, action=1, x=0.25),
                         (1, view.UNCERTAIN))
        self.assertEqual(self.events(view)[-1]['reason'], 'blocked_forward')
        self.assertEqual(view.stats['vlm_calls'], 0)

    def test_cache_bounded_and_original_input_unchanged(self):
        view = self.reviewer()
        rgb, mask = self.rgb.copy(), self.mask.copy()
        for step in range(60):
            self.observe(view, step)
        self.assertLessEqual(len(view.cache), 48)
        self.assertTrue(all(view.step - e['step'] <= 20 for e in view.cache))
        np.testing.assert_array_equal(self.rgb, rgb)
        np.testing.assert_array_equal(self.mask, mask)

    def test_high_confidence_cannot_enter_review(self):
        view = self.reviewer()
        self.observe(view, 1)
        target = self.target(view)
        target[1] = target[2]
        with self.assertRaises(ValueError):
            view.begin(target)
        self.assertEqual(view.stats['vlm_calls'], 0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
