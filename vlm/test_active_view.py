"""Small observation-chain regressions; real projection/crops, no simulator GT."""

import json
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from vlm.active_view import SelectedTargetReviewer
from vlm.linked_review import LinkedTargetReviewer


class ReviewerTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.rgb = np.full((64, 64, 3), 100, dtype=np.uint8)
        self.mask = np.zeros((64, 64), dtype=bool)
        self.mask[24:40, 24:40] = True

    def reviewer(self, recent=True, turn=False, score=0.8, linked=False, **budgets):
        cfg = SimpleNamespace(mode='selected_review', recent_evidence=recent,
                              turn_first=turn, itm_confirm_threshold=0.65,
                              itm_reject_threshold=0.20, max_inspection_steps=48,
                              max_stationary_steps=24)
        for key, value in budgets.items():
            setattr(cfg, key, value)
        reviewer = LinkedTargetReviewer if linked else SelectedTargetReviewer
        return reviewer(cfg, SimpleNamespace(normalize_depth=False,
                                      min_depth=0.1, max_depth=10, hfov=90),
                                      'chair', lambda image, text: score,
                                      self.directory.name, 1,
                                      **({'episode_token': 123} if linked else {}))

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

    def assign(self, view, ids=None, token=123, step=None):
        step = view.step if step is None else step
        ids = ids if ids is not None else [1] * len(view.pending[step]['rows'])
        # Deliberately disagree with full-mask center: the actual mapper owns ID.
        view.associate(token, step, ids, ['assigned' if i else 'filtered' for i in ids],
                       [20.0, 20.0, 20.0, 20.0] * len(ids))

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

    def test_linked_source_survives_54_step_gap_and_geometry_disagreement(self):
        view = self.reviewer(linked=True)
        self.observe(view, 388)
        self.assign(view)
        target = self.target(view)
        self.observe(view, 442, detected=False)
        self.assign(view, [])
        self.assertEqual(view.begin(target), (1, view.CONFIRMED))
        event = next(e for e in self.events(view) if e['event'] == 'verification')
        self.assertEqual(event['evidence_age'], 54)
        self.assertFalse(next(e for e in self.events(view) if e['event'] == 'source_association')['legacy_match'])

    def test_linked_identity_never_uses_a_nearby_different_candidate(self):
        view = self.reviewer(linked=True)
        self.observe(view, 1)
        self.assign(view, [2])
        self.assertIsNone(view.begin(self.target(view)))
        self.assertEqual(view.finish_inspection(False), (1, view.PENDING_EVIDENCE))
        self.assertEqual(view.stats['vlm_calls'], 0)

    def test_linked_missing_mask_keeps_original_indices(self):
        view = self.reviewer(linked=True)
        obs = {'gps': np.zeros(3), 'compass': np.zeros(1), 'depth': np.full((64,64,1),2.)}
        view.observe(self.rgb, obs, [np.zeros_like(self.mask), self.mask], [.4,.4], [0,0],0.,1)
        self.assign(view, [0,7])
        self.assertEqual(list(view.frames), [7])
        self.assertEqual(view.frames[7][0]['key'], (1,1))
        self.assertEqual(len(view.frames[7]), 1)

    def test_linked_rejects_wrong_episode_and_malformed_association(self):
        view = self.reviewer(linked=True)
        self.observe(view, 1)
        self.assign(view, token=124)
        self.assertEqual(view.frames, {})
        with self.assertRaises(RuntimeError):
            self.assign(view, [])
        self.assign(view)
        self.assign(view, ids=[1], step=1)
        self.assertEqual(len(view.frames[1]), 1)

    def test_pending_retries_only_on_new_evidence_without_second_detour(self):
        view = self.reviewer(linked=True, score=.4)
        self.observe(view, 1, detected=False)
        self.assign(view, [])
        target = self.target(view)
        view.begin(target)
        self.assertEqual(view.finish_inspection(False), (1,view.PENDING_EVIDENCE))
        self.assertEqual(view.readiness(), [])
        self.observe(view, 55)
        self.assign(view)
        self.assertEqual(view.readiness(), [(1,55)])
        self.assertEqual(view.begin(target), (1,view.UNCERTAIN))
        self.assertEqual(view.stats['inspections'], 1)
        self.assertEqual(view.readiness(), [])
        self.observe(view, 56, x=.2)
        self.assign(view)
        self.assertEqual(view.begin(target), (1,view.UNCERTAIN))
        self.assertEqual(view.stats['vlm_calls'], 2)
        self.observe(view, 57, x=.4)
        self.assign(view)
        self.assertEqual(view.readiness(), [])

    def test_linked_per_candidate_and_total_image_memory_bounded(self):
        view = self.reviewer(linked=True)
        for step in range(1,105):
            self.observe(view, step)
            self.assign(view, [(step+1)//2])
        self.assertEqual(len(view.frames), 48)
        self.assertTrue(all(len(frames)<=2 for frames in view.frames.values()))
        self.assertLessEqual(len(list(view.trace_dir.glob('*.npz'))), 96)

    def test_linked_pending_does_not_become_semantic_rejection(self):
        view = self.reviewer(linked=True, score=.1)
        self.observe(view, 1, detected=False)
        self.assign(view, [])
        view.begin(self.target(view))
        self.assertEqual(view.finish_inspection(False), (1,view.PENDING_EVIDENCE))
        self.assertEqual(view.stats['rejected'], 0)

    def test_stop_extension_requires_b_and_explicit_stop_metadata(self):
        for enabled in (False, True):
            view = self.reviewer(linked=True, stop_review=enabled)
            self.observe(view, 1)
            self.assign(view)
            target = self.target(view)
            target[1] = .9
            with self.assertRaises(ValueError):
                view.begin(target)
            if enabled:
                self.assertEqual(view.begin(target + [1.]), (1,view.CONFIRMED))
            else:
                with self.assertRaises(ValueError):
                    view.begin(target + [1.])

    def test_stop_quality_requires_current_same_id_unconflicted_target(self):
        view = self.reviewer(linked=True, stop_review=True)
        view.detector_threshold = .35
        self.observe(view, 1)
        self.assign(view)
        self.assertEqual(view.stop_quality(), ([1], []))
        self.observe(view, 2, detected=False)
        self.assign(view, [])
        self.assertEqual(view.stop_quality(), ([], []))  # Stored old keyframe is not arrival evidence.
        obs = {'gps': np.zeros(3), 'compass': np.zeros(1), 'depth': np.full((64,64,1),2.)}
        view.observe(self.rgb,obs,[self.mask,self.mask],[.4,.6],[0,1],0.,3)
        self.assign(view,[1,1])
        self.assertEqual(view.stop_quality(), ([], [1]))

    def test_old_semantic_keyframe_cannot_confirm_stop_without_current_evidence(self):
        view = self.reviewer(linked=True, stop_review=True)
        self.observe(view,1)
        self.assign(view)
        target = self.target(view)
        target[1] = .9
        self.observe(view,55,detected=False)
        self.assign(view,[])
        self.assertIsNone(view.begin(target+[1.]))
        self.assertEqual(view.stats['vlm_calls'],0)
        self.assertEqual(view.finish_inspection(False),(1,view.PENDING_EVIDENCE))


if __name__ == '__main__':
    unittest.main(verbosity=2)
