"""Plant-only acceptance, provenance and real reviewer flow; no provider calls."""
import itertools
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from docker.run_batch import evaluation_command
from vlm.framing_review import FramingTargetReviewer
from vlm.qwen_review import QwenReviewer
from vlm.test_qwen_review import response


class PlantConfidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ledger = Path(self.tmp.name) / 'budget.json'
        self.ledger.write_text(json.dumps({'limit_cny': 20, 'initial_spent_cny': 0, 'calls': {}}))
        self.rgb = np.zeros((64, 64, 3), np.uint8)
        self.client = self.make_client(.6)

    def make_client(self, threshold, scores=True):
        with patch('vlm.qwen_review.read_credentials', return_value='fake-key'):
            return QwenReviewer(self.ledger, target_confidence=scores,
                                plant_confirm_threshold=threshold)

    def data(self, score, decision='uncertain', quality='adequate'):
        data = response(decision, quality)
        answer = json.loads(data['choices'][0]['message']['content'])
        answer.update(target_confidence=score, visible_category='potted plant')
        data['choices'][0]['message']['content'] = json.dumps(answer)
        return data

    def call(self, data, target='potted plant', client=None):
        with patch('vlm.qwen_review.requests.post', return_value=Mock(
                status_code=200, json=lambda: data)) as post:
            result = (client or self.client)(self.rgb, target)
        self.assertEqual(post.call_count, 1)
        return result, post.call_args.kwargs

    def test_exact_threshold_and_boundary_override_raw_label_and_quality(self):
        for score, raw, quality in itertools.product(
                [math.nextafter(.6, 0), .6, math.nextafter(.6, 1), 1.0],
                ['confirmed', 'uncertain', 'rejected'], ['adequate', 'insufficient']):
            with self.subTest(score=score, raw=raw, quality=quality):
                result, _ = self.call(self.data(score, raw, quality))
                expected = ('confirmed' if score >= .6 else
                            'rejected' if raw == 'rejected' and quality == 'adequate' else 'uncertain')
                self.assertEqual(result['decision'], expected)
                self.assertEqual(result['answer']['decision'], raw)
                self.assertEqual(result['answer']['quality'], quality)
                bill = json.loads(self.ledger.read_text())['calls'][result['call_id']]
                self.assertEqual(bill['effective_decision'], expected)
                self.assertEqual(bill['decision_policy'], 'plant_target_confidence_v1')
                self.assertEqual(bill['plant_confirm_threshold'], .6)
                self.assertEqual(bill['answer'], result['answer'])

    def test_other_targets_keep_labels_and_quality_at_all_scores(self):
        for target, score, raw, quality in itertools.product(
                ['chair', 'tv', 'bed', 'couch', 'toilet'], [.1, .6, .99],
                ['confirmed', 'uncertain', 'rejected'], ['adequate', 'insufficient']):
            with self.subTest(target=target, score=score, raw=raw, quality=quality):
                result, _ = self.call(self.data(score, raw, quality), target)
                self.assertEqual(result['decision'], raw if quality == 'adequate' else 'uncertain')
                self.assertNotIn('decision_policy', result)

    def test_disabled_policy_keeps_legacy_plant_semantics_and_identical_request(self):
        legacy = self.make_client(None)
        for score, raw, quality in [(0.9, 'uncertain', 'adequate'),
                                    (.1, 'confirmed', 'adequate'),
                                    (.9, 'confirmed', 'insufficient')]:
            data = self.data(score, raw, quality)
            old, old_request = self.call(data, client=legacy)
            new, new_request = self.call(data)
            self.assertEqual(old['decision'], raw if quality == 'adequate' else 'uncertain')
            self.assertNotIn('decision_policy', old)
            self.assertEqual(old_request, new_request)
            self.assertEqual(old['answer'], new['answer'])
            self.assertEqual(old['charged_cny'], new['charged_cny'])
            self.assertEqual(old['image_sha256'], new['image_sha256'])
            self.assertEqual(old['prompt_sha256'], new['prompt_sha256'])
        no_score = self.make_client(None, scores=False)
        result, _ = self.call(response(), client=no_score)
        self.assertEqual(result['decision'], 'confirmed')

    def test_invalid_score_or_missing_field_still_stops(self):
        for value in [None, True, '0.9', float('nan'), float('inf'), -.1, 1.1]:
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                self.call(self.data(value))
        with self.assertRaises(RuntimeError):
            self.call(response())
        self.assertTrue(all(b['state'] == 'error' for b in json.loads(self.ledger.read_text())['calls'].values()))

    def test_bad_policy_fails_before_credentials_or_http(self):
        with patch('vlm.qwen_review.read_credentials') as credentials, patch('vlm.qwen_review.requests.post') as post:
            for value in [True, '0.6', float('nan'), float('inf'), -.1, 1.1]:
                with self.subTest(value=value), self.assertRaises(ValueError):
                    QwenReviewer(self.ledger, target_confidence=True, plant_confirm_threshold=value)
            with self.assertRaises(ValueError):
                QwenReviewer(self.ledger, plant_confirm_threshold=.6)
            credentials.assert_not_called()
            post.assert_not_called()

    def test_only_q6_qwen_profile_enables_policy_in_effective_config(self):
        for profile, backend in itertools.product(
                ['linked_stop_preserve', 'linked_early', 'linked_fused', 'linked_confidence', 'linked_framing'],
                ['blip2', 'qwen']):
            cmd = evaluation_command('hm3dv2', 1, '/tmp/plant-test', False,
                                     method='selected_review', review_profile=profile, review_backend=backend)
            flags = [s for s in cmd if 'qwen_plant_confirm_threshold' in s]
            self.assertEqual(flags, ['+active_view.qwen_plant_confirm_threshold=0.6']
                             if (profile, backend) == ('linked_framing', 'qwen') else [])

    def test_real_framing_reviewer_confirms_or_keeps_conditional_observation(self):
        cfg = SimpleNamespace(mode='selected_review', fused_review=True, confidence_review=True,
                              framing_review=True, backend='qwen', stop_review=False,
                              itm_confirm_threshold=.65, itm_reject_threshold=.2,
                              max_inspection_steps=48, max_stationary_steps=24)
        sensor = SimpleNamespace(normalize_depth=False, min_depth=.1, max_depth=10.,
                                 hfov=79., position=[0, .88, 0])
        for score in [.6, .59]:
            with self.subTest(score=score):
                view = FramingTargetReviewer(cfg, sensor, 'potted plant', self.client,
                                             self.tmp.name, 1, episode_token=42)
                mask = np.zeros((64, 64), bool); mask[20:36, 20:36] = True
                view.observe(self.rgb, {'gps':np.zeros(3), 'compass':[0.],
                             'depth':np.full((64,64,1),2.)}, [mask], [.8], [0], 0., 1, action=1)
                view.associate(42, 1, [1], ['assigned'], [1, 1, 3, 3])
                target = [1, .5, .4, 2, 2, 1, 1, 1, 3, 3, 3.]
                with patch('vlm.qwen_review.requests.post', return_value=Mock(
                        status_code=200, json=lambda: self.data(score))) as post:
                    result = view.begin(target)
                self.assertEqual(result, (1, view.CONFIRMED if score == .6 else view.UNCERTAIN))
                self.assertEqual(view.qwen_scores[1], score)
                self.assertEqual(view.stats['vlm_calls'], 1)
                post.assert_called_once()
                events = [json.loads(s) for s in (view.directory/'events.jsonl').read_text().splitlines()]
                event = next(e for e in events if e['event'] == 'verification')
                self.assertEqual(event['model_reply']['answer']['decision'], 'uncertain')
                self.assertEqual(event['model_reply']['plant_confirm_threshold'], .6)
                if score == .6:
                    self.assertNotIn(1, view.inspection_readiness())
                    self.assertFalse(any(e['event'] == 'additional_view_pending' for e in events))
                else:
                    self.assertIn(1, view.inspection_readiness())
                    view.begin(target[:-1] + [4.])
                    self.assertIsNotNone(view.active)
                    self.assertEqual(view.stats['inspections'], 1)


if __name__ == '__main__':
    unittest.main()
