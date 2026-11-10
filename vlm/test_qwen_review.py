"""No-network regressions for real API failure, cost and review-state boundaries."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock

import numpy as np
import requests

from vlm.qwen_review import QwenReviewer, parse_answer, RESERVE_CNY, MODEL
from vlm import test_active_view


def response(decision='confirmed', quality='adequate'):
    return {'model': MODEL, 'id': 'test', 'usage': {'prompt_tokens': 154, 'completion_tokens': 73},
            'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(
                {'decision': decision, 'quality': quality, 'visible_category': 'chair',
                 'reason': 'visible seat and back'})}}]}


class QwenTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.ledger = Path(self.temp.name) / 'budget.json'
        self.ledger.write_text(json.dumps({'limit_cny': 20, 'initial_spent_cny': 0.000884,
                                          'calls': {}}))
        with patch('vlm.qwen_review.read_credentials', return_value='fake-key'):
            self.client = QwenReviewer(self.ledger)
        self.rgb = np.zeros((64, 64, 3), dtype=np.uint8)

    def calls(self):
        return json.loads(self.ledger.read_text())['calls']

    def test_actual_usage_and_insufficient_does_not_confirm(self):
        with patch('vlm.qwen_review.requests.post', return_value=Mock(
                status_code=200, json=lambda: response(quality='insufficient'))) as post:
            result = self.client(self.rgb, 'chair')
        self.assertEqual(result['decision'], 'uncertain')
        self.assertAlmostEqual(result['charged_cny'], .000884)
        self.assertFalse(post.call_args.kwargs['allow_redirects'])
        self.assertFalse(post.call_args.kwargs['json']['enable_thinking'])
        self.assertNotIn('fake-key', self.ledger.read_text())

    def test_timeout_retains_reservation_and_never_retries(self):
        with patch('vlm.qwen_review.requests.post', side_effect=requests.Timeout('fake-key')) as post:
            with self.assertRaises(RuntimeError) as error:
                self.client(self.rgb, 'chair')
        self.assertNotIn('fake-key', str(error.exception))
        self.assertEqual(post.call_count, 1)
        self.assertEqual(next(iter(self.calls().values()))['charged_cny'], RESERVE_CNY)

    def test_restart_cannot_reset_budget_or_reuse_lost_reserve(self):
        self.ledger.write_text(json.dumps({'limit_cny': .03, 'initial_spent_cny': 0,
                                          'calls': {'lost': {'charged_cny': RESERVE_CNY}}}))
        with patch('vlm.qwen_review.requests.post') as post:
            with self.assertRaisesRegex(RuntimeError, 'budget exhausted'):
                self.client(self.rgb, 'chair')
            post.assert_not_called()

    def test_bad_http_json_finish_and_usage_stop_without_navigation_fallback(self):
        bad_finish = response(); bad_finish['choices'][0]['finish_reason'] = 'length'
        bad_usage = response(); bad_usage['usage']['completion_tokens'] = 500
        bad_answer = response(); bad_answer['choices'][0]['message']['content'] = '{bad'
        for status, data in [(401, {}), (200, bad_finish), (200, bad_usage), (200, bad_answer)]:
            with self.subTest(status=status, data=data):
                with patch('vlm.qwen_review.requests.post', return_value=Mock(
                        status_code=status, json=lambda: data)):
                    with self.assertRaises(RuntimeError):
                        self.client(self.rgb, 'chair')
        self.assertTrue(all(c['state'] == 'error' for c in self.calls().values()))

    def test_original_review_state_machine_uses_decisions_not_thresholds(self):
        # Exercise the actual reviewer and existing source-association helper.
        helper = test_active_view.ReviewerTest(); helper.setUp()
        try:
            for partial, answer in [(False, 'confirmed'), (False, 'rejected'),
                                    (True, 'rejected'), (False, 'uncertain')]:
                view = helper.reviewer(linked=True, backend='qwen')
                view.verifier = lambda image, label: {'decision': answer, 'call_id': 'test'}
                helper.observe(view, 1, partial=partial); helper.assign(view)
                result = view.begin(helper.target(view))
                expected = {'confirmed': view.CONFIRMED, 'rejected': view.REJECTED}.get(answer)
                if partial or answer == 'uncertain': expected = None
                self.assertEqual(result, (1, expected) if expected else None)
                event = next(e for e in helper.events(view) if e['event'] == 'verification')
                self.assertIsNone(event['score'])
                self.assertIsNone(event['confirm_threshold'])
                self.assertEqual(view.stats['vlm_calls'], 1)
        finally:
            helper.doCleanups()


if __name__ == '__main__':
    unittest.main()
