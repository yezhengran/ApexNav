"""No-network regressions for real API failure, cost and review-state boundaries."""

import json
import errno
from pathlib import Path
import socket
import ssl
import tempfile
import threading
import unittest
from unittest.mock import patch, Mock

import numpy as np
import requests

from vlm.qwen_review import QwenReviewer, parse_answer, RESERVE_CNY, MODEL, transport_diagnostic
from vlm import test_active_view


def response(decision='confirmed', quality='adequate'):
    return {'model': MODEL, 'id': 'test', 'usage': {'prompt_tokens': 154, 'completion_tokens': 73},
            'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(
                {'decision': decision, 'quality': quality, 'visible_category': 'chair',
                 'reason': 'visible seat and back'})}}]}


def ssl_eof():
    return requests.exceptions.SSLError(
        ssl.SSLEOFError(8, '[SSL: UNEXPECTED_EOF_WHILE_READING] fake-key secret-context'))


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
        for status, data in [(401, {}), (429, {}), (503, {}), (200, bad_finish),
                             (200, bad_usage), (200, bad_answer)]:
            with self.subTest(status=status, data=data):
                with patch('vlm.qwen_review.requests.post', return_value=Mock(
                        status_code=status, json=lambda: data)) as post:
                    with self.assertRaises(RuntimeError):
                        self.client(self.rgb, 'chair')
                    self.assertEqual(post.call_count, 1)
        self.assertTrue(all(c['state'] == 'error' for c in self.calls().values()))

    def test_two_connection_failures_then_success_keep_payload_and_all_reserves(self):
        reply = Mock(status_code=200, json=lambda: response())
        with patch('vlm.qwen_review.requests.post', side_effect=[ssl_eof(), ssl_eof(), reply]) as post, \
                patch('vlm.qwen_review.time.sleep') as sleep:
            result = self.client(self.rgb, 'chair')
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [2, 5])
        self.assertEqual(post.call_count, 3)
        self.assertTrue(all(c == post.call_args_list[0] for c in post.call_args_list))
        self.assertTrue(post.call_args.kwargs['verify'])
        bills = self.calls()
        self.assertEqual(result['transport_attempt_ids'], list(bills))
        self.assertEqual(len({c['request_id'] for c in bills.values()}), 1)
        self.assertEqual([c['state'] for c in bills.values()], ['error', 'error', 'completed'])
        self.assertAlmostEqual(sum(c['charged_cny'] for c in bills.values()), .04 + .000884)
        self.assertEqual(result['decision'], 'confirmed')
        self.assertEqual(result['call_id'], list(bills)[-1])
        for secret in ('fake-key', 'secret-context'):
            self.assertNotIn(secret, self.ledger.read_text())
        reply.close.assert_called_once()

    def test_third_connection_failure_stops_and_retains_three_reservations(self):
        with patch('vlm.qwen_review.requests.post', side_effect=ssl_eof()) as post, \
                patch('vlm.qwen_review.time.sleep') as sleep:
            with self.assertRaisesRegex(RuntimeError, 'connection_interrupted'):
                self.client(self.rgb, 'chair')
        self.assertEqual(post.call_count, 3)
        self.assertEqual(sleep.call_count, 2)
        self.assertAlmostEqual(sum(c['charged_cny'] for c in self.calls().values()), .06)
        self.assertFalse(list(self.calls().values())[-1]['retry_scheduled'])

    def test_certificate_and_unknown_ssl_never_retry(self):
        errors = [requests.exceptions.SSLError(ssl.SSLCertVerificationError(
                      1, 'CERTIFICATE_VERIFY_FAILED UNEXPECTED_EOF_WHILE_READING fake-key')),
                  requests.exceptions.SSLError('unknown SSL failure fake-key'),
                  requests.exceptions.SSLError('WRONG_VERSION_NUMBER')]
        for error in errors:
            with self.subTest(error_type=type(error).__name__), \
                    patch('vlm.qwen_review.requests.post', side_effect=error) as post, \
                    patch('vlm.qwen_review.time.sleep') as sleep:
                with self.assertRaises(RuntimeError):
                    self.client(self.rgb, 'chair')
                self.assertEqual(post.call_count, 1)
                sleep.assert_not_called()
        self.assertNotIn('fake-key', self.ledger.read_text())

    def test_response_timeout_records_unfinished_review_for_episode_recovery(self):
        helper = test_active_view.ReviewerTest(); helper.setUp()
        try:
            view = helper.reviewer(linked=True, backend='qwen')
            view.verifier = self.client
            helper.observe(view, 1); helper.assign(view)
            with patch('vlm.qwen_review.requests.post', side_effect=requests.ReadTimeout('fake-key')), \
                    patch('vlm.qwen_review.time.sleep'):
                with self.assertRaisesRegex(RuntimeError, 'response_timeout'):
                    view.begin(helper.target(view))
            events = helper.events(view)
            failure = next(e for e in events if e['event'] == 'verification_transport_error')
            self.assertTrue(failure['retryable'])
            self.assertEqual(self.calls()[failure['call_id']]['transport_attempt'], 3)
            self.assertEqual(view.stats['vlm_calls'], 0)
            self.assertFalse(any(e['event'] == 'verification' for e in events))
        finally:
            helper.doCleanups()

    def test_retry_budget_exhaustion_stops_before_second_http_attempt(self):
        self.ledger.write_text(json.dumps({'limit_cny': .03, 'initial_spent_cny': 0, 'calls': {}}))
        with patch('vlm.qwen_review.requests.post', side_effect=ssl_eof()) as post, \
                patch('vlm.qwen_review.time.sleep'):
            with self.assertRaisesRegex(RuntimeError, 'budget exhausted'):
                self.client(self.rgb, 'chair')
        self.assertEqual(post.call_count, 1)
        self.assertEqual(len(self.calls()), 1)
        self.assertEqual(next(iter(self.calls().values()))['charged_cny'], RESERVE_CNY)

    def test_typed_connection_causes_without_freeform_transport_text(self):
        for error, expected in [(requests.ConnectTimeout('private URL'), True),
                                (requests.ConnectionError(ConnectionResetError(
                                    errno.ECONNRESET, 'secret header')), True),
                                (requests.ConnectionError('unclassified'), False)]:
            diagnostic = transport_diagnostic(error)
            self.assertEqual(diagnostic['retryable'], expected)
            self.assertNotIn('private URL', json.dumps(diagnostic))
            self.assertNotIn('secret header', json.dumps(diagnostic))

    def test_real_local_tls_disconnect_is_recognized_without_disabling_verification(self):
        # A real local TCP peer closes during TLS negotiation. No credentials,
        # images, provider traffic or certificate bypass is involved.
        with socket.socket() as listener:
            listener.bind(('127.0.0.1', 0)); listener.listen(1)
            listener.settimeout(5)
            def drop_tls():
                with listener.accept()[0] as conn:
                    conn.recv(4096)
            thread = threading.Thread(target=drop_tls, daemon=True)
            thread.start()
            with requests.Session() as session:
                session.trust_env = False
                with self.assertRaises(requests.exceptions.SSLError) as error:
                    session.get(f'https://127.0.0.1:{listener.getsockname()[1]}',
                                timeout=(2, 2), verify=True)
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
        diagnostic = transport_diagnostic(error.exception)
        self.assertTrue(diagnostic['retryable'])
        self.assertEqual(diagnostic['category'], 'connection_interrupted')

    def test_transport_retry_is_one_review_result_not_an_extra_candidate_review(self):
        helper = test_active_view.ReviewerTest(); helper.setUp()
        try:
            view = helper.reviewer(linked=True, backend='qwen')
            view.verifier = self.client
            helper.observe(view, 1); helper.assign(view)
            with patch('vlm.qwen_review.requests.post', side_effect=[
                    ssl_eof(), Mock(status_code=200, json=lambda: response())]), \
                    patch('vlm.qwen_review.time.sleep'):
                result = view.begin(helper.target(view))
            self.assertEqual(result, (1, view.CONFIRMED))
            self.assertEqual(view.stats['vlm_calls'], 1)
            self.assertEqual(len(self.calls()), 2)
        finally:
            helper.doCleanups()

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
