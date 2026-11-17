"""Exercise v2 on real payload/ledger/audit paths without online inference."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
from vlm import qwen_review as Q
from vlm.test_qwen_review import response
from vlm import test_semantic_prompt as prior_tests
from docker import audit_annular as audit
from docker.run_batch import evaluation_command

POLICY = 'semantic_guard_v2'


class SemanticV2Tests(unittest.TestCase):
    def test_all_previous_prompt_hashes_remain_byte_identical(self):
        fixture = json.loads(Path(__file__).with_name('semantic_prompt_compatibility.json').read_text())
        self.assertEqual(len(fixture), 48)
        for item in fixture:
            actual = Q.build_prompt(item['target'], policy=item['policy'], **item['kwargs'])
            self.assertEqual(hashlib.sha256(actual.encode()).hexdigest(), item['sha256'], item)

    def test_only_three_target_categories_change_from_v1(self):
        for target in ['bed', 'chair', 'couch', 'potted plant', 'toilet', 'tv']:
            for kwargs in [dict(target_confidence=True), dict(target_confidence=True, multiview=True, count=5),
                           dict(candidate_ids=[2, 8], choice_id_policy='image_order_v1')]:
                v1 = Q.build_prompt(target, policy='semantic_guard_v1', **kwargs)
                v2 = Q.build_prompt(target, policy=POLICY, **kwargs)
                self.assertEqual(v1 == v2, target in ('chair', 'potted plant', 'toilet'))

    def test_single_joint_and_choice_payloads_preserve_answers_and_strict_mapping(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / 'ledger.json'
            ledger.write_text(json.dumps(dict(limit_cny=20, initial_spent_cny=0, calls={})))
            with patch.object(Q, 'read_credentials', return_value='fake-key'):
                client = Q.QwenReviewer(ledger, target_confidence=True, prompt_policy=POLICY,
                                        choice_id_policy='image_order_v1')
            images = [np.full((8, 8, 3), value, np.uint8) for value in (0, 64, 128, 192, 255)]
            crops = []
            for n, img in enumerate(images):
                path = Path(tmp) / f'{n}.png'; Image.fromarray(img).save(path); crops.append(str(path))
            for target in ('bed', 'couch', 'tv'):
                for mode in ('single', 'joint', 'choice'):
                    data = response('uncertain')
                    body = json.loads(data['choices'][0]['message']['content'])
                    body['target_confidence'] = .41
                    if mode == 'choice':
                        body = dict(outcome='selected', candidate_id=5, reason='best visible evidence')
                    data['choices'][0]['message']['content'] = json.dumps(body)
                    with patch.object(Q.requests, 'post', return_value=Mock(status_code=200, json=lambda: data)) as post:
                        if mode == 'single': result = client(images[0], target)
                        elif mode == 'joint': result = client.review_images(images, target)
                        else: result = client.compare_candidates(images, target, [1, 2, 3, 6, 7])
                    self.assertEqual(post.call_count, 1)
                    payload = post.call_args.kwargs['json']
                    content = payload['messages'][0]['content']
                    self.assertEqual(len(content)-1, 1 if mode == 'single' else 5)
                    self.assertEqual(result['prompt_policy'], POLICY)
                    self.assertEqual(result['prompt_sha256'], hashlib.sha256(content[-1]['text'].encode()).hexdigest())
                    self.assertEqual(payload['max_tokens'], 192)
                    if mode != 'choice':
                        self.assertEqual(result['answer'], body)  # No forced decisions or score rewrite.
                    else:
                        self.assertEqual(result['raw_answer'], body)
                        self.assertEqual(result['answer'], {**body, 'candidate_id': 7})
                        event = dict(model_reply=result, candidate_ids=[1, 2, 3, 6, 7], outcome='selected',
                                     selected_candidate=7, crops=crops)
                        bill = json.loads(ledger.read_text())
                        audit.validate_comparison_http(event, bill, target, POLICY, 'image_order_v1')
                        with self.assertRaises(AssertionError):
                            audit.validate_comparison_http(event, bill, target, 'semantic_guard_v1', 'image_order_v1')
                        bad = copy.deepcopy(bill)
                        bad['calls'][result['call_id']]['prompt_sha256'] = 'changed'
                        with self.assertRaises(AssertionError):
                            audit.validate_comparison_http(event, bad, target, POLICY, 'image_order_v1')
            calls = json.loads(ledger.read_text())['calls']
            self.assertEqual(len(calls), 9)
            self.assertTrue(all(c['prompt_policy'] == POLICY and c['transport_attempt'] == 1 for c in calls.values()))

    def test_real_cli_adds_only_new_prompt_policy(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd() / 'outputs/runtime') as tmp:
            ledger = Path(tmp) / 'budget.json'
            ledger.write_text(json.dumps(dict(limit_cny=20, initial_spent_cny=0, calls={})))
            base = [sys.executable, 'docker/run_batch.py', 'hm3dv2', '--dry-run', '--episodes', '1',
                    '--episode-index', '0', '--output', 'outputs/runs/semantic-v2-cli-dry-run',
                    '--uncertain-after-steps', '400', '--qwen-budget-ledger', str(ledger),
                    '--method', 'selected_review', '--review-backend', 'qwen',
                    '--review-profile', 'linked_choice', '--step-sync', '--qwen-choice-id-policy', 'image_order_v1']
            outputs = [json.loads(subprocess.check_output(base + ['--qwen-prompt-policy', p], text=True))
                       for p in ['semantic_guard_v1', POLICY]]
            self.assertEqual(outputs[0]['command'],
                             [s.replace('qwen_prompt_policy=semantic_guard_v2', 'qwen_prompt_policy=semantic_guard_v1')
                              for s in outputs[1]['command']])
        with self.assertRaises(ValueError):
            evaluation_command('hm3dv2', 1, '/tmp/run', False, qwen_prompt_policy=POLICY)


class SemanticV2AuditTests(prior_tests.SemanticAuditTests):
    def setUp(self):
        p = patch.object(prior_tests, 'POLICY', POLICY)
        p.start(); self.addCleanup(p.stop)


if __name__ == '__main__':
    unittest.main()
