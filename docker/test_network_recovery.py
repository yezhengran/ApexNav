"""Fault-injection accounting tests; no provider requests or navigation results."""
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from basic_utils.record_episode.experiment import save_result, read_rows, record_episode_identity
from network_recovery import eligible_failure, recover_episode


class RecoveryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name)
        self.ledger = self.run / 'shared-budget.json'
        # A completed navigation failure must never be replaced by this recovery.
        self.row = dict(episode_number=1, scene_id='a', episode_id='0', target='chair',
                        steps=500, success=0, spl=0, soft_spl=.1, distance_to_goal=2,
                        distance_to_goal_reward=1, result='stepout', elapsed_seconds=5,
                        cumulative_seconds=5, video_path='failure.mp4')
        save_result(self.run, self.row, 3)
        record_episode_identity(self.run, 1, 'a', '0', 'chair')
        record_episode_identity(self.run, 2, 'a', '1', 'chair')
        self.prefix = read_rows(self.run)
        self.attempt = self.run / 'verification/episode-0002/attempt-1'
        self.attempt.mkdir(parents=True)
        crop = self.attempt / 'crop.png'; crop.write_bytes(b'fixture-image')
        self.event = dict(event='verification_transport_error', candidate=1,
                          call_id='failed', retryable=True, crop=str(crop))
        self.bill = dict(state='error', error_stage='transport', started_unix=12,
                         charged_cny=.02, image_sha256=hashlib.sha256(crop.read_bytes()).hexdigest(),
                         transport=dict(retryable=True, category='response_timeout'))
        self.write()
        (self.run / 'episode-0002.partial.mp4').write_bytes(b'incomplete-video')

    def write(self, end=False):
        events = [self.event] + ([{'event':'episode_end'}] if end else [])
        (self.attempt / 'events.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in events))
        self.ledger.write_text(json.dumps(dict(limit_cny=20, initial_spent_cny=0,
                                             spent_or_reserved_cny=.02, calls={'failed':self.bill})))

    def recover(self, **kwargs):
        return recover_episode(self.run, self.ledger, 10, 2, 'frozen', wait=lambda _: None,
                               probe=kwargs.get('probe', lambda: 401))

    def test_preserve_then_resume_same_episode_and_merge_once(self):
        original_ledger = self.ledger.read_bytes()
        first = self.recover()
        self.assertEqual(first['episode'], 2)
        self.assertEqual(read_rows(self.run), self.prefix)
        backup = self.run / 'network-recovery/episode-0002/retry-01'
        self.assertEqual((backup/'episode-0002.partial.mp4').read_bytes(), b'incomplete-video')
        self.assertEqual(self.ledger.read_bytes(), original_ledger)
        # Successful retry appends one row; it cannot replace the earlier failure.
        row = {**self.row, 'episode_number':2, 'episode_id':'1', 'success':1, 'result':'success'}
        record_episode_identity(self.run, 2, 'a', '1', 'chair')
        save_result(self.run, row, 3)
        self.assertEqual(read_rows(self.run)[:1], self.prefix)
        self.assertEqual(len(read_rows(self.run)), 2)
        self.assertIsNone(eligible_failure(self.run, self.ledger, 10))
        with self.assertRaises(ValueError):
            save_result(self.run, row, 3)

    def test_limit_persists_and_no_completed_episode_restart(self):
        self.assertEqual(self.recover()['retry'], 1)
        self.assertEqual(self.recover()['retry'], 2)
        self.assertIsNone(self.recover())
        self.write(end=True)
        self.assertIsNone(eligible_failure(self.run, self.ledger, 10))

    def test_stale_certificate_answer_and_http_error_never_retry(self):
        for update in ({'started_unix':1}, {'http_status':503}, {'state':'completed'},
                       {'error_stage':'response'}, {'transport':{'retryable':False, 'category':'certificate_verification'}}):
            original = self.bill.copy()
            self.bill.update(update); self.write()
            self.assertIsNone(eligible_failure(self.run, self.ledger, 10))
            self.bill = original

    def test_bad_network_probe_stops_after_archival_without_inference(self):
        with self.assertRaisesRegex(RuntimeError, 'reconnection failed'):
            self.recover(probe=lambda: 503)
        record = json.loads((self.run/'network-recovery/episode-0002/retry-01/recovery.json').read_text())
        self.assertEqual(record['state'], 'reconnect_failed')
        self.assertEqual(read_rows(self.run), self.prefix)

    def test_budget_exhaustion_and_disabled_policy_do_not_restart(self):
        state = json.loads(self.ledger.read_text()); state['spent_or_reserved_cny'] = 19.99
        self.ledger.write_text(json.dumps(state))
        self.assertIsNone(self.recover())
        self.assertIsNone(recover_episode(self.run, self.ledger, 10, 0, 'frozen'))
        self.assertFalse((self.run/'network-recovery').exists())

    def test_image_mismatch_is_not_silently_recovered(self):
        self.bill['image_sha256'] = 'wrong'; self.write()
        with self.assertRaisesRegex(RuntimeError, 'image does not match'):
            self.recover()

    def test_actual_batch_loop_restarts_worker_and_appends_once(self):
        self.run_batch_loop_case()

    def test_fixed_index_network_restart_preserves_identity_and_appends_once(self):
        self.run_batch_loop_case(single_index=196)

    def run_batch_loop_case(self, single_index=None):
        import run_batch
        import network_recovery
        root = self.run
        output = root / 'outputs/runs/recovery'
        launches = []
        number = 2 if single_index is None else 1
        def worker(command, **kwargs):
            process = Mock(pid=100 + len(launches))
            process.poll.return_value = None
            if command[0] == 'ros2':
                return process
            launches.append(command)
            process.stdout = io.StringIO('fixture worker\n')
            def finish():
                if len(launches) == 1:
                    if single_index is None:
                        save_result(output, self.row, 2)
                    attempt = output/f'verification/episode-{number:04d}/attempt-1'
                    attempt.mkdir(parents=True)
                    crop = attempt/'crop.png'; crop.write_bytes(b'fixture-image')
                    event = {**self.event, 'crop':str(crop)}
                    (attempt/'events.jsonl').write_text(json.dumps(event)+'\n')
                    state = json.loads(self.ledger.read_text())
                    state['calls']['failed']['started_unix'] = run_batch.time.time()
                    self.ledger.write_text(json.dumps(state))
                    (output/f'episode-{number:04d}.partial.mp4').write_bytes(b'partial')
                    return 1
                self.assertEqual(len(read_rows(output)), number-1)
                save_result(output, {**self.row, 'episode_number':number, 'episode_id':'1',
                                     'success':1, 'result':'success'}, number)
                return 0
            process.wait.side_effect = finish
            return process
        modules = {
            'habitat':types.ModuleType('habitat'),
            'habitat.config':types.ModuleType('habitat.config'),
            'habitat.config.default_structured_configs':types.ModuleType('habitat.config.default_structured_configs'),
            'hydra':types.ModuleType('hydra')}
        modules['habitat.config.default_structured_configs'].HabitatConfigPlugin = object
        modules['habitat.config.default_structured_configs'].register_hydra_plugin = lambda _: None
        from contextlib import nullcontext
        modules['hydra'].initialize_config_dir = lambda **_: nullcontext()
        from omegaconf import OmegaConf
        modules['hydra'].compose = lambda **_: OmegaConf.create({})
        args = ['batch', 'hm3dv2', '--episodes',str(number), '--output',str(output),
                '--method','selected_review','--review-profile','linked_stop_preserve',
                '--review-backend','qwen','--qwen-budget-ledger',str(self.ledger),
                '--network-episode-retries','2']
        if single_index is not None:
            args += ['--episode-index', str(single_index)]
        original_recover = network_recovery.recover_episode
        def recovery(*a):
            return original_recover(*a, wait=lambda _:None, probe=lambda:401)
        import os
        cwd = Path.cwd()
        try:
            with patch.object(run_batch,'ROOT',root), patch.object(sys,'argv',args), \
                    patch.dict(sys.modules,modules), patch.object(run_batch,'source_provenance',return_value={'source_sha256':'frozen'}), \
                    patch.object(run_batch,'runtime_sources',return_value=('sha',{},False)), \
                    patch.object(run_batch,'source_digest',return_value='frozen'), \
                    patch.object(run_batch.subprocess,'run'), patch.object(run_batch.subprocess,'Popen',side_effect=worker), \
                    patch.object(run_batch,'stop_owned'), patch.object(run_batch.time,'sleep'), \
                    patch.object(network_recovery,'recover_episode',side_effect=recovery):
                self.assertEqual(run_batch.main(), 0)
        finally:
            os.chdir(cwd)
        self.assertEqual(len(launches), 2)
        self.assertEqual(launches[0], launches[1])
        self.assertEqual([r['success'] for r in read_rows(output)], ['0','1'] if single_index is None else ['1'])
        self.assertIn(f'test_epi_num={single_index if single_index is not None else -1}', launches[1])
        self.assertTrue((output/f'network-recovery/episode-{number:04d}/retry-01/episode-{number:04d}.partial.mp4').exists())
        self.assertEqual(json.loads((output/'status.json').read_text()), {'state':'completed','exit_code':0})


if __name__ == '__main__':
    unittest.main()
