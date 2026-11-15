"""Exercise the offline audit against real recorder output and tampered evidence."""
import copy
import json
import math
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import numpy as np
import cv2
from PIL import Image
from docker import audit_annular as A
from vlm import test_annular_review as fixture
from vlm.qwen_review import QwenReviewer
from vlm.test_qwen_review import response


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.f = fixture.AnnularTests(); self.f.setUp(); self.addCleanup(self.f.doCleanups)
        self.ledger = Path(self.f.tmp.name)/'budget.json'
        self.ledger.write_text(json.dumps(dict(limit_cny=20, initial_spent_cny=0, calls={})))
        with patch('vlm.qwen_review.read_credentials', return_value='fake-key'):
            self.f.v.multiview_backend = QwenReviewer(self.ledger, target_confidence=True)
        answer = response('uncertain'); body = json.loads(answer['choices'][0]['message']['content'])
        body['target_confidence'] = .4; answer['choices'][0]['message']['content'] = json.dumps(body)
        with patch('vlm.qwen_review.requests.post', return_value=Mock(status_code=200, json=lambda: answer)):
            for i in range(1, 6): self.f.frame(i, 1.-i*.03, yaw=math.pi/4)
            self.f.begin(); self.f.begin(4.); self.f.frame(6, .1, yaw=math.pi/4)
            self.f.v.finish_inspection(True); self.f.v.finish_episode()
        self.events = [json.loads(x) for x in (self.f.v.directory/'events.jsonl').read_text().splitlines()]
        self.bill = json.loads(self.ledger.read_text())
        self.calls = [e for e in self.events if e['event'] == 'verification']
        self.sources = [[e for e in self.events if e['event'] == 'verification_source' and e['phase'] == call['phase']] for call in self.calls]
        self.history = [e for e in self.events if e['event'] == 'candidate_image']
        self.trace = [dict(step=i, action=2 if i else None, pose=[0., 0., 0., math.pi/4, 0.]) for i in range(7)]

    def test_real_first_and_joint_inputs_and_bill(self):
        for i, event in enumerate(self.calls, 1):
            A.validate_inputs(event, self.sources[i-1], [e for e in self.history if e['step'] <= event['step']], i)
            A.validate_call(event, self.bill, 'chair', i)

    def test_real_crop_and_map_receipt(self):
        for group in self.sources:
            for source in group:
                receipt = next(e for e in self.events if e['event'] == 'source_association' and e['observation'] == source['observation'])
                A.validate_source(self.f.v.directory, source, receipt, self.trace, self.f.v.sensor)

    def test_full_recorded_episode(self):
        run = Path(self.f.tmp.name)
        lines = ['[Conditional inspection trigger] id=1 ',
                 '[Annular sample] id=1 sample=1/2 step=5 position=(0.0000,0.0000) yaw=0.7854',
                 '[Annular sample] id=1 sample=2/2 step=6 position=(0.0000,0.0000) yaw=0.7854']
        for call in self.calls:
            lines.append(f'[Candidate scores] step={call["step"]} id=1 fusion=0.7 state=2 has_qwen=1 qwen=0.4 rank=0.7 late=0')
        (run/'planner.log').write_text('\n'.join(lines))
        sensor = vars(self.f.v.sensor)
        cfg = {'habitat': {'simulator': {'agents': {'main_agent': {'sim_sensors': {'depth_sensor': sensor}}}}}}
        result = A.audit_evidence(run, self.f.v.directory, self.events, dict(success='1', steps='6', video_path='', target='chair'), cfg, self.trace, self.bill)
        self.assertEqual(len(result['calls']), 2)
        self.assertEqual(result['calls'][1]['image_count'], 5)

    def test_tampered_sources_and_order_rejected(self):
        for mutate in [lambda s: s.reverse(), lambda s: s.pop(), lambda s: s[0].update(candidate=2),
                       lambda s: s[0].update(mask_index=3), lambda s: s[0].update(detector_score=.01)]:
            sources = copy.deepcopy(self.sources[1]); mutate(sources)
            with self.subTest(mutate=mutate), self.assertRaises(AssertionError):
                A.validate_inputs(self.calls[1], sources, self.history, 2)

    def test_lower_score_new_view_cannot_replace_top5(self):
        history = copy.deepcopy(self.history); history[-1]['detector_score'] = 1.
        with self.assertRaises(AssertionError): A.validate_inputs(self.calls[1], self.sources[1], history, 2)

    def test_ledger_hash_order_target_and_cost_rejected(self):
        call = self.calls[1]; key = call['model_reply']['call_id']
        for mutate in [lambda b: b['image_sha256s'].reverse(), lambda b: b.update(image_count=4),
                       lambda b: b.update(target='tv'), lambda b: b.update(charged_cny=.02),
                       lambda b: b.update(prompt_sha256='wrong')]:
            bill = copy.deepcopy(self.bill); mutate(bill['calls'][key])
            with self.subTest(mutate=mutate), self.assertRaises(AssertionError): A.validate_call(call, bill, 'chair', 2)

    def test_third_call_and_promoted_uncertainty_rejected(self):
        with self.assertRaises(AssertionError): A.validate_inputs(self.calls[1], self.sources[1], self.history, 3)
        event = copy.deepcopy(self.calls[0]); event['decision'] = 1
        with self.assertRaises(AssertionError): A.validate_call(event, self.bill, 'chair', 1)

    def test_crop_pixels_or_source_pose_tampering_rejected(self):
        source = self.sources[0][0]
        receipt = next(e for e in self.events if e['event'] == 'source_association' and e['observation'] == source['observation'])
        path = Path(source['crop']); original = path.read_bytes()
        Image.fromarray(np.zeros((16, 16, 3), np.uint8)).save(path)
        with self.assertRaises(AssertionError): A.validate_source(self.f.v.directory, source, receipt, self.trace, self.f.v.sensor)
        path.write_bytes(original)
        trace = copy.deepcopy(self.trace); trace[source['observation']]['pose'][0] = 8
        with self.assertRaises(AssertionError): A.validate_source(self.f.v.directory, source, receipt, trace, self.f.v.sensor)

    def video(self, frames):
        run = Path(self.f.tmp.name)/'video-case'; (run/'infeasible').mkdir(parents=True)
        path = run/'infeasible/episode.mp4'
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), 10, (32, 32))
        assert writer.isOpened()
        for i in range(frames): writer.write(np.full((32, 32, 3), i, np.uint8))
        writer.release()
        return run, dict(success='0', steps=str(frames-1), video_path='infeasible/episode.mp4', target='chair', result='infeasible')

    def test_zero_step_infeasible_with_complete_initialization(self):
        run, row = self.video(1)
        (run/'planner.log').write_text('Init Mode Process -----> (0/26)\n[Observation applied] step=0 masks=0')
        event = dict(event='episode_end', step=0, **{k: 0 for k in ('selected_targets','vlm_calls','vlm_seconds','confirmed','uncertain','rejected','inspections','inspection_actions','pending_evidence')})
        cfg = {'habitat': {'simulator': {'agents': {'main_agent': {'sim_sensors': {'depth_sensor': vars(self.f.v.sensor)}}}}}}
        result = A.audit_evidence(run, self.f.v.directory, [event], row, cfg, [dict(step=0, action=None)], self.bill)
        self.assertEqual(result['video']['frames'], 1)
        # Removing initialization evidence must not become a broad bypass.
        (run/'planner.log').write_text('[Observation applied] step=0 masks=0')
        with self.assertRaises(AssertionError): A.audit_evidence(run, self.f.v.directory, [event], row, cfg, [dict(step=0, action=None)], self.bill)

    def test_failure_video_frame_mismatch(self):
        run, row = self.video(3); self.assertEqual(A.video_audit(run, row)['frames'], 3)
        row['steps'] = '3'
        with self.assertRaises(AssertionError): A.video_audit(run, row)

    def test_success_video_must_be_removed(self):
        run, row = self.video(1); row.update(success='1', video_path='')
        with self.assertRaises(AssertionError): A.video_audit(run, row)
        (run/'infeasible/episode.mp4').unlink(); self.assertIsNone(A.video_audit(run, row))

    def test_duplicate_episode_end_is_rejected(self):
        cfg = {'habitat': {'simulator': {'agents': {'main_agent': {'sim_sensors': {'depth_sensor': vars(self.f.v.sensor)}}}}}}
        with self.assertRaises(AssertionError):
            A.audit_evidence(Path(self.f.tmp.name), self.f.v.directory, self.events+[self.events[-1]], {}, cfg, self.trace, self.bill)


if __name__ == '__main__': unittest.main()
