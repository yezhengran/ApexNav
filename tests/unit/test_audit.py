"""Offline regression tests; external inference is mocked."""
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
from tests.unit import test_annular_review as annular_fixture
from vlm.qwen_review import QwenReviewer
from tests.unit.test_qwen_review import response
from tests.unit.test_choice_review import ChoiceTests
from docker import audit_annular as audit
from tests.unit import test_crop_pose_review as crop_pose_fixture

class AuditTests(unittest.TestCase):
    def setUp(self):
        self.f = annular_fixture.AnnularTests(); self.f.setUp(); self.addCleanup(self.f.doCleanups)
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

class ChoiceAuditTest(unittest.TestCase):
    def setUp(self):
        f=ChoiceTests();f.setUp();self.addCleanup(f.doCleanups);self.f=f
        self.run=Path(f.tmp.name);ledger=self.run/'budget.json'
        ledger.write_text(json.dumps(dict(limit_cny=20,initial_spent_cny=0,calls={})))
        with patch('vlm.qwen_review.read_credentials',return_value='fake-key'):
            f.v.multiview_backend=QwenReviewer(ledger,target_confidence=True)
        no_frontier=getattr(self,'no_frontier',False)
        f.v.cfg.choice_no_frontier=no_frontier
        single=response('uncertain' if no_frontier else 'confirmed');answer=json.loads(single['choices'][0]['message']['content']);answer['target_confidence']=.9
        single['choices'][0]['message']['content']=json.dumps(answer)
        comparative=response();comparative['choices'][0]['message']['content']=json.dumps(dict(outcome='selected',candidate_id=2,reason='best'))
        with patch('vlm.qwen_review.requests.post',side_effect=[Mock(status_code=200,json=lambda:single),Mock(status_code=200,json=lambda:single),Mock(status_code=200,json=lambda:comparative)]):
            f.frame(1,.9,ident=1);f.v.begin([1,.1,.65,2,2,1,1,1,3,3,3.])
            f.frame(2,.8,ident=2);f.v.begin([2,.1,.65,2,2,1,1,1,3,3,3.])
            f.v.compare([1,2],1,42,2,reason='no_frontier' if no_frontier else 'multiple_confirmed');f.v.finish_episode()
        self.events=[json.loads(x) for x in (f.v.directory/'events.jsonl').read_text().splitlines()]
        self.ledger=json.loads(ledger.read_text())
        self.cfg={'active_view':{'choice_review':True,'choice_no_frontier':no_frontier},'habitat':{'simulator':{'agents':{'main_agent':{'sim_sensors':{'depth_sensor':vars(f.v.sensor)}}}}}}
        self.trace=[dict(step=i,action=2 if i else None,pose=[0.,0.,0.,0.,0.]) for i in range(3)]
        self.row=dict(target='chair',success='1',steps='2',video_path='')
        (self.run/'planner.log').write_text('\n'.join(f'[Candidate scores] step={i} id={i} fusion=0.1 state=1 has_qwen=1 qwen=0.9 rank=0.1 late=0' for i in (1,2)))
        if no_frontier:
            with (self.run/'planner.log').open('a') as log:
                log.write('\n[No-frontier Qwen pool] count=2 provisional=1 extreme=0\n[Comparison request] request=1 candidates=2 phase=no_frontier step=2\n[Comparison pool] request=1 ids=1,2 phase=no_frontier step=2')
    def audit(self,events=None,bill=None):
        return A.audit_evidence(self.run,self.f.v.directory,events or self.events,self.row,self.cfg,self.trace,bill or self.ledger)
    def test_full_choice_record_rebuilds_all_sources_and_http_chain(self):
        result=self.audit();self.assertEqual(len(result['comparisons']),1);self.assertEqual(len(result['calls']),2)
    def test_invalid_trigger_selection_and_early_pool_rejected(self):
        for event,key,value in [('choice_trigger_check','crop_passed',False),('comparison_finished','selected_candidate',3),('comparison_requested','candidate_ids',[1,3])]:
            events=copy.deepcopy(self.events);next(e for e in events if e['event']==event)[key]=value
            with self.assertRaises((AssertionError,KeyError)):self.audit(events)
    def test_reordered_images_or_ledger_identity_rejected(self):
        events=copy.deepcopy(self.events);next(e for e in events if e['event']=='candidate_comparison')['crops'].reverse()
        with self.assertRaises(AssertionError):self.audit(events)
        bill=copy.deepcopy(self.ledger)
        call=next(e for e in self.events if e['event']=='candidate_comparison')['model_reply']['call_id']
        bill['calls'][call]['candidate_ids']=[2,1]
        with self.assertRaises(AssertionError):self.audit(bill=bill)

class NoFrontierChoiceAuditTest(ChoiceAuditTest):
    no_frontier=True

    def test_explicit_gate_and_exact_planner_pool_required(self):
        self.audit()
        self.cfg['active_view']['choice_no_frontier']=False
        with self.assertRaises(AssertionError):self.audit()
        self.cfg['active_view']['choice_no_frontier']=True
        path=self.run/'planner.log';path.write_text(path.read_text().replace('ids=1,2','ids=1,3'))
        with self.assertRaises(AssertionError):self.audit()

    def test_uncertain_pool_cannot_claim_multiple_confirmed(self):
        events=copy.deepcopy(self.events)
        next(e for e in events if e['event']=='comparison_requested')['reason']='multiple_confirmed'
        with self.assertRaises(AssertionError):self.audit(events)

class CropPoseAuditTests(unittest.TestCase):
    def setUp(self):
        self.f = crop_pose_fixture.CropPoseTests(); self.f.setUp(); self.addCleanup(self.f.doCleanups)
        self.run = Path(self.f.fixture.tmp.name)
        self.single_view=getattr(self,'single_view',False)
        self.f.v.cfg.crop_pose_single_view=self.single_view
        ledger = self.run/'budget.json'
        ledger.write_text(json.dumps(dict(limit_cny=20, initial_spent_cny=0, calls={})))
        with patch('vlm.qwen_review.read_credentials', return_value='fake-key'):
            self.f.v.multiview_backend = QwenReviewer(ledger, target_confidence=True)
        answer = response('uncertain')
        body = json.loads(answer['choices'][0]['message']['content']); body['target_confidence'] = .4
        answer['choices'][0]['message']['content'] = json.dumps(body)
        with patch('vlm.qwen_review.requests.post', return_value=Mock(status_code=200,json=lambda:answer)):
            self.f.frame(1,.9,yaw=math.pi/4); self.f.frame(2,.8,yaw=math.pi/4)
            self.f.begin(); self.f.begin(4.)
            self.f.v.finish_inspection(False,initial_unavailable=True)
            if not self.single_view:self.f.frame(3,.1,yaw=math.pi/4)
            self.f.v.finish_inspection(True); self.f.v.finish_episode()
        self.events = [json.loads(s) for s in (self.f.v.directory/'events.jsonl').read_text().splitlines()]
        self.bill = json.loads(ledger.read_text())
        self.cfg = {'active_view':{'choice_review':True,'crop_pose_fallback':True,'crop_pose_single_view':self.single_view},'habitat':{'simulator':{
            'agents':{'main_agent':{'sim_sensors':{'depth_sensor':vars(self.f.v.sensor)}}}}}}
        self.trace = [dict(step=i,action=2 if i else None,pose=[0.,0.,0.,math.pi/4,0.]) for i in range(3 if self.single_view else 4)]
        self.row = dict(success='1',steps='2' if self.single_view else '3',video_path='',target='chair')
        self.log = '\n'.join([
            '[Conditional inspection trigger] id=1 ',
            '[Annular search] id=1 mode=framing step=2 radius=3.0 generated=72 safe=0 visible=0 access=0 arcs=0 accepted=0',
            '[Annular sample] id=1 sample=1/2 step=2 position=(0.0000,0.0000) yaw=0.7854',
            '[Annular sample] id=1 sample=2/2 step=3 position=(0.0000,0.0000) yaw=0.7854',
            *[f'[Candidate scores] step={i} id=1 fusion=0.7 state=2 has_qwen=1 qwen=0.4 rank=0.7 late=0' for i in (2,3)]])
        if self.single_view:
            self.log=self.log.replace('sample=1/2','sample=1/1')
            self.log='\n'.join(line for line in self.log.splitlines() if 'sample=2/2' not in line)
        (self.run/'planner.log').write_text(self.log)

    def check(self, events=None):
        return audit.audit_evidence(self.run,self.f.v.directory,self.events if events is None else events,
                                    self.row,self.cfg,self.trace,self.bill)

    def test_recorded_fallback_source_joint_images_and_ledger(self):
        result = self.check()
        self.assertEqual(result['crop_pose_fallbacks'],[1])
        self.assertEqual([c['image_count'] for c in result['calls']],[1,2 if self.single_view else 3])

    def test_fallback_geometry_source_or_boundary_tamper(self):
        for key,value in [('radius',1.),('pose',[1.,0.,math.pi/4]),('observation',2),
                          ('detector_score',.8),('half_arc_degrees',60),('half_width',.5),
                          ('reason','primary_geometry_unavailable'),('step',400)]:
            events = copy.deepcopy(self.events)
            next(e for e in events if e['event']=='inspection_crop_pose_fallback')[key] = value
            with self.subTest(key=key),self.assertRaises(AssertionError): self.check(events)

    def test_second_fallback_or_fallback_after_action_is_rejected(self):
        events = copy.deepcopy(self.events)
        index = next(i for i,e in enumerate(events) if e['event']=='inspection_crop_pose_fallback')
        events.insert(index,copy.deepcopy(events[index]))
        with self.assertRaises(AssertionError): self.check(events)
        events = copy.deepcopy(self.events)
        events.insert(index,dict(event='inspection_evidence',candidate=1,step=2,action=2))
        with self.assertRaises(AssertionError): self.check(events)

    def test_primary_failure_receipt_and_explicit_switch_required(self):
        (self.run/'planner.log').write_text(self.log.replace('accepted=0','accepted=1'))
        with self.assertRaisesRegex(AssertionError,'primary'): self.check()
        (self.run/'planner.log').write_text(self.log)
        self.cfg['active_view']['crop_pose_fallback'] = False
        with self.assertRaises(AssertionError): self.check()

    def test_arrival_cannot_relax_recorded_band(self):
        events = copy.deepcopy(self.events)
        next(e for e in events if e['event']=='inspection_crop_pose_arrival')['minimum_radius'] = .6
        with self.assertRaises(AssertionError): self.check(events)

class SingleCropPoseAuditTests(CropPoseAuditTests):
    single_view=True

    def test_single_sample_and_same_step_require_opt_in(self):
        result=self.check()
        self.assertEqual(result['inspection_actions'],0)
        self.cfg['active_view']['crop_pose_single_view']=False
        with self.assertRaises(AssertionError):self.check()
