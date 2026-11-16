"""Record an actual fallback review with fake HTTP, then audit/tamper its evidence."""
import copy
import json
import math
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from docker import audit_annular as audit
from vlm import test_crop_pose_review as fixture
from vlm.qwen_review import QwenReviewer
from vlm.test_qwen_review import response


class CropPoseAuditTests(unittest.TestCase):
    def setUp(self):
        self.f = fixture.CropPoseTests(); self.f.setUp(); self.addCleanup(self.f.doCleanups)
        self.run = Path(self.f.fixture.tmp.name)
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
            self.f.frame(3,.1,yaw=math.pi/4)
            self.f.v.finish_inspection(True); self.f.v.finish_episode()
        self.events = [json.loads(s) for s in (self.f.v.directory/'events.jsonl').read_text().splitlines()]
        self.bill = json.loads(ledger.read_text())
        self.cfg = {'active_view':{'choice_review':True,'crop_pose_fallback':True},'habitat':{'simulator':{
            'agents':{'main_agent':{'sim_sensors':{'depth_sensor':vars(self.f.v.sensor)}}}}}}
        self.trace = [dict(step=i,action=2 if i else None,pose=[0.,0.,0.,math.pi/4,0.]) for i in range(4)]
        self.row = dict(success='1',steps='3',video_path='',target='chair')
        self.log = '\n'.join([
            '[Conditional inspection trigger] id=1 ',
            '[Annular search] id=1 mode=framing step=2 radius=3.0 generated=72 safe=0 visible=0 access=0 arcs=0 accepted=0',
            '[Annular sample] id=1 sample=1/2 step=2 position=(0.0000,0.0000) yaw=0.7854',
            '[Annular sample] id=1 sample=2/2 step=3 position=(0.0000,0.0000) yaw=0.7854',
            *[f'[Candidate scores] step={i} id=1 fusion=0.7 state=2 has_qwen=1 qwen=0.4 rank=0.7 late=0' for i in (2,3)]])
        (self.run/'planner.log').write_text(self.log)

    def check(self, events=None):
        return audit.audit_evidence(self.run,self.f.v.directory,self.events if events is None else events,
                                    self.row,self.cfg,self.trace,self.bill)

    def test_recorded_fallback_source_joint_images_and_ledger(self):
        result = self.check()
        self.assertEqual(result['crop_pose_fallbacks'],[1])
        self.assertEqual([c['image_count'] for c in result['calls']],[1,3])

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


if __name__ == '__main__': unittest.main()
