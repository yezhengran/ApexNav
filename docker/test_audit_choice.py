import copy
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from docker import audit_annular as A
from vlm.test_choice_review import ChoiceTests
from vlm.qwen_review import QwenReviewer
from vlm.test_qwen_review import response


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
