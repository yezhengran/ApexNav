"""Check opt-in prompt transport, provenance, strict audits and legacy compatibility."""
import copy
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
import unittest
from unittest.mock import Mock, patch
import numpy as np
from vlm import qwen_review as Q
from vlm.test_qwen_review import response
from docker import audit_annular as A
from docker.run_batch import evaluation_command
from docker import test_audit_annular as category_fixtures
from docker import test_audit_choice as choice_fixtures

POLICY='semantic_guard_v1'

class SemanticPromptTests(unittest.TestCase):
    def test_legacy_couch_prompt_matches_historical_http_hash(self):
        prompt=Q.build_prompt('couch',target_confidence=True)
        self.assertEqual(hashlib.sha256(prompt.encode()).hexdigest(),'332ec79df344601396b9ad9334c3cd523572e21f9b3d2628023e5455cdf65bf6')

    def test_unrelated_categories_keep_exact_prompt_bytes(self):
        for target in ['bed','potted plant','toilet']:
            for kwargs in [dict(target_confidence=True),dict(target_confidence=True,multiview=True,count=5),dict(candidate_ids=[2,8])]:
                self.assertEqual(Q.build_prompt(target,**kwargs),Q.build_prompt(target,policy=POLICY,**kwargs))

    def test_opt_in_single_joint_and_comparison_reach_http_with_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger=Path(tmp)/'budget.json';ledger.write_text(json.dumps(dict(limit_cny=20,initial_spent_cny=0,calls={})))
            with patch.object(Q,'read_credentials',return_value='fake-key'):
                client=Q.QwenReviewer(ledger,target_confidence=True,prompt_policy=POLICY)
            img=np.zeros((8,8,3),np.uint8)
            for target in ['chair','couch','tv','bed','potted plant','toilet']:
                for mode in ['single','joint','comparison']:
                    answer=response('uncertain');body=json.loads(answer['choices'][0]['message']['content']);body['target_confidence']=.4
                    if mode=='comparison':body=dict(outcome='selected',candidate_id=8,reason='best relative evidence')
                    answer['choices'][0]['message']['content']=json.dumps(body)
                    with patch.object(Q.requests,'post',return_value=Mock(status_code=200,json=lambda:answer)) as post:
                        if mode=='single':reply=client(img,target)
                        elif mode=='joint':reply=client.review_images([img,img+1],target)
                        else:reply=client.compare_candidates([img,img+1],target,[2,8])
                    payload=post.call_args.kwargs['json'];content=payload['messages'][0]['content'];text=content[-1]['text']
                    self.assertEqual(len(content)-1,1 if mode=='single' else 2)
                    self.assertEqual(reply['prompt_policy'],POLICY)
                    self.assertEqual(reply['prompt_sha256'],hashlib.sha256(text.encode()).hexdigest())
                    self.assertEqual(reply['answer'],body)  # Never override the model decision.
                    if target in ['chair','couch']:
                        self.assertIn('single-seat',text)
                        self.assertIn('uncertain' if mode!='comparison' else 'Prefer candidates',text)
                    if target=='tv':self.assertIn('art',text)
                    if mode=='comparison':self.assertIn('Choose exactly ONE',text)
                    if mode=='single':self.assertNotIn('image_sha256s',reply)
                    else:self.assertEqual(reply['image_count'],2)
            calls=json.loads(ledger.read_text())['calls'];self.assertEqual(len(calls),18)
            self.assertTrue(all(x['prompt_policy']==POLICY and x['transport_attempt']==1 for x in calls.values()))

    def test_unknown_policy_rejected_before_credentials_or_network(self):
        with patch.object(Q,'read_credentials') as read:
            with self.assertRaises(ValueError):Q.QwenReviewer('/missing',prompt_policy='typo')
            read.assert_not_called()
        with self.assertRaises(ValueError):Q.build_prompt('chair',policy='typo')

    def test_cli_override_is_only_prompt_policy(self):
        kwargs=dict(method='selected_review',review_profile='linked_choice',review_backend='qwen',qwen_budget_ledger='/tmp/mock.json',step_sync=True)
        old=evaluation_command('hm3dv2',1,'/tmp/run',True,**kwargs)
        new=evaluation_command('hm3dv2',1,'/tmp/run',True,**kwargs,qwen_prompt_policy=POLICY)
        flag='+active_view.qwen_prompt_policy='+POLICY
        self.assertIn(flag,new);self.assertEqual(old,[x for x in new if x!=flag])
        self.assertFalse(any('qwen_prompt_policy' in x for x in old))
        with self.assertRaises(ValueError):evaluation_command('hm3dv2',1,'/tmp/run',False,qwen_prompt_policy=POLICY)

    def test_real_cli_requires_qwen_and_rejects_unknown_policy(self):
        for args in [['--qwen-prompt-policy',POLICY],['--qwen-prompt-policy','typo']]:
            p=subprocess.run([sys.executable,'docker/run_batch.py','--dry-run',*args],capture_output=True,text=True)
            self.assertNotEqual(p.returncode,0)

class SemanticAuditTests(unittest.TestCase):
    def category_fixture(self):
        f=category_fixtures.AuditTests();self.addCleanup(f.doCleanups)
        with patch('docker.test_audit_annular.QwenReviewer',side_effect=lambda *a,**kw:Q.QwenReviewer(*a,**kw,prompt_policy=POLICY)):f.setUp()
        return f

    def test_real_single_and_joint_records_pass_only_matching_policy(self):
        f=self.category_fixture()
        for ordinal,event in enumerate(f.calls,1):
            A.validate_call(event,f.bill,'chair',ordinal,POLICY)
            with self.assertRaises(AssertionError):A.validate_call(event,f.bill,'chair',ordinal)
            corrupted=copy.deepcopy(f.bill);corrupted['calls'][event['model_reply']['call_id']]['prompt_sha256']='wrong'
            with self.assertRaises(AssertionError):A.validate_call(event,corrupted,'chair',ordinal,POLICY)
            corrupted=copy.deepcopy(f.bill);corrupted['calls'][event['model_reply']['call_id']].pop('prompt_policy')
            with self.assertRaises(AssertionError):A.validate_call(event,corrupted,'chair',ordinal,POLICY)

    def test_full_comparison_audit_enforces_config_spec_and_each_http(self):
        f=choice_fixtures.ChoiceAuditTest();self.addCleanup(f.doCleanups)
        with patch('docker.test_audit_choice.QwenReviewer',side_effect=lambda *a,**kw:Q.QwenReviewer(*a,**kw,prompt_policy=POLICY)):f.setUp()
        f.cfg['active_view']['qwen_prompt_policy']=POLICY
        spec=f.run/'experiment.json';spec.write_text(json.dumps({'review':{'prompt_policy':POLICY}}))
        result=f.audit();self.assertEqual(len(result['comparisons']),1)
        event=next(x for x in f.events if x['event']=='candidate_comparison')
        bad=copy.deepcopy(f.ledger);bad['calls'][event['model_reply']['call_id']]['prompt_policy']='legacy'
        with self.assertRaises(AssertionError):f.audit(bill=bad)
        f.cfg['active_view']['qwen_prompt_policy']='legacy'
        with self.assertRaises(AssertionError):f.audit()
        f.cfg['active_view']['qwen_prompt_policy']=POLICY;spec.write_text(json.dumps({'review':{}}))
        with self.assertRaises(AssertionError):f.audit()
        spec.write_text(json.dumps({'review':{'prompt_policy':POLICY}}));f.cfg['active_view']['qwen_prompt_policy']='unknown'
        with self.assertRaises(AssertionError):f.audit()

if __name__=='__main__':unittest.main()
