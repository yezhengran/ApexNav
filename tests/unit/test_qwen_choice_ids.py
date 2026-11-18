"""Offline regression tests; external inference is mocked."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import Mock, patch
import numpy as np
from PIL import Image
from vlm import qwen_review as Q
from tests.unit import test_qwen_review as fixtures
from tests.unit import test_choice_review as choice_fixtures
from docker import audit_annular as audit
from tests.unit import test_audit as audit_fixtures
from docker.run_batch import evaluation_command

POLICY = 'image_order_v1'

def reply(ident=5):
    data = fixtures.response()
    data['choices'][0]['message']['content'] = json.dumps(
        dict(outcome='selected', candidate_id=ident, reason='best visible evidence'))
    return data

class ChoiceIdTests(unittest.TestCase):
    def setUp(self):
        fixtures.QwenTest.setUp(self)
        with patch.object(Q, 'read_credentials', return_value='fake-key'):
            self.client = Q.QwenReviewer(self.ledger, target_confidence=True,
                                         prompt_policy='semantic_guard_v1', choice_id_policy=POLICY)
    calls = fixtures.QwenTest.calls

    def request(self, ids, local_id):
        data = reply(local_id)
        with patch.object(Q.requests, 'post', return_value=Mock(status_code=200, json=lambda:data)) as post:
            result = self.client.compare_candidates([self.rgb]*len(ids), 'couch', ids)
        return result, post.call_args.kwargs['json']

    def test_646_sparse_ids_and_overlapping_id_domains_map_exactly_once(self):
        for ids, local_id, expected in [([1,2,3,6,7],5,7), ([3,5,9],3,9), ([101,90],1,101)]:
            with self.subTest(ids=ids):
                result, payload = self.request(ids, local_id)
                self.assertEqual(result['answer']['candidate_id'], expected)
                self.assertEqual(result['raw_answer']['candidate_id'], local_id)
                self.assertEqual(result['candidate_ids'], ids)
                self.assertEqual(result['api_candidate_ids'], list(range(1,len(ids)+1)))
                self.assertEqual(result['candidate_id_map'], [dict(image_index=i,candidate_id=v) for i,v in enumerate(ids,1)])
                self.assertEqual(self.calls()[result['call_id']]['raw_answer'],result['raw_answer'])
                prompt = payload['messages'][0]['content'][-1]['text']
                self.assertIn('1-based image positions',prompt)
                self.assertIn('IDs '+str(list(range(1,len(ids)+1))),prompt)
                self.assertNotIn('IDs '+str(ids),prompt)
                self.assertEqual(payload['max_tokens'],192)
                self.assertEqual(payload['response_format'],{'type':'json_object'})

    def test_new_policy_rejects_out_of_range_or_wrong_types_without_global_fallback(self):
        for invalid in (0,6,7,True,1.0,'5',None):
            data=reply(invalid)
            with patch.object(Q.requests,'post',return_value=Mock(status_code=200,json=lambda:data)) as post:
                with self.assertRaisesRegex(RuntimeError,'comparison_choice'):
                    self.client.compare_candidates([self.rgb]*5,'couch',[1,2,3,6,7])
                self.assertEqual(post.call_count,1)
            bill=list(self.calls().values())[-1]
            self.assertEqual(bill['state'],'error');self.assertEqual(bill['charged_cny'],.02)
            self.assertFalse(bill['retry_scheduled'])

    def test_legacy_still_rejects_ordinal_and_keeps_original_prompt_hash(self):
        ids=[1,2,3,6,7]
        with self.assertRaises(ValueError):Q.parse_choice(reply(5),ids)
        prompt=Q.build_prompt('couch',candidate_ids=ids,policy='semantic_guard_v1')
        self.assertEqual(hashlib.sha256(prompt.encode()).hexdigest(),'29f1ad892a21607cbb3c144afdc7f0f4144826200618dc31b41c18b69cd68495')
        with patch.object(Q,'read_credentials',return_value='fake-key'):
            old=Q.QwenReviewer(self.ledger,prompt_policy='semantic_guard_v1')
        with patch.object(Q.requests,'post',return_value=Mock(status_code=200,json=lambda:reply(7))) as post:
            out=old.compare_candidates([self.rgb]*5,'couch',ids)
        self.assertEqual(out['answer']['candidate_id'],7)
        self.assertNotIn('raw_answer',out)
        self.assertEqual(post.call_args.kwargs['json']['messages'][0]['content'][-1]['text'],prompt)

    def test_single_and_joint_review_payloads_are_identical_under_both_id_policies(self):
        for target in ['chair','couch','tv','bed','potted plant','toilet']:
            for mode in ['single','joint']:
                data=fixtures.response();answer=json.loads(data['choices'][0]['message']['content']);answer['target_confidence']=.4
                data['choices'][0]['message']['content']=json.dumps(answer)
                payloads=[]
                for policy in ['global_v1',POLICY]:
                    with patch.object(Q,'read_credentials',return_value='fake-key'):
                        client=Q.QwenReviewer(self.ledger,target_confidence=True,prompt_policy='semantic_guard_v1',choice_id_policy=policy)
                    with patch.object(Q.requests,'post',return_value=Mock(status_code=200,json=lambda:data)) as post:
                        result=client(self.rgb,target) if mode=='single' else client.review_images([self.rgb]*3,target)
                    payloads.append(post.call_args.kwargs['json']);self.assertEqual(result['answer'],answer)
                    self.assertNotIn('choice_id_policy',result)
                self.assertEqual(*payloads)

    def test_transport_retries_keep_local_ids_global_map_and_image_order(self):
        with patch.object(Q.requests,'post',side_effect=[fixtures.ssl_eof(),Mock(status_code=200,json=lambda:reply(5))]) as post,patch.object(Q.time,'sleep'):
            result=self.client.compare_candidates([self.rgb]*5,'couch',[1,2,3,6,7])
        self.assertEqual(result['answer']['candidate_id'],7)
        self.assertEqual(post.call_args_list[0],post.call_args_list[1])
        for bill in self.calls().values():
            self.assertEqual(bill['candidate_id_map'],result['candidate_id_map'])
            self.assertEqual(bill['api_candidate_ids'],[1,2,3,4,5])

    def test_comparison_audit_rejects_mapping_raw_answer_or_policy_tampering(self):
        result,_=self.request([1,2,3,6,7],5)
        crops=[]
        for i in range(5):
            p=Path(self.temp.name)/f'{i}.png';Image.fromarray(self.rgb).save(p);crops.append(str(p))
        event=dict(model_reply=result,candidate_ids=[1,2,3,6,7],outcome='selected',selected_candidate=7,crops=crops)
        ledger=json.loads(self.ledger.read_text())
        audit.validate_comparison_http(event,ledger,'couch','semantic_guard_v1',POLICY)
        for field in ['candidate_id_map','api_candidate_ids','raw_answer','choice_id_policy','answer']:
            e=copy.deepcopy(event);b=copy.deepcopy(ledger);changed=e['model_reply']
            if field=='candidate_id_map':changed[field][4]['candidate_id']=6
            elif field=='api_candidate_ids':changed[field][4]=7
            elif field=='raw_answer':changed[field]['candidate_id']=4
            elif field=='choice_id_policy':changed[field]='global_v1'
            else:changed[field]['candidate_id']=6;e['selected_candidate']=6
            b['calls'][result['call_id']][field]=copy.deepcopy(changed[field])
            with self.subTest(field=field),self.assertRaises(AssertionError):
                audit.validate_comparison_http(e,b,'couch','semantic_guard_v1',POLICY)
        # Python considers bool/int/float values equal; the wire protocol does not.
        for destination in ('ledger', 'reply'):
            for field in ('candidate_id_map', 'api_candidate_ids', 'raw_answer'):
                e=copy.deepcopy(event);b=copy.deepcopy(ledger)
                changed=b['calls'][result['call_id']] if destination=='ledger' else e['model_reply']
                if field=='candidate_id_map':changed[field][0]['image_index']=True
                elif field=='api_candidate_ids':changed[field][0]=1.0
                else:changed[field]['candidate_id']=5.0
                with self.subTest(destination=destination,field=field),self.assertRaises(AssertionError):
                    audit.validate_comparison_http(e,b,'couch','semantic_guard_v1',POLICY)
        with self.assertRaises(AssertionError):audit.validate_comparison_http(event,ledger,'couch','semantic_guard_v1')

    def test_multi_round_tournament_rebinds_local_ids_and_preserves_candidate_states(self):
        f=choice_fixtures.ChoiceTests();f.setUp();self.addCleanup(f.doCleanups)
        f.v.multiview_backend=self.client
        for ident in range(1,13):f.frame(ident,ident=ident)
        before=copy.deepcopy(f.v.histories);scores=dict(f.v.qwen_scores)
        outputs=iter([reply(5),reply(5),reply(2),reply(3)])
        with patch.object(Q.requests,'post',side_effect=lambda *a,**kw:Mock(status_code=200,json=lambda:next(outputs))):
            self.assertEqual(f.v.compare(list(range(1,13)),1,42,12),12)
        self.assertEqual(f.v.histories,before);self.assertEqual(f.v.qwen_scores,scores)
        bills=list(self.calls().values())
        self.assertEqual([b['candidate_ids'] for b in bills],[[1,2,3,4,5],[6,7,8,9,10],[11,12],[5,10,12]])
        self.assertEqual([b['answer']['candidate_id'] for b in bills],[5,10,12,12])

    def test_full_evidence_audit_requires_matching_config_and_experiment_spec(self):
        f=audit_fixtures.ChoiceAuditTest();self.addCleanup(f.doCleanups)
        with patch('tests.unit.test_audit.QwenReviewer',side_effect=lambda *a,**kw:Q.QwenReviewer(*a,**kw,choice_id_policy=POLICY)):f.setUp()
        f.cfg['active_view']['qwen_choice_id_policy']=POLICY
        spec=f.run/'experiment.json';spec.write_text(json.dumps({'review':{'choice_id_policy':POLICY}}))
        self.assertEqual(len(f.audit()['comparisons']),1)
        f.cfg['active_view']['qwen_choice_id_policy']='global_v1'
        with self.assertRaises(AssertionError):f.audit()
        f.cfg['active_view']['qwen_choice_id_policy']=POLICY;spec.write_text(json.dumps({'review':{}}))
        with self.assertRaises(AssertionError):f.audit()

    def test_cli_adds_only_explicit_policy_and_rejects_unsupported_combinations(self):
        options=dict(method='selected_review',review_profile='linked_choice',review_backend='qwen',step_sync=True,qwen_prompt_policy='semantic_guard_v1')
        old=evaluation_command('hm3dv2',1,'/tmp/run',True,**options)
        new=evaluation_command('hm3dv2',1,'/tmp/run',True,**options,qwen_choice_id_policy=POLICY)
        flag='+active_view.qwen_choice_id_policy='+POLICY
        self.assertIn(flag,new);self.assertEqual(old,[x for x in new if x!=flag])
        for change in [dict(method='baseline'),dict(review_profile='linked_annular'),dict(review_backend='blip2')]:
            with self.assertRaises(ValueError):evaluation_command('hm3dv2',1,'/tmp/run',True,**{**options,**change},qwen_choice_id_policy=POLICY)
        for args in [['--qwen-choice-id-policy',POLICY],['--qwen-choice-id-policy','typo']]:
            p=subprocess.run([sys.executable,'docker/run_batch.py','--dry-run',*args],capture_output=True,text=True)
            self.assertNotEqual(p.returncode,0)
        with patch.object(Q,'read_credentials') as read:
            with self.assertRaises(ValueError):Q.QwenReviewer('/missing',choice_id_policy='typo')
            read.assert_not_called()
