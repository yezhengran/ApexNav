"""Offline regression tests; external inference is mocked."""
import copy
import hashlib
import json
import math
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np
from PIL import Image
from vlm.choice_review import ChoiceTargetReviewer
from vlm.qwen_review import QwenReviewer, parse_choice, QwenTransportError
from tests.unit.test_annular_review import AnnularTests, MultiImageRequestTests
from tests.unit.test_qwen_review import response, ssl_eof
from vlm.active_view import crop_target

class ChoiceTests(AnnularTests):
    def setUp(self):
        super().setUp()
        self.backend.compare_candidates = Mock(side_effect=lambda images,target,ids:
            {'answer':dict(outcome='selected',candidate_id=ids[-1],reason='best shape')})
        cfg=self.v.cfg;cfg.choice_review=True
        self.v=ChoiceTargetReviewer(cfg,self.v.sensor,'chair',self.backend,self.tmp.name,2,episode_token=42)

    def test_strict_crop_or_fusion_boundary_and_single_observation(self):
        self.frame(1,.7)
        with self.assertRaises(ValueError):self.v.begin([1,.1,.65,2,2,1,1,1,3,3,3.])
        self.frame(2,.700001)
        self.assertEqual(self.v.begin([1,.1,.65,2,2,1,1,1,3,3,3.]),(1,self.v.UNCERTAIN))
        self.assertEqual(self.v.inspection_readiness(),[1])
        with self.assertRaises(ValueError):self.v.begin([1,.1,.65,2,2,1,1,1,3,3,3.])
        self.frame(3,.2,ident=2)
        self.assertEqual(self.v.begin([2,.65,.65,2,2,1,1,1,3,3,3.]),(2,self.v.UNCERTAIN))

    def test_compare_best_image_does_not_change_individual_state(self):
        self.frame(1,.8,ident=1);self.frame(2,.9,ident=2);self.frame(3,.6,ident=2)
        before=copy.deepcopy(self.v.histories);scores=dict(self.v.qwen_scores)
        self.assertEqual(self.v.compare([1,2],1,42,3),2)
        args=self.backend.compare_candidates.call_args.args
        self.assertEqual([int(x[0,0,0]) for x in args[0]],[1,2])
        self.assertEqual(self.v.histories,before);self.assertEqual(self.v.qwen_scores,scores)
        events=[json.loads(x) for x in (self.v.directory/'events.jsonl').read_text().splitlines()]
        source=[x for x in events if x['event']=='candidate_comparison'][0]
        for row in source['sources']:
            with np.load(row['source_file'],allow_pickle=False) as p:
                self.assertEqual(int(p['candidate']),row['candidate'])
                mask=np.unpackbits(p['mask_bits'])[:math.prod(p['mask_shape'])].reshape(tuple(p['mask_shape'])).astype(bool)
                expected=Image.fromarray(crop_target(p['rgb'],mask)[0]);expected.thumbnail((256,256))
                np.testing.assert_array_equal(np.asarray(expected),np.asarray(Image.open(row['crop'])))
        with self.assertRaises(RuntimeError):self.v.compare([1,2],1,42,3)

    def test_compare_rejects_foreign_stale_duplicate_and_rejected(self):
        self.frame(1,ident=1);self.frame(2,ident=2)
        for ids,token,obs in [([1,3],42,2),([1,2],99,2),([1,2],42,1),([1,1],42,2),([2,1],42,2)]:
            with self.assertRaises(RuntimeError):self.v.compare(ids,1,token,obs)
        self.v.histories[2]={'calls':1,'decision':self.v.REJECTED}
        with self.assertRaises(RuntimeError):self.v.compare([1,2],1,42,2)
        self.backend.compare_candidates.assert_not_called()

    def test_no_frontier_gate_requires_switch_and_completed_reviews(self):
        self.frame(1,.8,ident=1);self.frame(2,.9,ident=2)
        with self.assertRaisesRegex(RuntimeError,'reviewed'):self.v.compare([1,2],1,42,2,reason='no_frontier')
        self.v.cfg.choice_no_frontier=True
        with self.assertRaisesRegex(RuntimeError,'reviewed'):self.v.compare([1,2],1,42,2,reason='no_frontier')
        for i in [1,2]:
            self.v.histories[i]={'calls':1,'decision':self.v.UNCERTAIN};self.v.qwen_scores[i]=.4
        self.assertEqual(self.v.compare([1,2],1,42,2,reason='no_frontier'),2)
        self.assertEqual(self.v.histories[2]['decision'],self.v.UNCERTAIN)

    def test_all_candidates_participate_in_bounded_groups(self):
        for ident in range(1,13):self.frame(ident,ident=ident)
        self.assertEqual(self.v.compare(list(range(1,13)),1,42,12),12)
        groups=[c.args[2] for c in self.backend.compare_candidates.call_args_list]
        self.assertEqual(groups,[[1,2,3,4,5],[6,7,8,9,10],[11,12],[5,10,12]])
        self.assertEqual(self.v.comparison_count,4)

class ChoiceTransportTests(MultiImageRequestTests):
    def choice_answer(self, ident=4):
        data=response();data['choices'][0]['message']['content']=json.dumps(
            dict(outcome='selected',candidate_id=ident,reason='best match'))
        return data

    def test_comparison_payload_ids_order_and_ledger(self):
        with patch('vlm.qwen_review.requests.post',return_value=Mock(status_code=200,json=self.choice_answer)) as post:
            result=self.client.compare_candidates(self.images[:2],'tv',[2,4])
        self.assertEqual(result['candidate_ids'],[2,4]);self.assertEqual(result['image_count'],2)
        self.assertEqual(result['image_selection'],'candidate_best_comparison_v1')
        text=post.call_args.kwargs['json']['messages'][0]['content'][-1]['text']
        self.assertIn('DIFFERENT',text);self.assertIn('[2, 4]',text)
        self.assertIn('exactly ONE',text)
        ledger=json.loads(self.ledger.read_text())['calls'][result['call_id']]
        self.assertEqual(ledger['image_sha256s'],result['image_sha256s'])

    def test_comparison_has_no_unselected_or_foreign_answer(self):
        for ident in (0,3,True):
            with self.assertRaises(ValueError):parse_choice(self.choice_answer(ident),[2,4])
        data=self.choice_answer();a=json.loads(data['choices'][0]['message']['content']);a['outcome']='uncertain'
        data['choices'][0]['message']['content']=json.dumps(a)
        with self.assertRaises(ValueError):parse_choice(data,[2,4])

    def test_comparison_transport_chain_preserves_all_images(self):
        with patch('vlm.qwen_review.requests.post',side_effect=[ssl_eof(),ssl_eof(),Mock(status_code=200,json=self.choice_answer)]) as post, patch('vlm.qwen_review.time.sleep'):
            result=self.client.compare_candidates(self.images[:2],'tv',[2,4])
        ledger=json.loads(self.ledger.read_text())['calls']
        self.assertEqual(len(ledger),3)
        self.assertTrue(all(v['candidate_ids']==[2,4] and v['image_sha256s']==result['image_sha256s'] for v in ledger.values()))
        self.assertEqual([v['state'] for v in ledger.values()],['error','error','completed'])
        self.assertTrue(all(c==post.call_args_list[0] for c in post.call_args_list))

    def test_comparison_invalid_input_does_not_reserve(self):
        for ids in ([1],[1,1],[1,2,3,4,5,6],[True,2],[0,2]):
            with self.assertRaises(ValueError):self.client.compare_candidates(self.images[:len(ids)],'tv',ids)
        self.assertEqual(json.loads(self.ledger.read_text())['calls'],{})
