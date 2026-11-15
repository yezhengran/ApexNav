"""Offline top-five identity, joint-request and bounded-inspection regressions."""
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

from vlm.annular_review import AnnularTargetReviewer, effective_decision
from vlm.qwen_review import QwenReviewer
from vlm.active_view import crop_target
from PIL import Image
from vlm.test_qwen_review import response, ssl_eof


class Backend:
    def __init__(self):
        self.images = []; self.joint = []; self.decision = 'uncertain'; self.quality = 'adequate'; self.score = .4
    def reply(self):
        return dict(decision=self.decision, answer=dict(decision=self.decision, quality=self.quality,
                    target_confidence=self.score))
    def __call__(self, image, target):
        self.images.append(image.copy()); return self.reply()
    def review_images(self, images, target):
        self.joint.append([image.copy() for image in images]); return self.reply()


class AnnularTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.backend = Backend()
        cfg = SimpleNamespace(mode='selected_review', fused_review=True, confidence_review=True,
            framing_review=True, annular_review=True, backend='qwen', stop_review=False,
            itm_confirm_threshold=.65, itm_reject_threshold=.2, max_inspection_steps=48, max_stationary_steps=24)
        sensor = SimpleNamespace(normalize_depth=False, min_depth=.1, max_depth=10, hfov=90, position=[0,1.25,0])
        self.v = AnnularTargetReviewer(cfg,sensor,'chair',self.backend,self.tmp.name,1,episode_token=42)

    def frame(self, step, score=.8, ident=1, yaw=0.):
        rgb=np.full((64,64,3),step%256,np.uint8); mask=np.zeros((64,64),bool);mask[20:36,20:36]=True
        self.v.observe(rgb,{'gps':np.array([0.,0.,0.]),'compass':np.array([yaw]),
                           'depth':np.full((64,64,1),2.)},[mask],[score],[0],0.,step,action=2)
        self.v.associate(42,step,[ident],['assigned'],[1,1,3,3])

    def begin(self, phase=3.):
        return self.v.begin([1,.7,.65,2,2,1,1,1,3,3,phase])

    def test_strict_top5_keeps_high_scores_not_new_view(self):
        for i, score in enumerate([.9,.8,.7,.6,.5,.4,.95],1): self.frame(i,score)
        rows=self.v.frames[1]
        self.assertEqual([r['step'] for r in rows],[7,1,2,3,4])
        self.assertEqual(len(list(self.v.trace_dir.glob('*.npz'))),5)
        self.frame(8,1.,ident=2)
        self.assertEqual([r['step'] for r in self.v.frames[1]],[7,1,2,3,4])
        self.assertEqual(len(self.v.frames[2]),1)

    def test_second_joint_uses_exact_top5_and_latest_score(self):
        for i in range(1,6): self.frame(i,1.-i*.03)
        self.assertEqual(self.begin(),(1,self.v.UNCERTAIN))
        self.assertEqual(int(self.backend.images[0][0,0,0]),1)
        self.begin(4.); self.assertEqual(len(self.v.request()),8)
        self.frame(6,.1,yaw=math.pi/4); self.backend.score=.7
        self.assertEqual(self.v.finish_inspection(True),(1,self.v.UNCERTAIN))
        self.assertEqual([int(x[0,0,0]) for x in self.backend.joint[0]],[1,2,3,4,5])
        self.assertEqual(self.v.qwen_scores[1],.7)
        self.assertEqual(self.v.inspection_readiness(),[])
        with self.assertRaises(ValueError): self.begin(4.)
        events=[json.loads(x) for x in (self.v.directory/'events.jsonl').read_text().splitlines()]
        second=[e for e in events if e['event']=='verification'][-1]
        self.assertEqual(second['image_count'],5); self.assertEqual(second['new_view_images'],0)

    def test_fewer_than_five_and_clear_rejection(self):
        self.frame(1); self.backend.decision='rejected'
        self.assertEqual(self.begin(),(1,self.v.REJECTED))
        self.assertEqual(self.v.inspection_readiness(),[])

    def test_insufficient_quality_overrides_rejection_and_plant_score(self):
        self.v.label='potted plant'; self.frame(1)
        self.backend.decision='rejected';self.backend.quality='insufficient';self.backend.score=.99
        self.assertEqual(self.begin(),(1,self.v.UNCERTAIN))
        self.assertEqual(self.v.inspection_readiness(),[1])

    def test_no_route_preserves_score_without_second_call(self):
        self.frame(1);self.begin();self.begin(4.)
        self.assertEqual(self.v.finish_inspection(False),(1,self.v.UNCERTAIN))
        self.assertEqual(self.backend.joint,[]);self.assertEqual(self.v.qwen_scores[1],.4)

    def test_facing_and_fresh_association_required(self):
        self.frame(1);self.begin();self.begin(4.)
        with self.assertRaises(RuntimeError):self.v.finish_inspection(True)
        self.frame(2)
        with self.assertRaises(RuntimeError):self.v.finish_inspection(True)

    def test_source_identity_tampering_stops(self):
        self.frame(1); source=Path(self.v.frames[1][0]['source_file'])
        with np.load(source) as f:data={k:f[k] for k in f.files}
        data['candidate']=2;np.savez_compressed(source,**data)
        with self.assertRaisesRegex(RuntimeError,'lineage'):self.begin()

    def test_step_budget_keeps_first_result(self):
        self.frame(1);self.begin();self.begin(4.)
        self.frame(50,.99)
        self.assertIsNone(self.v.active);self.assertEqual(self.backend.joint,[])

    def test_late_budget_ends_detour_without_promoting_uncertainty(self):
        self.frame(390);self.begin();self.begin(4.)
        self.frame(400,.99)
        self.assertIsNone(self.v.active);self.assertEqual(self.backend.joint,[])
        self.assertEqual(self.v.histories[1]['decision'],self.v.UNCERTAIN)
        self.assertEqual(self.v.qwen_scores[1],.4)
        self.assertEqual(self.v.inspection_readiness(),[])

    def test_joint_can_contain_less_than_five_including_only_one(self):
        self.frame(1);self.begin();self.begin(4.)
        # Unassigned observation changes pose but cannot add another candidate crop.
        rgb=np.zeros((64,64,3),np.uint8)
        self.v.observe(rgb,{'gps':np.zeros(3),'compass':np.array([math.pi/4]),
                           'depth':np.ones((64,64,1))},[],[],[],0.,2,action=2)
        self.v.associate(42,2,[],[],[])
        self.v.finish_inspection(True)
        self.assertEqual(len(self.backend.joint[0]),1)

    def test_source_geometry_tampering_stops(self):
        self.frame(1);source=Path(self.v.frames[1][0]['source_file'])
        with np.load(source) as f:data={k:f[k] for k in f.files}
        data['bounds']=np.zeros(4);np.savez_compressed(source,**data)
        with self.assertRaisesRegex(RuntimeError,'pose/bounds'):self.begin()

    def test_outer_band_receipt_is_rejected(self):
        self.frame(1);self.begin();self.begin(4.)
        self.frame(2,yaw=math.pi/4);self.v.pose[:2]=[-10.,-10.]
        with self.assertRaisesRegex(RuntimeError,'outside'):self.v.finish_inspection(True)
        self.assertEqual(self.backend.joint,[])

    def test_plant_policy_boundaries(self):
        for score,quality,decision,expected in [(.60,'adequate','uncertain','confirmed'),
                (.59,'adequate','confirmed','uncertain'),(.99,'insufficient','confirmed','uncertain'),
                (.9,'adequate','rejected','rejected')]:
            self.assertEqual(effective_decision({'answer':dict(target_confidence=score,quality=quality,
                             decision=decision)},'potted plant'),expected)

    def test_saved_joint_inputs_reconstruct_and_match_http_ledger(self):
        ledger=Path(self.tmp.name)/'budget.json'
        ledger.write_text(json.dumps(dict(limit_cny=20,initial_spent_cny=0,calls={})))
        with patch('vlm.qwen_review.read_credentials',return_value='fake-key'):
            self.v.multiview_backend=QwenReviewer(ledger,target_confidence=True)
        answer=response('uncertain')
        body=json.loads(answer['choices'][0]['message']['content']);body['target_confidence']=.4
        answer['choices'][0]['message']['content']=json.dumps(body)
        with patch('vlm.qwen_review.requests.post',return_value=Mock(status_code=200,json=lambda:answer)):
            for i in range(1,6):self.frame(i,1.-i*.03)
            self.begin();self.begin(4.);self.frame(6,.1,yaw=math.pi/4)
            self.v.finish_inspection(True)
        events=[json.loads(x) for x in (self.v.directory/'events.jsonl').read_text().splitlines()]
        event=[x for x in events if x['event']=='verification'][-1]
        hashes=[]
        for row in event['sources']:
            with np.load(row['source_file'],allow_pickle=False) as p:
                self.assertEqual(int(p['candidate']),event['candidate'])
                shape=tuple(p['mask_shape'])
                mask=np.unpackbits(p['mask_bits'])[:math.prod(shape)].reshape(shape).astype(bool)
                expected=Image.fromarray(crop_target(p['rgb'],mask)[0]);expected.thumbnail((256,256))
                np.testing.assert_array_equal(np.asarray(expected),np.asarray(Image.open(row['crop'])))
            hashes.append(hashlib.sha256(Path(row['crop']).read_bytes()).hexdigest())
        calls=list(json.loads(ledger.read_text())['calls'].values())
        self.assertEqual(len(calls),2)
        self.assertEqual(hashes,event['model_reply']['image_sha256s'])
        self.assertEqual(hashes,calls[-1]['image_sha256s'])


class MultiImageRequestTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.ledger=Path(self.tmp.name)/'budget.json'
        self.ledger.write_text(json.dumps(dict(limit_cny=20,initial_spent_cny=0,calls={})))
        with patch('vlm.qwen_review.read_credentials',return_value='fake-key'):
            self.client=QwenReviewer(self.ledger,target_confidence=True)
        self.images=[np.full((32,32,3),i,np.uint8) for i in range(5)]

    def answer(self):
        r=response('uncertain');a=json.loads(r['choices'][0]['message']['content']);a['target_confidence']=.55
        r['choices'][0]['message']['content']=json.dumps(a);return r

    def test_payload_and_ordered_hashes(self):
        with patch('vlm.qwen_review.requests.post',return_value=Mock(status_code=200,json=self.answer)) as post:
            result=self.client.review_images(self.images,'chair')
        parts=post.call_args.kwargs['json']['messages'][0]['content']
        self.assertEqual(len(parts),6);self.assertEqual(result['image_count'],5)
        self.assertEqual(len(set(result['image_sha256s'])),5)
        self.assertEqual(result['image_sha256'],result['image_sha256s'][0])
        self.assertIn('ONE candidate',parts[-1]['text'])

    def test_bounds_fail_before_reservation(self):
        for imgs in [[],self.images+[self.images[0]],[np.zeros((257,20,3),np.uint8)]]:
            with self.assertRaises(ValueError):self.client.review_images(imgs,'chair')
        self.assertEqual(json.loads(self.ledger.read_text())['calls'],{})

    def test_single_image_joint_and_retry_reservations(self):
        with patch('vlm.qwen_review.requests.post',side_effect=[ssl_eof(),Mock(status_code=200,json=self.answer)]) as post, \
                patch('vlm.qwen_review.time.sleep'):
            result=self.client.review_images(self.images[:1],'chair')
        calls=list(json.loads(self.ledger.read_text())['calls'].values())
        self.assertEqual(len(calls),2);self.assertEqual(calls[0]['charged_cny'],.02)
        self.assertEqual(calls[0]['image_sha256s'],calls[1]['image_sha256s'])
        self.assertEqual(post.call_args_list[0],post.call_args_list[1]);self.assertEqual(result['image_count'],1)

if __name__=='__main__':unittest.main()
