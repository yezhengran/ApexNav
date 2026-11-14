"""Offline invariants for the optional facing/score profile."""
import json, math, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from vlm.confidence_review import ConfidenceTargetReviewer
from vlm.qwen_review import parse_answer
from vlm.test_qwen_review import response

class ConfidenceTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
  self.images=[];self.score=.3
  cfg=SimpleNamespace(mode='selected_review',fused_review=True,confidence_review=True,backend='qwen',
    stop_review=False,itm_confirm_threshold=.65,itm_reject_threshold=.2,max_inspection_steps=48,max_stationary_steps=24)
  def verify(image,target):
   self.images.append(image.copy());return {'decision':'uncertain','answer':{'target_confidence':self.score}}
  self.v=ConfidenceTargetReviewer(cfg,SimpleNamespace(normalize_depth=False,min_depth=.1,max_depth=10,hfov=90),
    'chair',verify,self.tmp.name,1,episode_token=42)
 def frame(self,step,score=.8,x=0.,yaw=0.,ident=1,assign=True):
  rgb=np.full((64,64,3),step,np.uint8);mask=np.zeros((64,64),bool);mask[20:36,20:36]=True
  r=self.v.observe(rgb,{'gps':np.array([0,0,-x]),'compass':np.array([yaw]),'depth':np.full((64,64,1),2.)},[mask],[score],[0],0.,step,action=1)
  if assign:self.v.associate(42,step,[ident],['assigned'],[1,1,3,3])
  return r
 def begin(self,phase=3.):return self.v.begin([1,.7,.65,2,2,1,1,1,3,3,phase])
 def test_second_waits_for_facing_and_can_reuse_best(self):
  self.frame(1,.99);self.begin();self.begin(4.)
  self.frame(2,.7,x=.25);self.frame(3,.6,x=.5)
  self.assertEqual(len(self.images),1);self.assertEqual(len(self.v.frames[1]),1)
  with self.assertRaises(RuntimeError):self.v.finish_inspection(True)
  self.frame(4,.8,x=.5,yaw=math.atan2(2,1.5));self.score=.4
  self.assertEqual(self.v.finish_inspection(True),(1,self.v.UNCERTAIN))
  self.assertEqual(len(self.images),2);self.assertTrue((self.images[1]==1).all())
  self.assertEqual(self.v.qwen_scores[1],.4)
  with self.assertRaises(ValueError):self.begin(4.)
 def test_new_best_replaces_old_and_candidate_identity_is_separate(self):
  self.frame(1,.8);self.begin();self.begin(4.)
  self.frame(2,.99,x=.5,yaw=math.atan2(2,1.5));self.frame(3,1.,ident=2,x=.5,yaw=math.atan2(2,1.5))
  self.v.finish_inspection(True);self.assertTrue((self.images[1]==2).all())
  self.assertEqual([len(v) for v in self.v.frames.values()],[1,1])
  self.assertEqual(len(list(self.v.trace_dir.glob('*.npz'))),2)
 def test_route_failure_never_triggers_second(self):
  self.frame(1);self.begin();self.begin(4.);self.frame(2,.99,x=.5)
  self.v.finish_inspection(False);self.assertEqual(len(self.images),1);self.assertEqual(self.v.qwen_scores[1],.3)
 def test_stationary_budget_does_not_call(self):
  self.frame(1);self.begin();self.begin(4.)
  for step in range(2,26):self.frame(step)
  self.assertIsNone(self.v.active);self.assertEqual(len(self.images),1)
 def test_post_motion_receipt_required(self):
  self.frame(1,yaw=math.pi/4);self.begin();self.begin(4.)
  with self.assertRaises(RuntimeError):self.v.finish_inspection(True)
  self.frame(2,yaw=math.pi/4,assign=False)
  with self.assertRaises(RuntimeError):self.v.finish_inspection(True)
 def test_strict_schema_and_legacy_unchanged(self):
  data=response();self.assertNotIn('target_confidence',parse_answer(data))
  with self.assertRaises(ValueError):parse_answer(data,True)
  for value in [-.01,1.01,float('nan'),float('inf'),True,'0.5',None]:
   answer=json.loads(response()['choices'][0]['message']['content']);answer['target_confidence']=value
   data['choices'][0]['message']['content']=json.dumps(answer)
   with self.subTest(value=value),self.assertRaises(ValueError):parse_answer(data,True)
  answer['target_confidence']=.5;data['choices'][0]['message']['content']=json.dumps(answer)
  self.assertEqual(parse_answer(data,True)['target_confidence'],.5)
  with self.assertRaises(ValueError):parse_answer(data)

if __name__=='__main__':unittest.main()
