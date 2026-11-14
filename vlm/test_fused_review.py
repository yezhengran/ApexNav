"""Real mapper-bound images and active inspection, with no network/model calls."""
import json,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from vlm.fused_review import FusedTargetReviewer

class FusedReviewTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
  self.cfg=SimpleNamespace(mode='selected_review',fused_review=True,stop_review=False,
    itm_confirm_threshold=.65,itm_reject_threshold=.2,max_inspection_steps=48,max_stationary_steps=24)
  self.images=[];self.answer=.4
  def verify(image,text):self.images.append(image.copy());return self.answer
  self.v=FusedTargetReviewer(self.cfg,SimpleNamespace(normalize_depth=False,min_depth=.1,max_depth=10,hfov=90),'bed',verify,self.tmp.name,1,episode_token=42)
 def frame(self,step,score=.8,ident=1,x=0.,yaw=0.,size=16,label=0,token=42,assign=True):
  rgb=np.full((64,64,3),step%256,np.uint8);mask=np.zeros((64,64),bool);mask[20:20+size,20:20+size]=True
  result=self.v.observe(rgb,{'gps':np.array([0,0,-x]),'compass':np.array([yaw]),'depth':np.full((64,64,1),2.)},[mask],[score],[label],0.,step,action=1)
  if assign:self.v.associate(token,step,[ident],['assigned' if ident else 'filtered'],[1,1,3,3])
  return result
 def begin(self,phase=3.,ident=1,fused=.7,threshold=.65):
  return self.v.begin([ident,fused,threshold,2,2,1,1,1,3,3,phase])
 def events(self,kind):return [e for e in map(json.loads,(self.v.directory/'events.jsonl').read_text().splitlines()) if e['event']==kind]
 def test_fused_qualification_not_image_threshold(self):
  self.frame(1,.95)
  with self.assertRaises(ValueError):self.begin(fused=.64)
  self.assertEqual(len(self.images),0)
  self.frame(2,.61,ident=2)
  self.answer=.8;self.assertEqual(self.begin(ident=2),(2,self.v.CONFIRMED))
  self.assertTrue((self.images[0]==2).all())
 def test_best_crop_bound_to_object_survives_more_than_48_candidates(self):
  self.frame(1,.95);self.frame(2,.8)
  for n in range(3,55):self.frame(n,.99,ident=n)
  self.answer=.8;self.begin()
  self.assertTrue((self.images[0]==1).all())
  self.assertTrue(all('crop' not in e for rows in self.v.frames.values() for e in rows))
  self.assertEqual(self.events('verification_source')[0]['detector_score'],.95)
 def test_negative_and_uncertain_both_wait_for_planner_inspection(self):
  for answer in (.1,.4):
   with self.subTest(answer=answer):
    self.answer=answer;ident=1 if answer==.1 else 2
    self.frame(ident,.8,ident=ident)
    self.assertEqual(self.begin(ident=ident),(ident,self.v.UNCERTAIN))
    self.assertIn(ident,self.v.inspection_readiness());self.assertEqual(self.v.request(),[])
    self.assertEqual(self.v.first_decisions[ident],self.v.REJECTED if answer==.1 else self.v.UNCERTAIN)
 def test_new_lower_score_view_is_used_only_after_actual_inspection(self):
  self.frame(1,.99);self.begin();self.assertEqual(self.v.readiness(),[])
  self.assertIsNone(self.begin(phase=4.));self.assertTrue(self.v.request())
  self.frame(2,.6,x=.25)
  self.answer=.8
  self.assertEqual(self.frame(3,.59,x=.5),(1,self.v.CONFIRMED))
  self.assertTrue((self.images[1]==2).all());self.assertEqual(self.v.frames[1][0]['score'],.99)
  self.assertEqual(len(self.images),2);self.assertEqual(self.v.inspection_readiness(),[])
  self.assertGreater(self.v.stats['inspection_actions'],0)
  with self.assertRaises(ValueError):self.begin(phase=4.)
 def test_no_novel_frame_does_not_resend_best_image(self):
  self.frame(1);self.begin();self.begin(phase=4.)
  self.frame(2);self.frame(3)
  self.assertEqual(self.v.finish_inspection(False),(1,self.v.UNCERTAIN))
  self.assertEqual(len(self.images),1);self.assertEqual(self.v.inspection_readiness(),[])
 def test_second_novel_view_ranking_highest_score(self):
  self.frame(1,.99);self.begin()
  # Natural novel views while another confirmed target can suppress inspection.
  self.frame(2,.7,x=.2);self.frame(3,.9,x=.3);self.frame(4,.8,x=.4)
  self.assertEqual([e['score'] for e in self.v.frames[1]],[.99,.9])
  self.begin(phase=4.)
  # Pre-inspection images must not finish inspection without a new actual view.
  self.frame(5,.6,x=.5);self.v.finish_inspection(True)
  self.assertTrue((self.images[1]==5).all())
 def test_confirmed_target_has_no_followup(self):
  self.frame(1);self.answer=.8;self.begin();self.frame(2,x=.3)
  self.assertEqual(self.v.readiness(),[]);self.assertEqual(self.v.inspection_readiness(),[])
 def test_wrong_mask_or_episode_or_unusable_cannot_supply_crop(self):
  self.frame(1,ident=0);self.frame(2,label=1);self.frame(3,size=7);self.frame(4,token=43)
  self.assertEqual(self.v.readiness(),[]);self.assertEqual(self.images,[])
 def test_actual_call_copy_survives_best_replacement(self):
  self.frame(1,.7);self.begin()
  source=Path(self.events('verification_source')[0]['source_file']);original=source.read_bytes()
  self.frame(2,.99,x=.25)
  self.assertEqual(source.read_bytes(),original)
 def test_unreachable_or_no_translation_is_bounded(self):
  self.frame(1);self.begin();self.begin(phase=4.)
  result=None
  for step in range(2,26):result=self.frame(step)
  self.assertEqual(result,(1,self.v.UNCERTAIN));self.assertEqual(len(self.images),1)
  self.assertEqual(self.v.inspection_readiness(),[])

if __name__=='__main__':unittest.main()
