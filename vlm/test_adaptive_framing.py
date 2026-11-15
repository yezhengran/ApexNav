import math,unittest
from vlm.framing_review import adaptive_framing_distance,framing_distance
class AdaptiveGeometry(unittest.TestCase):
 def envelope(self,low=.05,high=2.6,center=1.1):
  return {'low':[0,0,low],'high':[1,1,high],'centroid':[.5,.5,center],'border_clipped':False}
 def distance(self,e=None,extent=.9,height=.88,pitch=0.):
  return adaptive_framing_distance(extent,e,height,pitch,79,(480,640))
 def test_tall_partial_view_has_visible_center_and_strict_standoff(self):
  e=self.envelope();r=self.distance(e);old=framing_distance(.9,e,.88,0,79,(480,640))
  self.assertLess(r['minimum_radius'],old['minimum_radius'])
  self.assertGreaterEqual(r['minimum_radius']-r['footprint_radius'],1.1)
  angle=math.atan2(1.1-.88,r['minimum_radius']-r['footprint_radius'])
  self.assertLessEqual(angle,math.radians(r['vertical_half_fov_degrees']))
  self.assertTrue(r['partial_view_allowed']);self.assertEqual(r['preferred_full_radius'],old['radius'])
 def test_high_center_requires_more_standoff(self):
  self.assertGreater(self.distance(self.envelope(2,3,2.8))['minimum_radius'],self.distance(self.envelope())['minimum_radius'])
 def test_unobservable_height_still_fails(self):
  self.assertFalse(self.distance(self.envelope(8,12,10))['feasible'])
 def test_missing_depth_keeps_conservative_original_distance(self):
  a=self.distance();b=framing_distance(.9,None,.88,0,79,(480,640))
  for key in ('radius','minimum_radius','feasible'):self.assertEqual(a[key],b[key])
  self.assertFalse(a['partial_view_allowed'])
 def test_no_pitch_action_or_invalid_geometry_shortcut(self):
  self.assertFalse(self.distance(self.envelope(),pitch=math.pi/2)['feasible'])
  with self.assertRaises(ValueError):self.distance(self.envelope(center=10))
 def test_height_translation_invariance(self):
  a=self.distance(self.envelope());b=self.distance(self.envelope(1.05,3.6,2.1),height=1.88)
  self.assertAlmostEqual(a['minimum_radius'],b['minimum_radius'])

from types import SimpleNamespace
import tempfile,numpy as np
from vlm.framing_review import FramingTargetReviewer
from vlm import test_framing_review as legacy_framing
class AdaptiveReviewer(unittest.TestCase):
 frame=legacy_framing.ReviewerTests.frame
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.calls=[]
  cfg=SimpleNamespace(mode='selected_review',fused_review=True,confidence_review=True,framing_review=True,
      adaptive_framing=True,backend='qwen',stop_review=False,itm_confirm_threshold=.65,itm_reject_threshold=.2,
      max_inspection_steps=48,max_stationary_steps=24)
  sensor=SimpleNamespace(normalize_depth=False,min_depth=0.,max_depth=10.,hfov=79.,position=[0,.88,0])
  def verify(image,target):
   self.calls.append(image.copy());return {'decision':'uncertain','answer':{'target_confidence':.3}}
  self.v=FramingTargetReviewer(cfg,sensor,'potted plant',verify,self.tmp.name,1,episode_token=42)
  self.frame(1);self.v.begin([1,.7,.45,2,0,.5,1.75,-.25,2.25,.25,3.])
  self.v.begin([1,.7,.45,2,0,.5,1.75,-.25,2.25,.25,4.])
 def test_partial_view_still_uses_best_history_and_facing_receipt(self):
  self.assertEqual(len(self.v.request()),8)
  radius=self.v.active['minimum_radius']+.01
  self.frame(2,x=2-radius)
  self.v.finish_inspection(True)
  self.assertEqual(len(self.calls),2);self.assertTrue((self.calls[-1]==1).all())
 def test_safe_distance_does_not_bypass_center_facing(self):
  radius=self.v.active['minimum_radius']+.01
  self.frame(2,x=2-radius,yaw=math.pi/2)
  with self.assertRaises(RuntimeError):self.v.finish_inspection(True)
  self.assertEqual(len(self.calls),1)
if __name__=='__main__':unittest.main()
