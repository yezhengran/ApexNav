"""Geometry, provenance and live reviewer gates without a model request."""
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from vlm.framing_review import FramingTargetReviewer, framing_distance, object_envelope
from docker.run_batch import evaluation_command


class GeometryTests(unittest.TestCase):
    def envelope(self, low=.7, high=1.1, clipped=False):
        return dict(low=[1.9, -.2, low], high=[2.1, .2, high], border_clipped=clipped)

    def distance(self, envelope=None, height=.88, extent=.3):
        return framing_distance(extent, envelope, height, 0, 79, (480, 640))

    def test_high_and_low_objects_require_more_distance(self):
        middle = self.distance(self.envelope())['radius']
        self.assertGreater(self.distance(self.envelope(1.8, 2.2))['radius'], middle)
        self.assertGreater(self.distance(self.envelope(0, .25))['radius'], middle)

    def test_height_dominated_geometry_serializes_as_native_json(self):
        # Real first demo failed only when a NumPy height expression won max().
        for low,high in [(0,.87),(1.5,1.94),(4.,5.)]:
            result=self.distance(self.envelope(low,high))
            self.assertIs(type(result['feasible']),bool)
            self.assertEqual(json.loads(json.dumps(result,allow_nan=False)),result)

    def test_height_is_relative_to_current_camera(self):
        a = self.distance(self.envelope(1.8, 2.2), .88)
        b = self.distance(self.envelope(2.8, 3.2), 1.88)
        self.assertAlmostEqual(a['radius'], b['radius'])

    def test_extremes_fit_even_at_inner_arrival_tolerance(self):
        e = self.envelope(0, 1.7)
        d = self.distance(e, extent=.8)
        near_surface = d['radius']-.3125-d['footprint_radius']
        top = math.atan2(e['high'][2]+.17-.88, near_surface)
        bottom = math.atan2(.88-e['low'][2]+.17, near_surface)
        self.assertLessEqual(max(top, bottom), math.radians(d['vertical_half_fov_degrees'])+1e-10)
        half_width = math.asin(d['footprint_radius']/(d['radius']-.3125))
        self.assertLessEqual(half_width+math.radians(18), math.radians(d['horizontal_half_fov_degrees'])+1e-10)

    def test_clipped_frame_adds_margin_and_wide_object_increases_distance(self):
        a = self.distance(self.envelope(1.8, 2.2))
        self.assertGreater(self.distance(self.envelope(1.8, 2.2, True))['radius'], a['radius'])
        self.assertGreater(self.distance(self.envelope(), extent=1.)['radius'], self.distance(self.envelope())['radius'])

    def test_no_silent_clipping_when_geometry_exceeds_search_bound(self):
        self.assertFalse(self.distance(self.envelope(4, 5))['feasible'])
        self.assertFalse(framing_distance(.3, self.envelope(), .88, math.pi/2, 79, (480,640))['feasible'])

    def test_missing_depth_is_explicit_standoff(self):
        d = self.distance()
        self.assertEqual(d['reason'], 'missing_depth_standoff')
        self.assertGreaterEqual(d['radius'], 1.5+d['footprint_radius']+.3125)

    def packet(self, normalized=False):
        mask = np.zeros((100,100), bool); mask[25:75,25:75] = True
        depth = np.full((100,100), .2 if normalized else 2.)
        return dict(depth=depth, mask_bits=np.packbits(mask), mask_shape=mask.shape,
                    pitch=0., pose=[0.,0.,0.], camera_height=.88, bounds=[1.9,-1.1,2.1,1.1])

    def test_real_projection_outliers_and_normalized_depth(self):
        s=SimpleNamespace(normalize_depth=False,min_depth=0.,max_depth=10.,hfov=90.)
        p=self.packet();a=object_envelope(p,s)
        self.assertAlmostEqual(a['centroid'][0], 2.)
        self.assertAlmostEqual(a['centroid'][2], .9)
        p['depth'][25,25]=float('nan');p['depth'][26,25]=9.
        b=object_envelope(p,s)
        self.assertLess(abs(a['high'][2]-b['high'][2]), .05)
        s.normalize_depth=True;c=object_envelope(self.packet(True),s)
        np.testing.assert_allclose(a['low'],c['low'])

    def test_other_map_cluster_and_missing_depth_are_not_used(self):
        p=self.packet();s=SimpleNamespace(normalize_depth=False,min_depth=0.,max_depth=10.,hfov=90.)
        p['bounds']=[10,10,11,11];self.assertIsNone(object_envelope(p,s))
        p=self.packet();p['depth'][:]=0;self.assertIsNone(object_envelope(p,s))

    def test_profiles_preserve_old_flags(self):
        for profile in ['linked_fused','linked_confidence','linked_framing']:
            cmd=evaluation_command('hm3dv2',1,'/tmp/demo',True,method='selected_review',review_profile=profile)
            self.assertEqual('+active_view.framing_review=true' in cmd,profile=='linked_framing')
            self.assertEqual('+active_view.confidence_review=true' in cmd,profile!='linked_fused')


class ReviewerTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.calls=[]
        cfg=SimpleNamespace(mode='selected_review',fused_review=True,confidence_review=True,framing_review=True,
            backend='qwen',stop_review=False,itm_confirm_threshold=.65,itm_reject_threshold=.2,
            max_inspection_steps=48,max_stationary_steps=24)
        sensor=SimpleNamespace(normalize_depth=False,min_depth=0.,max_depth=10.,hfov=79.,position=[0,.88,0])
        def verify(image,target):
            self.calls.append(image.copy());return {'decision':'uncertain','answer':{'target_confidence':.3}}
        self.v=FramingTargetReviewer(cfg,sensor,'tv',verify,self.tmp.name,1,episode_token=42)
        self.frame(1)
        self.v.begin([1,.7,.45,2,0,.5,1.75,-.25,2.25,.25,3.])
        self.v.begin([1,.7,.45,2,0,.5,1.75,-.25,2.25,.25,4.])

    def frame(self,step,x=0,yaw=0):
        mask=np.zeros((100,100),bool);mask[40:60,40:60]=True
        self.v.observe(np.full((100,100,3),step,np.uint8),
            {'gps':[0,0,-x],'compass':[yaw],'depth':np.full((100,100,1),2.)},[mask],[.99 if step==1 else .5],[0],0,step,action=1)
        self.v.associate(42,step,[1],['assigned'],[1.75,-.25,2.25,.25])

    def test_request_and_saved_geometry_bind_exact_mask(self):
        self.assertEqual(len(self.v.request()),7)
        events=[json.loads(s) for s in (self.v.directory/'events.jsonl').read_text().splitlines()]
        g=next(e for e in events if e['event']=='inspection_geometry')
        with np.load(g['source_file']) as p:
            self.assertEqual(float(p['camera_height']),.88)
            self.assertEqual([int(p[k]) for k in ['candidate','observation','mask_index']],[1,1,0])

    def test_far_facing_second_review_still_uses_best_historical_image(self):
        radius=self.v.active['radius'];self.frame(2,x=2-radius)
        self.v.finish_inspection(True)
        self.assertEqual(len(self.calls),2);self.assertTrue((self.calls[-1]==1).all())

    def test_too_close_arrival_never_calls_second(self):
        self.frame(2,x=1.8);self.v.finish_inspection(True)
        self.assertEqual(len(self.calls),1);self.assertIsNone(self.v.active)


if __name__=='__main__':unittest.main()
