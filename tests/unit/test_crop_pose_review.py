"""Offline regression tests; external inference is mocked."""
import json
import math
from pathlib import Path
import unittest
from unittest.mock import Mock, patch
import numpy as np
from vlm.crop_pose_review import CropPoseTargetReviewer, crop_pose_band
from tests.unit import test_annular_review as fixture_module

class CropPoseTests(unittest.TestCase):
    def setUp(self):
        fixture=fixture_module.AnnularTests();fixture.setUp();self.addCleanup(fixture.doCleanups)
        self.fixture=fixture;self.backend=fixture.backend
        self.backend.compare_candidates=Mock()
        cfg=fixture.v.cfg;cfg.choice_review=True;cfg.crop_pose_fallback=True
        self.v=fixture.v=CropPoseTargetReviewer(cfg,fixture.v.sensor,'chair',self.backend,
                                               fixture.tmp.name,2,episode_token=42)
        self.frame=fixture.frame;self.begin=fixture.begin

    def prepare(self):
        self.frame(1,.9);self.frame(2,.8)
        self.begin();self.assertIsNone(self.begin(4.))

    def test_measured_radius_and_one_fallback_shared_budget(self):
        self.prepare();started=self.v.active['started'];primary=self.v.request()
        self.assertEqual(len(primary),8)
        self.assertIsNone(self.v.finish_inspection(False,initial_unavailable=True))
        value=self.v.request();self.assertEqual(len(value),11)
        self.assertAlmostEqual(value[3],math.sqrt(8))
        self.assertEqual(value[-3:],[0.,0.,2.]);self.assertEqual(self.v.active['started'],started)
        self.assertEqual(self.v.stats['inspections'],1);self.assertEqual(len(self.backend.images),1)
        events=[json.loads(x) for x in (self.v.directory/'events.jsonl').read_text().splitlines()]
        e=next(x for x in events if x['event']=='inspection_crop_pose_fallback')
        self.assertEqual(e['observation'],1) # Highest score, not newest/current camera pose.
        self.assertEqual(self.v.finish_inspection(False,initial_unavailable=True),(1,2))
        self.assertEqual(self.backend.joint,[]);self.assertEqual(self.v.inspection_readiness(),[])

    def test_execution_failure_does_not_retry_an_arc(self):
        self.prepare();self.assertEqual(self.v.finish_inspection(False),(1,2))
        self.assertEqual(len(self.backend.images),1)

    def test_no_fallback_after_any_real_action(self):
        self.prepare();self.frame(3,.8)
        self.assertEqual(self.v.finish_inspection(False,initial_unavailable=True),(1,2))
        self.assertEqual(self.backend.joint,[])

    def test_400_step_boundary_closes_inspection(self):
        self.frame(399);self.begin();self.begin(4.)
        self.v.finish_inspection(False,initial_unavailable=True)
        self.frame(400)
        self.assertIsNone(self.v.active);self.assertEqual(self.backend.joint,[])
        self.assertEqual(self.v.histories[1]['decision'],self.v.UNCERTAIN)

    def test_geometry_infeasible_still_uses_reference(self):
        self.frame(1);self.begin()
        original=self.v._inspection_geometry
        def unavailable(extent):
            geometry=original(extent);geometry['feasible']=False;return geometry
        with patch.object(self.v,'_inspection_geometry',side_effect=unavailable):
            self.assertIsNone(self.begin(4.))
        self.assertTrue(self.v.active['crop_pose_mode'])

    def test_successful_fallback_requires_facing_and_new_mapped_observation(self):
        self.prepare();self.v.finish_inspection(False,initial_unavailable=True)
        with self.assertRaises(RuntimeError):self.v.finish_inspection(True)
        self.frame(3,.1,yaw=math.pi/4)
        self.assertEqual(self.v.finish_inspection(True),(1,2))
        self.assertEqual(len(self.backend.joint),1);self.assertEqual(len(self.backend.joint[0]),3)

    def test_invalid_arrival_and_source_identity_stop(self):
        self.prepare();self.v.finish_inspection(False,initial_unavailable=True)
        self.v.pose=[20.,20.,0.]
        with self.assertRaisesRegex(RuntimeError,'outside'):self.v.finish_inspection(True)

    def test_source_capture_pose_tamper_rejected(self):
        self.prepare();best=min(self.v.frames[1],key=self.v._rank)
        source=Path(best['source_file'])
        with np.load(source) as p:data={k:p[k] for k in p.files}
        data['pose']=np.array([1,1,0]);np.savez_compressed(source,**data)
        with self.assertRaisesRegex(RuntimeError,'pose'):self.v.finish_inspection(False,initial_unavailable=True)

    def test_single_point_switch_and_current_mapped_facing_view(self):
        self.v.cfg.crop_pose_single_view=True
        self.frame(1,.9,yaw=math.pi/4);self.frame(2,.8,yaw=math.pi/4)
        self.begin();self.begin(4.);self.v.finish_inspection(False,initial_unavailable=True)
        self.assertEqual(self.v.request()[-1],3.)
        self.assertEqual(self.v.finish_inspection(True),(1,2))
        self.assertEqual(len(self.backend.joint),1)
        self.assertEqual(self.v.stats['inspection_actions'],0) # No fictitious motion.

    def test_single_point_still_requires_center_facing_and_association(self):
        self.v.cfg.crop_pose_single_view=True;self.prepare()
        self.v.finish_inspection(False,initial_unavailable=True)
        with self.assertRaisesRegex(RuntimeError,'facing'):self.v.finish_inspection(True)
        self.v.last_association=0
        with self.assertRaisesRegex(RuntimeError,'mapped'):self.v.finish_inspection(True)

    def test_radius_never_clamped_or_shrunk(self):
        self.assertFalse(crop_pose_band([0,0],[.4,0,0])['feasible'])
        self.assertEqual(crop_pose_band([0,0],[5,0,0])['radius'],5)
        self.assertFalse(crop_pose_band([0,0],[5,0,0])['feasible'])
        self.assertAlmostEqual(crop_pose_band([1,2],[4,6,0])['radius'],5)
        with self.assertRaises(ValueError):crop_pose_band([0,0],[math.nan,0,0])
