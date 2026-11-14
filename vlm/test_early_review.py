"""Early-review behavior with real crops/mapper receipts and a fake VLM."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from vlm.early_review import EarlyTargetReviewer


class EarlyReviewTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cfg = SimpleNamespace(mode='selected_review', early_review=True,
            early_review_threshold=.65, stop_review=False, itm_confirm_threshold=.65,
            itm_reject_threshold=.20, max_inspection_steps=48, max_stationary_steps=24)
        self.images = []
        self.answer = .8
        def verify(image, text):
            self.images.append(image.copy())
            return self.answer
        self.view = EarlyTargetReviewer(self.cfg, SimpleNamespace(normalize_depth=False,
            min_depth=.1, max_depth=10, hfov=90), 'bed', verify, self.tmp.name, 1, episode_token=42)

    def frame(self, step, score=.8, ident=1, label=0, size=16, partial=False, x=0., yaw=0., assign=True):
        rgb = np.full((64,64,3), step, dtype=np.uint8)
        mask = np.zeros((64,64), bool)
        mask[20:20+size, (0 if partial else 20):(0 if partial else 20)+size] = True
        self.view.observe(rgb, {'gps':np.array([0,0,-x]), 'compass':np.array([yaw]),
            'depth':np.full((64,64,1),2.)}, [mask], [score], [label], 0., step)
        if assign:
            self.view.associate(42, step, [ident], ['assigned' if ident else 'filtered'], [1,1,3,3])

    def begin(self, ident=1, fused=.95):
        return self.view.begin([ident,fused,.65,2,2,1,1,1,3,3,2.])

    def events(self, kind):
        return [json.loads(x) for x in (self.view.directory/'events.jsonl').read_text().splitlines()
                if json.loads(x)['event']==kind]

    def test_exact_threshold_independent_of_fused_confidence(self):
        self.frame(1, .649)
        self.assertEqual(self.view.readiness(), [])
        self.assertEqual(self.begin(), (1,self.view.PENDING_EVIDENCE))
        self.assertEqual(len(self.images),0)
        self.frame(2, .65)
        self.assertEqual(self.view.readiness(), [(1,2)])
        self.assertEqual(self.begin(fused=.99), (1,self.view.CONFIRMED))
        self.assertEqual(len(self.images),1)
        self.assertIsNone(self.view.active)

    def test_mapper_identity_and_target_label_are_required(self):
        self.frame(1,.99,ident=2)
        self.assertEqual(self.begin(1), (1,self.view.PENDING_EVIDENCE))
        self.frame(2,.99,ident=3,label=1)
        self.frame(3,.99,ident=0)
        self.frame(4,.99,ident=4,assign=False)
        self.assertEqual(self.view.readiness(), [(2,1)])
        self.assertEqual(len(self.images),0)

    def test_confidence_wins_over_size_completeness_and_recency(self):
        self.frame(1,.95,size=12,partial=True)
        self.frame(2,.85,size=32)
        self.frame(3,.70,size=32)
        self.assertEqual(len(self.view.frames[1]),2)
        self.assertEqual(self.begin(), (1,self.view.CONFIRMED))
        self.assertTrue((self.images[0] == 1).all())
        source = self.events('verification_source')[0]
        self.assertEqual(source['detector_score'], .95)
        self.assertTrue(Path(source['source_file']).is_file())
        self.assertEqual(self.events('verification')[0]['evidence_age'],2)

    def test_tiny_high_score_is_not_usable(self):
        self.frame(1,.99,size=7)
        self.assertEqual(self.view.readiness(),[])
        self.frame(2,.65,size=8)
        self.assertEqual(self.view.readiness(),[(1,2)])
        self.begin()
        self.assertTrue((self.images[0] == 2).all())

    def test_uncertain_waits_for_natural_new_view_and_obeys_two_call_limit(self):
        self.answer=.4
        self.frame(18)
        self.assertEqual(self.begin(), (1,self.view.UNCERTAIN))
        self.assertEqual(self.view.request(),[])
        self.frame(19)
        self.assertEqual(self.view.readiness(),[])
        self.frame(20,x=.25)
        self.assertEqual(self.view.readiness(),[(1,20)])
        self.assertEqual(self.begin(), (1,self.view.UNCERTAIN))
        self.frame(21,x=.5)
        self.assertEqual(self.view.readiness(),[])
        self.begin()
        self.assertEqual(len(self.images),2)
        self.assertEqual(self.view.stats['inspection_actions'],0)

    def test_decisive_result_not_scheduled_again(self):
        self.frame(18)
        self.begin()
        self.frame(19,x=.25)
        self.assertEqual(self.view.readiness(),[])
        self.assertEqual(len(self.images),1)

    def test_partial_negative_still_does_not_reject_candidate(self):
        self.answer=.1
        self.frame(18,partial=True)
        self.assertEqual(self.begin(),(1,self.view.UNCERTAIN))
        self.assertTrue(self.events('verification')[0]['partial_negative_withheld'])

    def test_reject_old_trigger_metadata_and_invalid_threshold(self):
        with self.assertRaises(ValueError):
            self.view.begin([1,.9,.65,2,2,1,1,1,3,3,1.])
        self.cfg.early_review_threshold=float('nan')
        with self.assertRaises(ValueError):
            EarlyTargetReviewer(self.cfg,self.view.sensor,'bed',lambda *a:.8,self.tmp.name,2,episode_token=43)

    def test_retained_call_source_survives_cache_eviction(self):
        self.frame(1)
        self.begin()
        source=Path(self.events('verification_source')[0]['source_file'])
        original=source.read_bytes()
        for n in range(2,52):
            self.frame(n,ident=n)
        self.assertEqual(len(self.view.frames),48)
        self.assertNotIn(1,self.view.frames)
        self.assertEqual(source.read_bytes(),original)


if __name__ == '__main__':
    unittest.main()
