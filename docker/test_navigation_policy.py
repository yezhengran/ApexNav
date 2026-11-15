import importlib.util, pathlib, unittest
ROOT=pathlib.Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('run_batch_navigation',ROOT/'docker/run_batch.py')
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)
class NavigationPolicy(unittest.TestCase):
 def test_legacy_invocation_identical(self):
  a=M.evaluation_command('hm3dv2',1,'out',False)
  b=M.evaluation_command('hm3dv2',1,'out',False,navigation_profile='legacy')
  self.assertEqual(a,b);self.assertFalse(any('navigation_profile' in s for s in a))
 def test_opt_in_preserves_review(self):
  base=dict(method='selected_review',step_sync=True,review_profile='linked_framing',review_backend='qwen',uncertain_after_steps=400)
  a=M.evaluation_command('hm3dv2',1,'out',True,**base)
  b=M.evaluation_command('hm3dv2',1,'out',True,**base,navigation_profile='stable')
  self.assertEqual([s for s in b if not s.startswith('+navigation_profile=')],a)
  self.assertIn('+navigation_profile=stable',b)
  self.assertIn('+active_view.qwen_plant_confirm_threshold=0.6',b)
class RegionalPolicy(unittest.TestCase):
 def test_legacy_invocation_identical(self):
  a=M.evaluation_command('hm3dv2',1,'out',False)
  b=M.evaluation_command('hm3dv2',1,'out',False,navigation_profile='legacy')
  self.assertEqual(a,b);self.assertFalse(any('navigation_profile' in s for s in a))
 def test_opt_in_preserves_review(self):
  base=dict(method='selected_review',step_sync=True,review_profile='linked_framing',review_backend='qwen',uncertain_after_steps=400)
  a=M.evaluation_command('hm3dv2',1,'out',True,**base)
  b=M.evaluation_command('hm3dv2',1,'out',True,**base,navigation_profile='regional')
  self.assertEqual([s for s in b if not s.startswith('+navigation_profile=')],a)
  self.assertIn('+navigation_profile=regional',b)
  self.assertIn('+active_view.qwen_plant_confirm_threshold=0.6',b)
class AdaptivePolicy(unittest.TestCase):
 def test_legacy_invocation_identical(self):
  a=M.evaluation_command('hm3dv2',1,'out',False)
  b=M.evaluation_command('hm3dv2',1,'out',False,navigation_profile='legacy')
  self.assertEqual(a,b);self.assertFalse(any('navigation_profile' in s for s in a))
 def test_opt_in_preserves_review(self):
  base=dict(method='selected_review',step_sync=True,review_profile='linked_framing',review_backend='qwen',uncertain_after_steps=400)
  a=M.evaluation_command('hm3dv2',1,'out',True,**base)
  b=M.evaluation_command('hm3dv2',1,'out',True,**base,navigation_profile='adaptive')
  self.assertEqual([s for s in b if not s.startswith(('+navigation_profile=', '+active_view.adaptive_framing='))],a)
  self.assertIn('+navigation_profile=adaptive',b)
  self.assertIn('+active_view.adaptive_framing=true',b)
  self.assertIn('+active_view.qwen_plant_confirm_threshold=0.6',b)
if __name__=='__main__':unittest.main()
