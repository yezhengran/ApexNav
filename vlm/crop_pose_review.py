"""One fallback arc at the radius of the exact best crop's capture position.

The original full-framing ring is unchanged. Only an unavailable initial route
can fall back; this is not a radius-shrinking loop or another inspection budget.
"""
import math
import shutil

import numpy as np

from .choice_review import ChoiceTargetReviewer
from .confidence_review import ConfidenceTargetReviewer
from .fused_review import FusedTargetReviewer


def crop_pose_band(center, source_pose):
    if len(center) != 2 or len(source_pose) != 3 or not np.isfinite([*center, *source_pose]).all():
        raise ValueError('Invalid crop-pose geometry')
    radius = math.dist(center, source_pose[:2])
    # Keep the established observation search range. Never silently clamp the
    # requested radius: it must equal the recorded capture distance exactly.
    return dict(radius=radius, minimum_radius=max(.6, radius-.25),
                max_radius=min(4.5, radius+.25), feasible=.6 <= radius <= 4.5,
                reference_angle=math.atan2(source_pose[1]-center[1], source_pose[0]-center[0]))


class CropPoseTargetReviewer(ChoiceTargetReviewer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not getattr(self.cfg, 'crop_pose_fallback', False):
            raise ValueError('Crop-pose fallback requires its explicit switch')

    def begin(self, data):
        # Fused.begin creates the exact same first review / inspection state.
        # Handle a geometrically infeasible primary band before Framing.begin
        # would close that state, so the single fallback remains possible.
        result = FusedTargetReviewer.begin(self, data)
        if self.active is not None and not self.active['feasible']:
            return self._fallback('primary_geometry_unavailable')
        return result

    def _fallback(self, reason):
        target = self.active
        if target is None:
            raise RuntimeError('Fallback requires an active inspection')
        if target.get('crop_pose_mode') or self.step != target['started'] or self.step >= 400:
            return self._finish('no_safe_viewpoint')
        chosen = self._load(min(self.frames[target['id']], key=self._rank))
        source = self.directory / (f'fallback-{target["id"]:03d}-'
                                  f's{chosen["step"]:04d}-m{chosen["key"][1]:02d}.npz')
        shutil.copyfile(chosen['source_file'], source)
        band = crop_pose_band(target['center'], chosen['pose'])
        self.event('inspection_crop_pose_fallback', candidate=target['id'], reason=reason,
                   center=target['center'].tolist(), observation=chosen['step'],
                   mask_index=chosen['key'][1], source_file=str(source),
                   detector_score=chosen['score'], pose=chosen['pose'], bounds=chosen['bounds'],
                   selection='best_object_history_at_inspection', half_arc_degrees=30,
                   half_width=.25, original_radius=target['radius'], **band)
        if not band['feasible']:
            return self._finish('crop_pose_radius_out_of_range')
        # Keep started/last_motion/call count and the original object center.
        # A new ring is a continuation of this inspection, not a fresh attempt.
        target.update(band, crop_pose_mode=True, crop_pose_source=chosen['pose'][:2])
        return None

    def request(self):
        value = super().request()
        if value and self.active.get('crop_pose_mode'):
            # Eleven-field v2/v3 contract; v3 explicitly permits one point.
            # The planner verifies the reference
            # pose implies the exact radius and bounded band before accepting.
            value += list(self.active['crop_pose_source']) + [3. if getattr(self.cfg, 'crop_pose_single_view', False) else 2.]
        return value

    def finish_inspection(self, reachable=True, initial_unavailable=False):
        target = self.active
        if not reachable and initial_unavailable and target is not None and not target.get('crop_pose_mode'):
            return self._fallback('primary_route_unavailable')
        if reachable and target is not None and target.get('crop_pose_mode'):
            distance = math.dist(target['center'], self.pose[:2])
            if not target['minimum_radius']-1e-5 <= distance <= target['max_radius']+1e-5:
                raise RuntimeError('Arrival outside crop-pose band')
            self.event('inspection_crop_pose_arrival', candidate=target['id'], distance=distance,
                       minimum_radius=target['minimum_radius'], max_radius=target['max_radius'])
            # This mode intentionally uses the observed crop's radius, not the
            # full-frame distance guarantee. Facing/current association
            # and the two-call limit are still checked by Confidence/Annular.
            return ConfidenceTargetReviewer.finish_inspection(self, True,
                allow_current=getattr(self.cfg, 'crop_pose_single_view', False))
        return super().finish_inspection(reachable)
