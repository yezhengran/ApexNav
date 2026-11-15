"""Opt-in fixed-pitch observation distance from linked RGB-D object geometry.

The map still owns candidate identity/XY center. Geometry uses only that ID's
best saved detection, never semantic ground truth or the navigation goal.
"""
import math
import shutil

import numpy as np

from .active_view import observation_radius
from .confidence_review import ConfidenceTargetReviewer

ARRIVAL_MARGIN = .3125
MAX_RADIUS = 4.5
IMAGE_FRACTION = .8
YAW_TOLERANCE = math.radians(18)


def object_envelope(packet, sensor):
    """5–95% visible 3D bounds, rejecting depth outside this mapper cluster."""
    shape = tuple(packet['mask_shape'])
    mask = np.unpackbits(packet['mask_bits'])[:math.prod(shape)].reshape(shape).astype(bool)
    depth = np.asarray(packet['depth'], dtype=float).squeeze()
    if depth.shape != shape:
        raise ValueError('Depth/mask shape mismatch')
    if sensor.normalize_depth:
        depth = depth * (sensor.max_depth - sensor.min_depth) + sensor.min_depth
    v, u = np.where(mask & np.isfinite(depth) & (depth > max(.05, sensor.min_depth)) &
                    (depth < sensor.max_depth * .95))
    if len(u) < 64:
        return None
    # Deterministic and bounded; never consume the simulator RNG.
    stride = max(1, math.ceil(len(u) / 4096))
    u, v = u[::stride], v[::stride]
    h, w = shape
    focal = w / (2 * math.tan(math.radians(sensor.hfov) / 2))
    z = depth[v, u]
    down, left = (v-h/2)*z/focal, -(u-w/2)*z/focal
    pitch = float(packet['pitch'])  # Existing convention: positive looks up.
    forward = z*math.cos(pitch) + down*math.sin(pitch)
    up = z*math.sin(pitch) - down*math.cos(pitch)
    px, py, yaw = packet['pose']
    points = np.column_stack((px + forward*math.cos(yaw)-left*math.sin(yaw),
                              py + forward*math.sin(yaw)+left*math.cos(yaw),
                              float(packet['camera_height']) + up))
    box = np.asarray(packet['bounds'])
    keep = np.all((points[:, :2] >= box[:2]-.20) & (points[:, :2] <= box[2:]+.20), axis=1)
    points = points[keep]
    if len(points) < 64:
        return None
    low, high = np.percentile(points, [5, 95], axis=0)
    # Observed, robust envelope only: unseen/occluded parts are not known.
    return {'low': low.tolist(), 'high': high.tolist(), 'centroid': np.median(points, axis=0).tolist(),
            'points': len(points), 'image_shape': [int(h), int(w)],
            'border_clipped': bool(mask[:3].any() or mask[-3:].any() or mask[:, :3].any() or mask[:, -3:].any())}


def framing_distance(extent, envelope, camera_height, pitch, hfov, shape):
    """Conservative center distance with frame margins and arrival tolerance.

    Enclose the XY footprint by a disk, fit vertical extremes at its nearest
    surface, and allow the existing 18-degree yaw quantization. No pitch motion.
    """
    if not all(math.isfinite(float(x)) for x in (extent, camera_height, pitch, hfov)) or not 0 < hfov < 180:
        raise ValueError('Invalid camera/extent geometry')
    half_h = math.atan(IMAGE_FRACTION * math.tan(math.radians(hfov)/2))
    half_v = math.atan(IMAGE_FRACTION * shape[0]/shape[1] * math.tan(math.radians(hfov)/2))
    usable_h = half_h - YAW_TOLERANCE
    if usable_h <= 0 or pitch-half_v >= 0 or pitch+half_v <= 0:
        return {'feasible': False, 'reason': 'fixed_pitch_or_fov_cannot_frame', 'radius': MAX_RADIUS,
                'minimum_radius': MAX_RADIUS, 'max_radius': MAX_RADIUS}
    half_width = max(.15, float(extent))/2
    # Map extent is max(dx,dy); half diagonal conservatively encloses it.
    footprint = math.sqrt(2)*half_width
    vertical = 0.
    if envelope is not None:
        low, high = np.asarray(envelope['low']), np.asarray(envelope['high'])
        if not np.isfinite([*low, *high]).all() or np.any(low > high):
            raise ValueError('Invalid object bounds')
        pad = max(.12, .1*(high[2]-low[2]))
        if envelope['border_clipped']:
            pad = max(.25, .2*(high[2]-low[2]))
        vertical = float(max(0., (high[2]+pad-camera_height)/math.tan(pitch+half_v),
                             (camera_height-low[2]+pad)/math.tan(half_v-pitch)))
    # Disk angular radius = asin(footprint / center distance).
    horizontal = footprint / math.sin(usable_h)
    minimum = max(footprint + 1.1, footprint + vertical, horizontal,
                  observation_radius(extent) + .25)
    # A missing valid depth envelope keeps a conservative extra standoff;
    # it does not invent an object height or claim a vertical-fit guarantee.
    if envelope is None:
        minimum = max(minimum, footprint + 1.5)
    radius = minimum + ARRIVAL_MARGIN
    return {'feasible': radius <= MAX_RADIUS, 'reason': 'rgbd_framing' if envelope else 'missing_depth_standoff',
            'radius': radius, 'minimum_radius': minimum, 'max_radius': min(MAX_RADIUS, radius+.70),
            'vertical_surface_distance': vertical, 'horizontal_center_distance': horizontal,
            'footprint_radius': footprint, 'camera_height': camera_height, 'pitch': pitch,
            'vertical_half_fov_degrees': math.degrees(half_v),
            'horizontal_half_fov_degrees': math.degrees(half_h), 'image_fraction': IMAGE_FRACTION}


class FramingTargetReviewer(ConfidenceTargetReviewer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not getattr(self.cfg, 'framing_review', False):
            raise ValueError('Framing requires its independent profile')
        self.camera_height = self.pitch = None

    def observe(self, rgb, observations, masks, scores, labels, pitch, step, action=None):
        self.camera_height = float(observations['gps'][1]) + float(self.sensor.position[1])
        self.pitch = float(pitch)
        result = super().observe(rgb, observations, masks, scores, labels, pitch, step, action)
        self.pending[step]['camera_height'] = self.camera_height
        return result

    def _source_metadata(self, packet):
        return {'camera_height': packet['camera_height']}

    def _inspection_geometry(self, extent):
        chosen = self.frames[self.selected['id']][0]
        source = self.directory / f'geometry-{self.selected["id"]:03d}-s{chosen["step"]:04d}-m{chosen["key"][1]:02d}.npz'
        shutil.copyfile(chosen['source_file'], source)
        with np.load(source, allow_pickle=False) as packet:
            envelope = object_envelope(packet, self.sensor)
            shape = tuple(packet['mask_shape'])
        geometry = framing_distance(extent, envelope, self.camera_height, self.pitch, self.sensor.hfov, shape)
        self.event('inspection_geometry', candidate=self.selected['id'], observation=chosen['step'],
                   mask_index=chosen['key'][1], source_file=str(source), envelope=envelope,
                   original_radius=observation_radius(extent), extent=extent, **geometry)
        return {**geometry, 'envelope': envelope, 'image_shape': shape, 'geometry_extent': extent}

    def begin(self, data):
        result = super().begin(data)
        if self.active is not None and not self.active['feasible']:
            return self._finish('no_framing_viewpoint_within_range')
        return result

    def request(self):
        if self.active is None:
            return []
        return super().request() + [self.active['minimum_radius'], self.active['max_radius']]

    def finish_inspection(self, reachable=True):
        if reachable:
            target = self.active
            needed = framing_distance(target['geometry_extent'], target['envelope'], self.camera_height,
                                      self.pitch, self.sensor.hfov, target['image_shape'])
            distance = float(np.linalg.norm(target['center']-np.array(self.pose[:2])))
            self.event('inspection_framing_arrival', candidate=target['id'], distance=distance,
                       minimum_radius=needed['minimum_radius'], camera_height=self.camera_height, pitch=self.pitch)
            if not needed['feasible'] or distance+1e-5 < needed['minimum_radius']:
                return self._finish('arrival_outside_framing_envelope')
        return super().finish_inspection(reachable)
