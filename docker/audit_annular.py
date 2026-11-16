"""Offline Q8 evidence audit. No model calls or navigation changes."""
import collections
import hashlib
import json
import math
import re
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image

from vlm.active_view import crop_target
from vlm.annular_review import effective_decision
from vlm.framing_review import object_envelope, framing_distance
from vlm.crop_pose_review import crop_pose_band
from vlm.qwen_review import CONFIDENCE_PROMPT, MULTIVIEW_PROMPT, CHOICE_PROMPT


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def ranked(images):
    # One source identity per map object, with the same documented tie breaks.
    unique = {(e['observation'], e['mask_index']): e for e in images if e['eligible']}
    return sorted(unique.values(), key=lambda e: (-e['detector_score'], not e['complete'],
                  -e['pixels'], -e['observation'], e['mask_index']))[:5]


def validate_pool(history, event):
    pool = ranked(history)
    assert event['retained_keys'] == [[e['observation'], e['mask_index']] for e in pool]
    assert event['retained_scores'] == [e['detector_score'] for e in pool]
    assert event['selection'] == 'strict_detector_top5'
    assert event['retained'] == any(e is event for e in pool)


def validate_inputs(event, sources, history, ordinal):
    assert ordinal in (1, 2)
    pool = ranked(history)
    expected = pool[:1] if ordinal == 1 else pool
    assert 1 <= len(expected) <= 5
    assert len(sources) == event['image_count'] == len(event['crops']) == len(expected)
    assert event['crop'] == event['crops'][0]
    assert event['phase'] == ('fused_first' if ordinal == 1 else 'center_facing')
    assert len(event['sources']) == len(sources)
    for index, (source, best, metadata) in enumerate(zip(sources, expected, event['sources'])):
        assert source['candidate'] == event['candidate'] and source['step'] == event['step']
        assert source['image_index'] == index and source['phase'] == event['phase']
        assert source['selection'] == ('best_object_history' if ordinal == 1 else 'strict_detector_top5')
        assert (source['observation'], source['mask_index']) == (best['observation'], best['mask_index'])
        assert source['detector_score'] == best['detector_score']
        assert source['crop'] == event['crops'][index] and source['detection_label'] == 0
        assert all(source[k] == value for k, value in metadata.items())
    assert event['evidence_step'] == sources[0]['observation']
    assert event['evidence_age'] == event['step'] - event['evidence_step'] >= 0


def validate_call(event, ledger, target, ordinal):
    reply = event['model_reply']; bill = ledger['calls'][reply['call_id']]
    assert bill['state'] == reply['state'] == 'completed'
    assert bill['answer'] == reply['answer']
    assert bill['target'] == target
    answer = reply['answer']; value = answer['target_confidence']
    assert type(value) in (int, float) and math.isfinite(value) and 0 <= value <= 1
    assert answer['decision'] in ('confirmed', 'uncertain', 'rejected')
    assert answer['quality'] in ('adequate', 'insufficient')
    decision = effective_decision(reply, target)
    assert event['effective_decision'] == decision
    assert event['decision_policy'] == 'adequate_top5_plant060_v1'
    assert event['decision'] == {'confirmed': 1, 'uncertain': 2, 'rejected': 3}[decision]
    hashes = [sha(p) for p in event['crops']]
    assert bill['image_sha256'] == reply['image_sha256'] == hashes[0]
    template = CONFIDENCE_PROMPT if ordinal == 1 else MULTIVIEW_PROMPT
    prompt = template.format(target=target, count=len(hashes))
    assert bill['prompt_sha256'] == reply['prompt_sha256'] == hashlib.sha256(prompt.encode()).hexdigest()
    image_fields = ('image_sha256',)
    if ordinal == 2:
        image_fields += ('image_count', 'image_sha256s', 'image_selection')
        for record in (bill, reply):
            assert record['image_count'] == len(hashes)
            assert record['image_sha256s'] == hashes
            assert record['image_selection'] == 'candidate_top5_joint_v1'
    else:
        assert 'image_sha256s' not in bill and len(hashes) == 1
    ids = reply['transport_attempt_ids']
    assert 1 <= len(ids) <= 3 and len(set(ids)) == len(ids) and ids[-1] == reply['call_id']
    assert reply['transport_attempt'] == len(ids)
    for index, call_id in enumerate(ids, 1):
        entry = ledger['calls'][call_id]
        assert entry['request_id'] == reply['request_id']
        assert entry['transport_attempt'] == index and entry['transport_attempt_ids'] == ids[:index]
        for field in (*image_fields, 'target', 'prompt_sha256'):
            assert entry[field] == bill[field]
        if index < len(ids):
            assert entry['state'] == 'error' and entry['charged_cny'] == .02
            assert entry['error_stage'] == 'transport' and entry['transport']['retryable']
            assert entry['retry_scheduled'] and entry['retry_delay_seconds'] == (2, 5)[index-1]
    assert bill['charged_cny'] == reply['charged_cny']
    assert abs(bill['charged_cny'] - (bill['usage']['prompt_tokens'] + 10*bill['usage']['completion_tokens'])/1e6) < 1e-12
    return ids


def validate_source(attempt, source, receipt, trace, sensor, crop=True, comparison=False, fallback=False):
    ident, obs, mask_index = source['candidate'], source['observation'], source['mask_index']
    path = Path(source['source_file'])
    assert path.resolve().parent == attempt.resolve()
    if fallback:
        assert not crop and not comparison
        assert path == attempt / f'fallback-{ident:03d}-s{obs:04d}-m{mask_index:02d}.npz'
    elif comparison:
        assert path == attempt / (f'comparison-{source["request_id"]:03d}-'
                    f'{source["round_index"]:02d}-{source["offset"]:03d}-{ident:03d}.npz')
        assert Path(source['crop']) == path.with_suffix('.png')
    elif crop:
        assert path == attempt / (f'candidate-{ident:03d}-{source["phase"]}-'
                     f'{0 if source["phase"] == "fused_first" else 1:02d}-{source["image_index"]:02d}.npz')
        assert Path(source['crop']) == path.with_suffix('.png')
    else:
        assert path == attempt / f'geometry-{ident:03d}-s{obs:04d}-m{mask_index:02d}.npz'
    assert receipt['reason'] == 'assigned' and receipt['usable']
    with np.load(path, allow_pickle=False) as data:
        assert tuple(int(data[k]) for k in ('candidate', 'observation', 'mask_index')) == (ident, obs, mask_index)
        np.testing.assert_allclose(data['bounds'], receipt['bounds'], rtol=0, atol=1e-8)
        pose = trace[obs]['pose']
        np.testing.assert_allclose(data['pose'], [-pose[2], -pose[0], pose[3]], rtol=0, atol=1e-7)
        assert abs(float(data['pitch']) - pose[4]) < 1e-7
        assert abs(float(data['camera_height']) - pose[1] - sensor.position[1]) < 1e-7
        if fallback:
            np.testing.assert_allclose(data['bounds'], source['bounds'], rtol=0, atol=1e-8)
            np.testing.assert_allclose(data['pose'], source['pose'], rtol=0, atol=1e-7)
        if crop:
            np.testing.assert_allclose(data['bounds'], source['bounds'], rtol=0, atol=1e-8)
            np.testing.assert_allclose(data['pose'], source['pose'], rtol=0, atol=1e-7)
            shape = tuple(data['mask_shape'])
            mask = np.unpackbits(data['mask_bits'])[:math.prod(shape)].reshape(shape).astype(bool)
            rebuilt, pixels, complete = crop_target(data['rgb'], mask)
            assert pixels == source['mask_pixels'] and complete == source['complete']
            image = Image.fromarray(rebuilt); image.thumbnail((256, 256))
            with Image.open(source['crop']) as actual:
                assert np.array_equal(np.asarray(image), np.asarray(actual)), 'Source/crop pixel mismatch'
        return object_envelope(data, sensor), tuple(data['mask_shape'])


def video_audit(run, row):
    videos = list(run.glob('*/*.mp4'))
    if int(row['success']):
        assert not row['video_path'] and not videos
        return None
    path = run / row['video_path']
    assert path.is_file() and videos == [path]
    cap = cv2.VideoCapture(str(path)); fps = cap.get(cv2.CAP_PROP_FPS); count = 0
    while cap.read()[0]: count += 1
    cap.release()
    assert count == int(row['steps']) + 1
    return dict(path=str(path), frames=count, fps=fps, sha256=sha(path))


def validate_comparison_http(event, ledger, target):
    reply=event['model_reply']; ids=event['candidate_ids']; bill=ledger['calls'][reply['call_id']]
    assert 2 <= len(ids) <= 5 and len(set(ids)) == len(ids)
    assert event['outcome'] == reply['answer']['outcome'] == 'selected'
    assert event['selected_candidate'] == reply['answer']['candidate_id'] in ids
    assert type(reply['answer']['candidate_id']) is int
    hashes=[sha(p) for p in event['crops']]
    prompt=CHOICE_PROMPT.format(target=target,ids=ids)
    expected=dict(image_count=len(ids), image_sha256s=hashes, image_sha256=hashes[0],
                  candidate_ids=ids, image_selection='candidate_best_comparison_v1',
                  target=target, prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())
    attempts=reply['transport_attempt_ids']
    assert 1 <= len(attempts) <= 3 and len(set(attempts))==len(attempts) and attempts[-1]==reply['call_id']
    for n,call in enumerate(attempts,1):
        entry=ledger['calls'][call]
        assert all(entry[k]==v for k,v in expected.items())
        assert entry['request_id']==reply['request_id'] and entry['transport_attempt']==n
        assert entry['transport_attempt_ids']==attempts[:n]
        if n<len(attempts):
            assert entry['state']=='error' and entry['error_stage']=='transport'
            assert entry['transport']['retryable'] and entry['charged_cny']==.02
            assert entry['retry_scheduled'] and entry['retry_delay_seconds']==(2,5)[n-1]
    assert all(reply[k]==v for k,v in expected.items())
    assert bill['state']==reply['state']=='completed' and bill['http_status']==200
    assert bill['answer']==reply['answer'] and bill['charged_cny']==reply['charged_cny']
    assert abs(bill['charged_cny']-(bill['usage']['prompt_tokens']+10*bill['usage']['completion_tokens'])/1e6)<1e-12
    return attempts


def audit_evidence(run, attempt, events, row, cfg, trace, ledger):
    assert events and events[-1]['event'] == 'episode_end'
    assert sum(e['event'] == 'episode_end' for e in events) == 1
    assert not any(e['event'] in ('association_missing', 'stop_review_started') for e in events)
    sensor = SimpleNamespace(**cfg['habitat']['simulator']['agents']['main_agent']['sim_sensors']['depth_sensor'])
    choice = bool(cfg.get('active_view',{}).get('choice_review',False))
    crop_pose = bool(cfg.get('active_view',{}).get('crop_pose_fallback',False))
    assert not crop_pose or choice
    fallbacks = {}
    comparisons=[]; comparison_requests={}; comparison_groups={}; comparison_finished={}; comparison_keys=set()
    comparison_sources=None; decisions={}; trigger_checks={}
    receipts = {}; histories = collections.defaultdict(list); pending = []
    calls = []; counts = collections.Counter(); first = {}; requested = {}; facing = {}; arrivals = {}
    geo_sources = collections.defaultdict(list); geometry = {}; actions = []; used = collections.defaultdict(set)
    for e in events:
        ident = e.get('candidate')
        if e['event'] == 'source_association':
            key = (ident, e['observation'], e['mask_index']); assert key not in receipts
            receipts[key] = e
        elif e['event'] == 'candidate_image':
            assert e['label'] == 0 or not e['eligible']
            assert e['eligible'] == (e['label'] == 0 and e['pixels'] >= 64 and e['short_side'] >= 8)
            assert math.isfinite(e['detector_score']) and 0 <= e['detector_score'] <= 1
            histories[ident].append(e); validate_pool(histories[ident], e)
        elif e['event'] == 'choice_trigger_check':
            assert choice
            pool=ranked(histories[ident]);crop=bool(pool and pool[0]['detector_score']>.7)
            assert e['crop_threshold']==.7 and e['crop_passed']==crop
            assert e['allowed']==(e['fused_confidence']>=e['threshold'] or crop)
            trigger_checks[ident]=e
        elif e['event'] == 'fused_review_started':
            assert ident not in first
            if choice:
                check=trigger_checks[ident]
                assert check['allowed'] and check['step']==e['step']
                assert check['fused_confidence']==e['fused_confidence'] and check['threshold']==e['threshold']
            else:
                assert e['fused_confidence'] >= e['threshold']
            first[ident] = e
        elif e['event'] == 'annular_geometry_source':
            envelope, shape = validate_source(attempt, e, receipts[ident, e['observation'], e['mask_index']], trace, sensor, False)
            assert envelope == e['envelope']; geo_sources[ident].append((e, shape))
        elif e['event'] == 'inspection_geometry':
            sources = geo_sources[ident]; pool = ranked(histories[ident])
            assert len(sources) == e['source_count'] == len(pool)
            assert [(x['observation'], x['mask_index']) for x, _ in sources] == [(x['observation'], x['mask_index']) for x in pool]
            envelopes = [x['envelope'] for x, _ in sources if x['envelope']]
            envelope = None
            if envelopes:
                envelope = dict(low=np.quantile([x['low'] for x in envelopes], .2, axis=0).tolist(),
                                high=np.quantile([x['high'] for x in envelopes], .8, axis=0).tolist(),
                                border_clipped=any(x['border_clipped'] for x in envelopes))
            assert e['envelope'] == envelope
            pose = trace[e['step']]['pose']
            expected = framing_distance(e['extent'], envelope, pose[1]+sensor.position[1], pose[4], sensor.hfov, sources[-1][1])
            for k, v in expected.items():
                assert math.isclose(e[k], v, abs_tol=1e-8) if type(v) is float else e[k] == v
            geometry[ident] = e
        elif e['event'] == 'inspection_requested':
            assert ident in first and ident not in requested and ident in geometry
            assert counts[ident] == 1 and next(x for x in reversed(calls) if x['candidate'] == ident)['decision'] == 2
            assert e['reason'] == 'no_reachable_confirmed_alternative'
            requested[ident] = e
        elif e['event'] == 'inspection_crop_pose_fallback':
            assert crop_pose and ident in requested and ident not in fallbacks
            assert e['step'] == requested[ident]['step'] < 400
            assert not any(x['candidate'] == ident for x in actions)
            assert e['reason'] in ('primary_geometry_unavailable', 'primary_route_unavailable')
            assert geometry[ident]['feasible'] == (e['reason'] == 'primary_route_unavailable')
            best = ranked(histories[ident])[0]
            for k in ('observation','mask_index','detector_score','pose'):
                assert e[k] == best[k]
            assert e['selection'] == 'best_object_history_at_inspection'
            assert e['center'] == requested[ident]['center']
            assert e['original_radius'] == geometry[ident]['radius']
            assert e['half_arc_degrees'] == 30 and e['half_width'] == .25
            validate_source(attempt,e,receipts[ident,e['observation'],e['mask_index']],trace,sensor,
                            crop=False,fallback=True)
            expected = crop_pose_band(e['center'],e['pose'])
            for k,v in expected.items():
                assert math.isclose(e[k],v,abs_tol=1e-8) if type(v) is float else e[k] == v
            fallbacks[ident] = e
            if e['feasible']:
                geometry[ident] = e
        elif e['event'] == 'inspection_evidence':
            assert ident in requested and e['action']==trace[e['step']]['action']
            if e['action']==0:
                assert e['step']==int(row['steps'])==500 and trace[-1]['action']==0
                assert trace[-1]['pose']==trace[-2]['pose']
                assert any(x['event']=='review_finished' and x.get('reason')=='episode_ended_during_inspection' for x in events)
                assert not any(x['event']=='verification' and x['step']>=e['step'] for x in events)
            else:assert e['action'] in (1,2,3)
            actions.append(e)
        elif e['event'] in ('inspection_framing_arrival','inspection_crop_pose_arrival'):
            assert (e['event'] == 'inspection_crop_pose_arrival') == (ident in fallbacks)
            if ident in fallbacks:
                assert fallbacks[ident]['feasible']
                assert e['minimum_radius'] == geometry[ident]['minimum_radius']
                assert e['max_radius'] == geometry[ident]['max_radius']
            pose = trace[e['step']]['pose']; request = requested[ident]
            distance = math.dist([-pose[2], -pose[0]], request['center'])
            assert abs(distance-e['distance']) < 1e-7
            assert distance <= geometry[ident]['max_radius'] + 1e-5
            arrivals[ident] = e
        elif e['event'] == 'inspection_facing_ready':
            pose = trace[e['step']]['pose']; robot = [-pose[2], -pose[0], pose[3]]
            np.testing.assert_allclose(e['pose'], robot, rtol=0, atol=1e-7)
            yaw = math.atan2(e['center'][1]-robot[1], e['center'][0]-robot[0])
            error = abs(math.atan2(math.sin(yaw-robot[2]), math.cos(yaw-robot[2])))
            assert error <= math.radians(18)+1e-6 and abs(math.degrees(error)-e['yaw_error_degrees']) < 1e-7
            arrival = arrivals[ident]; assert arrival['step'] == e['step']
            assert arrival['distance']+1e-5 >= arrival['minimum_radius']
            assert requested[ident]['step'] < e['step'] < 400
            facing[ident] = e
        elif e['event'] == 'verification_source':
            pending.append(e)
        elif e['event'] == 'verification':
            assert ident in first and e['backend'] == 'qwen' and e['score'] is None
            decisions[ident]=e['decision']
            counts[ident] += 1; ordinal = counts[ident]
            validate_inputs(e, pending, histories[ident], ordinal)
            if ordinal == 2:
                assert ident in facing and facing[ident]['step'] == e['step']
            assert e['new_view_images'] == sum(x['observation'] > first[ident]['step'] for x in pending)
            for source in pending:
                key = (source['observation'], source['mask_index'])
                assert source['reused'] == (key in used[ident])
                validate_source(attempt, source, receipts[ident, *key], trace, sensor)
                used[ident].add(key)
            ids = validate_call(e, ledger, row['target'], ordinal)
            calls.append(dict(candidate=ident, step=e['step'], phase=e['phase'], decision=e['decision'],
                              effective_decision=e['effective_decision'], reply=e['model_reply'],
                              http_attempts=len(ids), image_count=e['image_count'], new_view_images=e['new_view_images']))
            pending = []
        elif e['event'] == 'comparison_requested':
            assert choice and e['request_id']==len(comparison_requests)+1
            ids=e['candidate_ids'];assert len(ids)>=2 and ids==sorted(set(ids))
            assert all(ranked(histories[i]) and decisions.get(i)!=3 for i in ids)
            if e['step']<400:assert all(decisions.get(i)==1 for i in ids)
            else:assert not any(v==1 for v in decisions.values())
            key=(e['step']>=400,tuple((i,counts[i]) for i in ids))
            assert key not in comparison_keys;comparison_keys.add(key)
            comparison_requests[e['request_id']]=e;comparison_groups[e['request_id']]=[]
        elif e['event']=='comparison_sources':
            assert choice and comparison_sources is None
            comparison_sources=e
        elif e['event']=='candidate_comparison':
            assert choice and comparison_sources is not None
            source_event=comparison_sources;comparison_sources=None
            assert all(source_event[k]==e[k] for k in ('request_id','round_index','candidate_ids','sources','crops','step'))
            request=comparison_requests[e['request_id']];assert request['step']==e['step']
            assert len(e['sources'])==len(e['candidate_ids'])==len(e['crops'])
            for i,source in enumerate(e['sources']):
                c=source['candidate'];assert c==e['candidate_ids'][i] and source['crop']==e['crops'][i]
                best=ranked(histories[c])[0]
                assert (source['observation'],source['mask_index'],source['detector_score'])==(best['observation'],best['mask_index'],best['detector_score'])
                assert source['detection_label']==0 and decisions.get(c)!=3
                # Canonical path embeds the group offset; it is checked again in the tournament below.
                match=re.fullmatch(r'comparison-(\d+)-(\d+)-(\d+)-(\d+)\.npz',Path(source['source_file']).name)
                assert match and int(match[1])==e['request_id'] and int(match[2])==e['round_index'] and int(match[4])==c
                metadata={**source,'request_id':e['request_id'],'round_index':e['round_index'],'offset':int(match[3])}
                validate_source(attempt,metadata,receipts[c,source['observation'],source['mask_index']],trace,sensor,comparison=True)
            attempts=validate_comparison_http(e,ledger,row['target'])
            comparisons.append({**e,'http_attempts':len(attempts)})
            comparison_groups[e['request_id']].append(e)
        elif e['event']=='comparison_finished':
            request=comparison_requests[e['request_id']]
            assert e['request_id'] not in comparison_finished and e['candidate_ids']==request['candidate_ids'] and e['step']==request['step']
            remaining=list(request['candidate_ids']); groups=iter(comparison_groups[e['request_id']]);round_index=0
            while len(remaining)>1:
                winners=[]
                for offset in range(0,len(remaining),5):
                    group=remaining[offset:offset+5]
                    if len(group)==1:winners+=group;continue
                    call=next(groups)
                    assert call['candidate_ids']==group and call['round_index']==round_index
                    assert all(Path(s['source_file']).name==f'comparison-{e["request_id"]:03d}-{round_index:02d}-{offset:03d}-{s["candidate"]:03d}.npz' for s in call['sources'])
                    winners.append(call['selected_candidate'])
                remaining=winners;round_index+=1
            assert next(groups,None) is None and e['selected_candidate']==remaining[0] and e['outcome']=='selected'
            comparison_finished[e['request_id']]=e
    assert not pending and comparison_sources is None
    assert set(comparison_requests)==set(comparison_finished)
    assert sum(counts.values()) + len(comparisons) == events[-1]['vlm_calls']
    assert len(actions) == events[-1]['inspection_actions']
    assert all(n <= 2 for n in counts.values())
    assert all(sum(e['candidate'] == ident for e in actions) <= 48 for ident in requested)
    planner = (run/'planner.log').read_text(); snapshots = []
    pattern = r'\[Candidate scores\] step=(\d+) id=(\d+) fusion=([\d.eE+-]+) state=(\d+) has_qwen=(\d) qwen=([\d.eE+-]+) rank=([\d.eE+-]+) late=(\d)'
    for match in re.finditer(pattern, planner):
        step, ident, fusion, state, has, qwen, rank, late = match.groups()
        step, ident, state, has, late = map(int, (step, ident, state, has, late))
        fusion, qwen, rank = map(float, (fusion, qwen, rank))
        assert abs(rank - (qwen if late and has else fusion)) < 2e-8
        assert late == int(step >= 400)
        snapshots.append((step, ident, has, qwen))
    if not snapshots:
        assert int(row['steps']) == 0 and int(row['success']) == 0 and row['result'] == 'infeasible'
        assert len(events) == 1 and events[0]['step'] == 0
        assert len(trace) == 1 and trace[0]['step'] == 0 and trace[0]['action'] is None
        assert re.findall(r'\[Observation applied\] step=(\d+)[^\n]*masks=(\d+)', planner) == [('0', '0')]
        assert 'Init Mode Process -----> (0/26)' in planner
        for k in ('selected_targets', 'vlm_calls', 'vlm_seconds', 'confirmed', 'uncertain', 'rejected', 'inspections', 'inspection_actions', 'pending_evidence'):
            assert events[0][k] == 0
    for call in calls:
        assert any(step == call['step'] and ident == call['candidate'] and has and
                   abs(qwen-call['reply']['answer']['target_confidence']) < 1e-8 for step, ident, has, qwen in snapshots)
    samples = collections.defaultdict(list)
    for ident,e in fallbacks.items():
        if e['reason'] == 'primary_route_unavailable':
            p = (rf'\[Annular search\] id={ident} mode=framing step={e["step"]} '
                 r'radius=[\d.]+ generated=\d+ safe=\d+ visible=\d+ access=\d+ arcs=\d+ accepted=0')
            assert re.search(p,planner), 'Missing failed primary planner search'
    pattern = r'\[Annular sample\] id=(\d+) sample=(\d+)/(\d+) step=(\d+) position=\(([-\d.]+),([-\d.]+)\) yaw=([-\d.]+)'
    for match in re.finditer(pattern, planner):
        ident, number, total, step = map(int, match.groups()[:4])
        x, y, yaw = map(float, match.groups()[4:]); pose = trace[step]['pose']
        assert ident in requested and 2 <= total <= 3 and 1 <= number <= total
        np.testing.assert_allclose([x, y, yaw], [-pose[2], -pose[0], pose[3]], atol=6e-5, rtol=0)
        distance = math.dist([x, y], requested[ident]['center'])
        assert geometry[ident]['minimum_radius']-1e-4 <= distance <= geometry[ident]['max_radius']+1e-4
        if ident in fallbacks:
            # A 0.20m arrival disc can offset the realized angular position.
            a=math.atan2(y-requested[ident]['center'][1],x-requested[ident]['center'][0])
            delta=abs(math.atan2(math.sin(a-fallbacks[ident]['reference_angle']),math.cos(a-fallbacks[ident]['reference_angle'])))
            assert delta <= math.radians(30)+math.asin(min(1.,.2001/distance))+1e-4
        desired = math.atan2(requested[ident]['center'][1]-y, requested[ident]['center'][0]-x)
        assert abs(math.atan2(math.sin(desired-yaw), math.cos(desired-yaw))) <= math.radians(18)+1e-4
        samples[ident].append((number, total, step))
    for call in calls:
        if call['phase'] == 'center_facing':
            group = samples[call['candidate']]
            assert group and [x[0] for x in group] == list(range(1, group[-1][1]+1))
            assert len({x[1] for x in group}) == 1 and group[-1][2] == call['step']
    assert all(f'[Conditional inspection trigger] id={ident} ' in planner for ident in requested)
    return dict(crop_pose_fallbacks=sorted(fallbacks), comparisons=comparisons, calls=calls, inspection_candidates=sorted(requested), inspection_actions=len(actions),
                inspection_motion_types=dict(collections.Counter(e['action'] for e in actions)),
                review={k: events[-1].get(k, 0) for k in ('selected_targets', 'vlm_calls', 'vlm_seconds', 'confirmed', 'uncertain', 'rejected', 'pending_evidence', 'inspection_actions')},
                events_sha256=sha(attempt/'events.jsonl'), video=video_audit(run, row))
