"""Bailian visual verification only; never replaces ApexNav's map ITM scores."""

import base64
import errno
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import ssl
import time
import uuid

from PIL import Image
import requests


BASE_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
MODEL = 'qwen3-vl-plus'
MAX_TOKENS = 192
# Conservative reservation for up to five <=256px images and a short prompt. Actual
# token usage is checked against this ceiling; lost responses keep the reserve.
RESERVE_CNY = 0.02
RETRY_DELAYS = (2, 5)
PROMPT = (
    'Inspect the supplied object crop for target category {target}. Use visible '
    'evidence only; do not assume the target is present. Return a JSON object with '
    'exactly these keys: decision (confirmed, rejected, or uncertain), quality '
    '(adequate or insufficient), visible_category (short string), reason (short '
    'string). Confirm only if the target is visually identifiable. Reject only '
    'if the image clearly shows a different object. If small, occluded, truncated '
    'or ambiguous enough to prevent identification, return uncertain. '
    'Do not infer navigation success or location from the image.'
)
CONFIDENCE_PROMPT = PROMPT.replace(
    'exactly these keys:', 'exactly these five keys: target_confidence (number from 0 to 1),') + (
    ' target_confidence estimates how likely the visible object belongs to the requested '
    'target category, not confidence in your chosen decision label. Use the full range; '
    'do not default to 1 for uncertain or rejected. This is an uncalibrated visual '
    'estimate, independent of navigation distance or success.')

MULTIVIEW_PROMPT = (
    'The following {count} crops are observations associated by the robot with ONE '
    'candidate object, ordered by detector confidence. That association can be wrong. '
    'Assess them jointly; do not count repeated or near-identical views as independent '
    'evidence. If the images conflict or do not clearly identify the object, say uncertain. '
    'Return one decision and one target_confidence for the candidate, not one per image. '
) + CONFIDENCE_PROMPT


CHOICE_PROMPT = (
    'These images show DIFFERENT candidate objects for target category {target}. '
    'Image order corresponds exactly to candidate IDs {ids}. Compare visible evidence '
    'and select the candidate most likely to belong to the target category. '
    'Choose exactly ONE best candidate even when evidence is ambiguous: this is a '
    'relative navigation preference, not proof that the target is present. Do not infer distance, reachability '
    'or navigation success. Return JSON with exactly: outcome (selected), '
    'candidate_id (the chosen integer ID), reason (short string). '
    'A relative selection is not a semantic confirmation.'
)


CHOICE_ID_POLICIES = ('global_v1', 'image_order_v1')
IMAGE_ORDER_GUIDANCE = (
    'Candidate IDs in this request are exactly the 1-based image positions: '
    'the first image is candidate_id 1, the second is candidate_id 2, and so on. '
    'Return only one of the listed IDs. No external object IDs are used.'
)

PROMPT_POLICIES = ('legacy', 'semantic_guard_v1')
SEATING_REVIEW_GUIDANCE = (
    'Use conservative category boundaries for chair and couch. A standalone, plush '
    'upholstered single-person seat, armchair, recliner, or single-seat sofa is ambiguous '
    'between these categories for this task: return uncertain rather than confirming it as '
    'either category. Do not broaden couch to include an armchair just because it is '
    'upholstered or has a tufted back, armrests, cushions, or a skirt. Prefer visible '
    'evidence of a continuous seat and backrest extending across multiple seating positions '
    'when confirming couch; do not infer unseen width or count cropped cushions as proof of '
    'seat count. A close crop may hide the rest of a genuine sofa, so a partial sofa-like '
    'shape is uncertain, not automatically rejected or labeled a standalone chair. Ordinary '
    'clearly identifiable chairs outside this upholstered single-seat ambiguity retain the '
    'normal decision rules. A sharp image may still have ambiguous category identity: quality '
    'can be adequate while decision is uncertain. Reflect the unresolved category ambiguity '
    'in target_confidence and briefly state the ambiguity.'
)

TV_REVIEW_GUIDANCE = (
    'Distinguish a physical television or monitor from framed artwork, a picture, a poster, a '
    'mirror, or a window. A rectangular outline, dark border, wall mounting, or a '
    'picture-like surface alone is not enough to confirm TV. Use visible display hardware or '
    'screen-specific evidence, such as a bezel and panel structure, stand, controls, connections, or '
    'screen surface; no single cue is mandatory. A switched-off screen can still be a TV, and '
    'a TV can display art, so neither darkness nor painted-looking content alone decides the '
    'category. If the available crop or views cannot distinguish a physical display from '
    'artwork, return uncertain and reflect that ambiguity in target_confidence. Reject only '
    'when the visible evidence clearly establishes a different object. Do not infer dataset '
    'instance membership or navigation success.'
)

SEATING_CHOICE_GUIDANCE = (
    'Prefer candidates with clear evidence for the requested category over ambiguous '
    'standalone upholstered single-person seats, armchairs, recliners, or single-seat sofas. '
    'For couch, a clearly visible multi-seat sofa is stronger evidence than an armchair-like '
    'object or an isolated cropped seat. For chair, prefer clearly identifiable chairs over '
    'the ambiguous plush single-seat category. Do not select an obviously wrong-category '
    'object merely to avoid an ambiguous candidate. A tightly cropped part of a sofa is not '
    'evidence that the whole object is single-seat. If every candidate remains ambiguous, '
    'still select the best-supported candidate and mention the uncertainty briefly; selection '
    'does not confirm its category.'
)

TV_CHOICE_GUIDANCE = (
    'Prefer candidates showing identifiable physical display hardware over wall rectangles '
    'that could be framed artwork, posters, mirrors, or windows. Do not rank a screen lower '
    'only because it is off or displaying art. If every candidate remains ambiguous, still '
    'select the best-supported candidate and briefly state the ambiguity; this relative '
    'selection is not a TV confirmation.'
)

def build_prompt(target, *, target_confidence=False, multiview=False, count=1,
                 candidate_ids=None, policy='legacy', choice_id_policy='global_v1'):
    """Keep legacy bytes stable; semantic guidance changes only requested categories."""
    if policy not in PROMPT_POLICIES:
        raise ValueError('Unknown Qwen prompt policy')
    if choice_id_policy not in CHOICE_ID_POLICIES:
        raise ValueError('Unknown Qwen choice ID policy')
    if candidate_ids is not None:
        wire_ids = (list(range(1, len(candidate_ids) + 1))
                    if choice_id_policy == 'image_order_v1' else list(candidate_ids))
        prompt = CHOICE_PROMPT.format(target=target, ids=wire_ids)
        if choice_id_policy == 'image_order_v1':
            prompt += ' ' + IMAGE_ORDER_GUIDANCE
    else:
        template = MULTIVIEW_PROMPT if multiview else (CONFIDENCE_PROMPT if target_confidence else PROMPT)
        prompt = template.format(target=target, count=count)
    if policy == 'semantic_guard_v1':
        if target in ('chair', 'couch'):
            prompt += ' ' + (SEATING_CHOICE_GUIDANCE if candidate_ids is not None else SEATING_REVIEW_GUIDANCE)
        elif target == 'tv':
            prompt += ' ' + (TV_CHOICE_GUIDANCE if candidate_ids is not None else TV_REVIEW_GUIDANCE)
    return prompt


def parse_choice(data, ids):
    choice = data['choices'][0]
    if choice['finish_reason'] != 'stop':
        raise ValueError('Incomplete candidate comparison')
    answer = json.loads(choice['message']['content'])
    if set(answer) != {'outcome', 'candidate_id', 'reason'}:
        raise ValueError('Unexpected comparison schema')
    outcome, ident = answer['outcome'], answer['candidate_id']
    if (outcome != 'selected' or type(ident) is not int or ident not in ids or
            not isinstance(answer['reason'], str) or not 0 < len(answer['reason']) <= 2000):
        raise ValueError('Invalid comparison choice')
    return answer


def read_credentials():
    """Read only two dotenv values as data; never execute or log the file."""
    values = {}
    path = Path(__file__).resolve().parents[1] / 'config/vlm.env'
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip().removeprefix('export ')
            if '=' in line and not line.startswith('#'):
                key, value = line.split('=', 1)
                if key.strip() in ('DASHSCOPE_API_KEY', 'DASHSCOPE_BASE_URL'):
                    values[key.strip()] = value.strip().strip('\"\'')
    for key in ('DASHSCOPE_API_KEY', 'DASHSCOPE_BASE_URL'):
        if os.environ.get(key):
            values[key] = os.environ[key]
    if not values.get('DASHSCOPE_API_KEY'):
        raise RuntimeError('Configure DASHSCOPE_API_KEY in config/vlm.env')
    if values.get('DASHSCOPE_BASE_URL', BASE_URL).rstrip('/') != BASE_URL:
        raise RuntimeError('This price-guarded experiment requires the verified Beijing endpoint')
    return values['DASHSCOPE_API_KEY']


def parse_answer(data, target_confidence=False):
    choice = data['choices'][0]
    if choice['finish_reason'] != 'stop':
        raise ValueError('Incomplete Qwen answer')
    answer = json.loads(choice['message']['content'])
    keys = {'decision', 'quality', 'visible_category', 'reason'}
    if target_confidence:
        keys.add('target_confidence')
    if set(answer) != keys:
        raise ValueError('Unexpected Qwen answer schema')
    if target_confidence:
        value = answer['target_confidence']
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError('Invalid Qwen target confidence')
    if answer['decision'] not in ('confirmed', 'rejected', 'uncertain'):
        raise ValueError('Invalid Qwen decision')
    if answer['quality'] not in ('adequate', 'insufficient'):
        raise ValueError('Invalid Qwen evidence quality')
    if any(not isinstance(answer[k], str) or not 0 < len(answer[k]) <= 2000
           for k in ('visible_category', 'reason')):
        raise ValueError('Invalid Qwen explanation')
    return answer


def transport_diagnostic(exc):
    """Keep typed causes and known SSL codes, never request/exception text."""
    pending, seen, chain, codes = [exc], set(), [], set()
    ssl_codes = ('CERTIFICATE_VERIFY_FAILED', 'HOSTNAME_MISMATCH',
                 'UNEXPECTED_EOF_WHILE_READING', 'EOF_OCCURRED_IN_VIOLATION_OF_PROTOCOL',
                 'WRONG_VERSION_NUMBER', 'TLSV1_ALERT_PROTOCOL_VERSION',
                 'TLSV1_ALERT_INTERNAL_ERROR')
    interrupted = False
    certificate_error = False
    while pending and len(chain) < 12:
        error = pending.pop(0)
        if not isinstance(error, BaseException) or id(error) in seen:
            continue
        seen.add(id(error))
        item = {'type': type(error).__name__}
        for field in ('errno', 'verify_code'):
            value = getattr(error, field, None)
            if type(value) is int:
                item[field] = value
        # Only recognize fixed markers; arbitrary transport text may contain
        # credentials, URLs or request data and must never reach the ledger.
        message = str(error).upper()
        found = [code for code in ssl_codes if code in message]
        if found:
            item['ssl_codes'] = found
            codes.update(found)
        certificate_error |= isinstance(error, ssl.SSLCertVerificationError)
        interrupted |= isinstance(error, (ssl.SSLEOFError, ConnectionResetError,
                                           ConnectionAbortedError, BrokenPipeError))
        interrupted |= (isinstance(error, OSError) and
                        error.errno in (errno.ECONNRESET, errno.ECONNABORTED, errno.EPIPE))
        chain.append(item)
        pending.extend([getattr(error, '__cause__', None),
                        getattr(error, '__context__', None), getattr(error, 'reason', None)])
        pending.extend(arg for arg in error.args if isinstance(arg, BaseException))
    certificate_error |= bool(codes & {'CERTIFICATE_VERIFY_FAILED', 'HOSTNAME_MISMATCH'})
    interrupted |= bool(codes & {'UNEXPECTED_EOF_WHILE_READING',
                                 'EOF_OCCURRED_IN_VIOLATION_OF_PROTOCOL'})
    transport_error = isinstance(exc, (requests.ConnectionError, requests.Timeout))
    if certificate_error:
        category = 'certificate_verification'
    elif codes & {'WRONG_VERSION_NUMBER', 'TLSV1_ALERT_PROTOCOL_VERSION'}:
        category = 'tls_configuration'
    elif isinstance(exc, requests.ConnectTimeout):
        category = 'connect_timeout'
    elif isinstance(exc, requests.ReadTimeout):
        category = 'response_timeout'
    elif transport_error and interrupted:
        category = 'connection_interrupted'
    else:
        category = 'unclassified_or_nonretryable'
    return {'category': category, 'chain': chain,
            'retryable': transport_error and not certificate_error and
                         category in ('connect_timeout', 'response_timeout', 'connection_interrupted')}


# Fixed validation labels avoid leaking arbitrary exception/request text. The
# response body is saved only for response-stage failures, never HTTP headers.
_RESPONSE_ERRORS = {
    'Usage outside the reserved request bounds': 'usage_bounds',
    'Cost exceeds reservation': 'reservation_exceeded',
    'Incomplete candidate comparison': 'incomplete_comparison',
    'Unexpected comparison schema': 'comparison_schema',
    'Invalid comparison choice': 'comparison_choice',
    'Incomplete Qwen answer': 'incomplete_answer',
    'Unexpected Qwen answer schema': 'answer_schema',
    'Invalid Qwen target confidence': 'target_confidence',
    'Invalid Qwen decision': 'decision',
    'Invalid Qwen evidence quality': 'evidence_quality',
    'Invalid Qwen explanation': 'explanation',
}


def response_failure_evidence(response, data, exc, validation_step, key):
    """Keep bounded, redacted provider evidence without weakening validation."""
    code = ('invalid_json' if isinstance(exc, json.JSONDecodeError) else
            _RESPONSE_ERRORS.get(str(exc), 'invalid_response'))
    evidence = {'response_validation_step': validation_step,
                'response_validation_code': code}
    raw = getattr(response, 'text', None)
    if not isinstance(raw, str) and data is not None:
        raw = json.dumps(data, ensure_ascii=False)
    if isinstance(raw, str):
        raw = raw.replace(key, '[REDACTED]')
        evidence.update(response_body=raw[:65536],
                        response_body_truncated=len(raw) > 65536,
                        response_body_sha256=hashlib.sha256(raw.encode()).hexdigest())
    return evidence


class QwenTransportError(RuntimeError):
    """A failed HTTP request, with safe linkage for episode-level recovery."""
    def __init__(self, call_id, diagnostic, error_type):
        self.call_id = call_id
        self.retryable = diagnostic['retryable']
        super().__init__(f'Qwen verification failed ({error_type}, {diagnostic["category"]}); '
                         'no BLIP or silent navigation fallback')


class QwenReviewer:
    def __init__(self, ledger, model=MODEL, target_confidence=False,
                 plant_confirm_threshold=None, prompt_policy='legacy', choice_id_policy='global_v1'):
        if prompt_policy not in PROMPT_POLICIES:
            raise ValueError('Unknown Qwen prompt policy')
        if choice_id_policy not in CHOICE_ID_POLICIES:
            raise ValueError('Unknown Qwen choice ID policy')
        self.choice_id_policy = choice_id_policy
        self.prompt_policy = prompt_policy
        if plant_confirm_threshold is not None:
            if (not target_confidence or type(plant_confirm_threshold) not in (int, float) or
                    not math.isfinite(plant_confirm_threshold) or
                    not 0 <= plant_confirm_threshold <= 1):
                raise ValueError('Plant confirmation threshold requires target confidence and [0, 1]')
        self.plant_confirm_threshold = plant_confirm_threshold
        if model != MODEL:
            raise ValueError('Use the verified Qwen model name for this controlled experiment')
        self.key = read_credentials()
        self.model = model
        self.target_confidence = target_confidence
        self.ledger = Path(ledger)
        if not self.ledger.is_file():
            raise RuntimeError('Initialize the shared budget ledger before calling Qwen')

    def _update(self, call_id, record=None):
        # A separate lock survives atomic ledger replacement and serializes all
        # evaluator/smoke processes. Reservations remain after crashes/timeouts.
        with self.ledger.with_suffix('.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = json.loads(self.ledger.read_text())
            if not 0 < state['limit_cny'] <= 20:
                raise RuntimeError('Qwen budget exceeds authorized 20 CNY')
            calls = state['calls']
            if record is None:
                total = state['initial_spent_cny'] + sum(c['charged_cny'] for c in calls.values())
                if total + RESERVE_CNY > state['limit_cny']:
                    raise RuntimeError('Qwen budget exhausted; evaluation stopped without fallback')
                calls[call_id] = {'state': 'reserved', 'charged_cny': RESERVE_CNY,
                                  'model': self.model, 'started_unix': time.time()}
            else:
                calls[call_id].update(record)
            state['spent_or_reserved_cny'] = state['initial_spent_cny'] + sum(
                c['charged_cny'] for c in calls.values())
            temporary = self.ledger.with_suffix('.tmp')
            with temporary.open('w') as stream:
                json.dump(state, stream, indent=2, ensure_ascii=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.ledger)

    def __call__(self, rgb, target):
        return self._request([rgb], target, multiview=False)

    def review_images(self, images, target):
        if not self.target_confidence or not isinstance(images, (list, tuple)) or not 1 <= len(images) <= 5:
            raise ValueError('Multi-view confidence review requires 1 to 5 images')
        return self._request(images, target, multiview=True)

    def compare_candidates(self, images, target, candidate_ids):
        if (not isinstance(candidate_ids, (list, tuple)) or not 2 <= len(candidate_ids) <= 5 or
                len(images) != len(candidate_ids) or len(set(candidate_ids)) != len(candidate_ids) or
                any(type(i) is not int or i < 1 for i in candidate_ids)):
            raise ValueError('Comparison requires 2 to 5 distinct positive candidate IDs')
        return self._request(images, target, multiview=True, candidate_ids=list(candidate_ids))

    def _request(self, images, target, multiview, candidate_ids=None):
        if target not in ('chair', 'bed', 'couch', 'tv', 'potted plant', 'toilet'):
            raise ValueError('Unknown ObjectNav target')
        pngs = []
        for rgb in images:
            if rgb.ndim != 3 or rgb.shape[2] != 3 or min(rgb.shape[:2]) < 1 or max(rgb.shape[:2]) > 256:
                raise ValueError('Qwen expects linked RGB crops, each <=256 pixels')
            image = io.BytesIO()
            Image.fromarray(rgb).save(image, format='PNG')
            pngs.append(image.getvalue())
        prompt = build_prompt(target, target_confidence=self.target_confidence, multiview=multiview,
                              count=len(images), candidate_ids=candidate_ids, policy=self.prompt_policy,
                              choice_id_policy=self.choice_id_policy)
        local_ids = candidate_ids is not None and self.choice_id_policy == 'image_order_v1'
        wire_ids = list(range(1, len(images) + 1)) if local_ids else candidate_ids
        content = [{'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' +
                   base64.b64encode(png).decode()}} for png in pngs]
        content.append({'type': 'text', 'text': prompt})
        payload = {'model': self.model, 'enable_thinking': False, 'temperature': 0,
                   'max_tokens': MAX_TOKENS, 'response_format': {'type': 'json_object'},
                   'messages': [{'role': 'user', 'content': content}]}
        request_id, attempt_ids = uuid.uuid4().hex, []
        request_started = time.monotonic()
        for attempt in range(len(RETRY_DELAYS) + 1):
            call_id = uuid.uuid4().hex
            # Each HTTP attempt has its own reservation, including lost replies.
            # Exhaustion stops before sending; retries cannot reset the budget.
            self._update(call_id)
            attempt_ids.append(call_id)
            started = time.monotonic()
            metadata = {'image_sha256': hashlib.sha256(pngs[0]).hexdigest(), 'target': target,
                        'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                        'request_id': request_id, 'transport_attempt': attempt + 1,
                        'transport_attempt_ids': list(attempt_ids)}
            if self.prompt_policy != 'legacy':
                metadata['prompt_policy'] = self.prompt_policy
            if multiview:
                metadata.update(image_count=len(pngs),
                                image_sha256s=[hashlib.sha256(png).hexdigest() for png in pngs],
                                image_selection='candidate_best_comparison_v1' if candidate_ids is not None else 'candidate_top5_joint_v1')
            if candidate_ids is not None:
                metadata['candidate_ids'] = candidate_ids
                if local_ids:
                    metadata.update(choice_id_policy=self.choice_id_policy,
                                    api_candidate_ids=wire_ids,
                                    candidate_id_map=[{'image_index': i, 'candidate_id': ident}
                                                      for i, ident in enumerate(candidate_ids, 1)])
            self._update(call_id, metadata)
            response, data, stage = None, None, 'transport'
            validation_step = None
            try:
                response = requests.post(BASE_URL + '/chat/completions', json=payload,
                                         headers={'Authorization': 'Bearer ' + self.key},
                                         timeout=(10, 60), allow_redirects=False, verify=True)
                stage = 'response'
                validation_step = 'http_status'
                metadata['http_status'] = response.status_code
                if response.status_code != 200:
                    try:
                        error = response.json().get('error', {})
                        metadata['provider_error_code'] = str(error.get('code', 'unknown'))[:100]
                    except (ValueError, AttributeError):
                        pass
                    raise RuntimeError(f'Qwen HTTP {response.status_code}')
                validation_step = 'json_body'
                data = response.json()
                validation_step = 'usage'
                usage = data['usage']
                inp, out = usage['prompt_tokens'], usage['completion_tokens']
                if (type(inp) is not int or type(out) is not int or
                        not 0 < inp <= 16384 or not 0 <= out <= MAX_TOKENS):
                    raise ValueError('Usage outside the reserved request bounds')
                validation_step = 'cost'
                cost = (inp + 10 * out) / 1e6
                if cost > RESERVE_CNY:
                    raise ValueError('Cost exceeds reservation')
                validation_step = 'candidate_choice' if candidate_ids is not None else 'category_answer'
                answer = (parse_choice(data, wire_ids) if candidate_ids is not None
                          else parse_answer(data, self.target_confidence))
                if local_ids:
                    # Validate in the API's local domain, then map exactly once.
                    # Preserve the original reply; never interpret its free text
                    # or guess that an out-of-range ID meant another candidate.
                    metadata['raw_answer'] = dict(answer)
                    answer = {**answer, 'candidate_id': candidate_ids[answer['candidate_id'] - 1]}
                decision = (answer['outcome'] if candidate_ids is not None else
                            answer['decision'] if answer['quality'] == 'adequate' else 'uncertain')
                if candidate_ids is None and target == 'potted plant' and self.plant_confirm_threshold is not None:
                    # Q6 plant policy: category score is the acceptance criterion,
                    # including uncertain/insufficient answers. Preserve the raw
                    # answer and record this local policy separately for audits.
                    if answer['target_confidence'] >= self.plant_confirm_threshold:
                        decision = 'confirmed'
                    elif decision == 'confirmed':
                        decision = 'uncertain'
                    metadata.update(effective_decision=decision,
                                    decision_policy='plant_target_confidence_v1',
                                    plant_confirm_threshold=self.plant_confirm_threshold)
                metadata.update(state='completed', charged_cny=cost, usage=usage,
                                response_id=data.get('id'), model_returned=data.get('model'),
                                answer=answer, seconds=time.monotonic() - started,
                                request_seconds=time.monotonic() - request_started)
                metadata = json.loads(json.dumps(metadata).replace(self.key, '[REDACTED]'))
                self._update(call_id, metadata)
                return {'decision': decision, 'call_id': call_id, **metadata}
            except Exception as exc:
                if stage == 'response':
                    metadata.update(response_failure_evidence(
                        response, data, exc, validation_step, self.key))
                diagnostic = transport_diagnostic(exc)
                retry = stage == 'transport' and diagnostic['retryable'] and attempt < len(RETRY_DELAYS)
                delay = RETRY_DELAYS[attempt] if retry else None
                metadata.update(state='error', error_type=type(exc).__name__,
                                error_stage=stage, transport=diagnostic, retry_scheduled=retry,
                                retry_delay_seconds=delay, seconds=time.monotonic() - started,
                                request_seconds=time.monotonic() - request_started)
                metadata = json.loads(json.dumps(metadata).replace(self.key, '[REDACTED]'))
                self._update(call_id, metadata)
                if retry:
                    print(f'[Qwen transport] {diagnostic["category"]}; '
                          f'attempt {attempt + 1}/3 failed, retry in {delay}s', flush=True)
                    time.sleep(delay)
                    continue
                if stage == 'transport':
                    raise QwenTransportError(call_id, diagnostic, type(exc).__name__) from None
                raise RuntimeError(f'Qwen verification failed ({type(exc).__name__}, '
                                   f'{metadata["response_validation_code"]}, call_id={call_id}); '
                                   'no BLIP or silent navigation fallback') from None
            finally:
                if response is not None:
                    response.close()
