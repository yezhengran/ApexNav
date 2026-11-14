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
# Conservative reservation for one <=256px image and a short prompt. Actual
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


class QwenTransportError(RuntimeError):
    """A failed HTTP request, with safe linkage for episode-level recovery."""
    def __init__(self, call_id, diagnostic, error_type):
        self.call_id = call_id
        self.retryable = diagnostic['retryable']
        super().__init__(f'Qwen verification failed ({error_type}, {diagnostic["category"]}); '
                         'no BLIP or silent navigation fallback')


class QwenReviewer:
    def __init__(self, ledger, model=MODEL, target_confidence=False,
                 plant_confirm_threshold=None):
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
        if target not in ('chair', 'bed', 'couch', 'tv', 'potted plant', 'toilet'):
            raise ValueError('Unknown ObjectNav target')
        if rgb.ndim != 3 or rgb.shape[2] != 3 or max(rgb.shape[:2]) > 256:
            raise ValueError('Qwen expects the unchanged linked-review RGB crop, <=256 pixels')
        image = io.BytesIO()
        Image.fromarray(rgb).save(image, format='PNG')
        png = image.getvalue()
        prompt = (CONFIDENCE_PROMPT if self.target_confidence else PROMPT).format(target=target)
        payload = {'model': self.model, 'enable_thinking': False, 'temperature': 0,
                   'max_tokens': MAX_TOKENS, 'response_format': {'type': 'json_object'},
                   'messages': [{'role': 'user', 'content': [
                       {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' +
                                                        base64.b64encode(png).decode()}},
                       {'type': 'text', 'text': prompt}]}]}
        request_id, attempt_ids = uuid.uuid4().hex, []
        request_started = time.monotonic()
        for attempt in range(len(RETRY_DELAYS) + 1):
            call_id = uuid.uuid4().hex
            # Each HTTP attempt has its own reservation, including lost replies.
            # Exhaustion stops before sending; retries cannot reset the budget.
            self._update(call_id)
            attempt_ids.append(call_id)
            started = time.monotonic()
            metadata = {'image_sha256': hashlib.sha256(png).hexdigest(), 'target': target,
                        'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                        'request_id': request_id, 'transport_attempt': attempt + 1,
                        'transport_attempt_ids': list(attempt_ids)}
            self._update(call_id, metadata)
            response, stage = None, 'transport'
            try:
                response = requests.post(BASE_URL + '/chat/completions', json=payload,
                                         headers={'Authorization': 'Bearer ' + self.key},
                                         timeout=(10, 60), allow_redirects=False, verify=True)
                stage = 'response'
                metadata['http_status'] = response.status_code
                if response.status_code != 200:
                    try:
                        error = response.json().get('error', {})
                        metadata['provider_error_code'] = str(error.get('code', 'unknown'))[:100]
                    except (ValueError, AttributeError):
                        pass
                    raise RuntimeError(f'Qwen HTTP {response.status_code}')
                data = response.json()
                usage = data['usage']
                inp, out = usage['prompt_tokens'], usage['completion_tokens']
                if (type(inp) is not int or type(out) is not int or
                        not 0 < inp <= 16384 or not 0 <= out <= MAX_TOKENS):
                    raise ValueError('Usage outside the reserved request bounds')
                cost = (inp + 10 * out) / 1e6
                if cost > RESERVE_CNY:
                    raise ValueError('Cost exceeds reservation')
                answer = parse_answer(data, self.target_confidence)
                decision = answer['decision'] if answer['quality'] == 'adequate' else 'uncertain'
                if target == 'potted plant' and self.plant_confirm_threshold is not None:
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
                                   f'{diagnostic["category"]}); '
                                   'no BLIP or silent navigation fallback') from None
            finally:
                if response is not None:
                    response.close()
