"""Bailian visual verification only; never replaces ApexNav's map ITM scores."""

import base64
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
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


def parse_answer(data):
    choice = data['choices'][0]
    if choice['finish_reason'] != 'stop':
        raise ValueError('Incomplete Qwen answer')
    answer = json.loads(choice['message']['content'])
    if set(answer) != {'decision', 'quality', 'visible_category', 'reason'}:
        raise ValueError('Unexpected Qwen answer schema')
    if answer['decision'] not in ('confirmed', 'rejected', 'uncertain'):
        raise ValueError('Invalid Qwen decision')
    if answer['quality'] not in ('adequate', 'insufficient'):
        raise ValueError('Invalid Qwen evidence quality')
    if any(not isinstance(answer[k], str) or not 0 < len(answer[k]) <= 2000
           for k in ('visible_category', 'reason')):
        raise ValueError('Invalid Qwen explanation')
    return answer


class QwenReviewer:
    def __init__(self, ledger, model=MODEL):
        if model != MODEL:
            raise ValueError('Use the verified Qwen model name for this controlled experiment')
        self.key = read_credentials()
        self.model = model
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
        prompt = PROMPT.format(target=target)
        payload = {'model': self.model, 'enable_thinking': False, 'temperature': 0,
                   'max_tokens': MAX_TOKENS, 'response_format': {'type': 'json_object'},
                   'messages': [{'role': 'user', 'content': [
                       {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' +
                                                        base64.b64encode(png).decode()}},
                       {'type': 'text', 'text': prompt}]}]}
        call_id = uuid.uuid4().hex
        self._update(call_id)
        started = time.monotonic()
        metadata = {'image_sha256': hashlib.sha256(png).hexdigest(), 'target': target,
                    'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest()}
        try:
            response = requests.post(BASE_URL + '/chat/completions', json=payload,
                                     headers={'Authorization': 'Bearer ' + self.key},
                                     timeout=(10, 60), allow_redirects=False)
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
            answer = parse_answer(data)
            metadata.update(state='completed', charged_cny=cost, usage=usage,
                            response_id=data.get('id'), model_returned=data.get('model'),
                            answer=answer, seconds=time.monotonic() - started)
            # Never retain authorization headers or transport exception bodies.
            metadata = json.loads(json.dumps(metadata).replace(self.key, '[REDACTED]'))
            self._update(call_id, metadata)
            decision = answer['decision'] if answer['quality'] == 'adequate' else 'uncertain'
            return {'decision': decision, 'call_id': call_id, **metadata}
        except Exception as exc:
            metadata = json.loads(json.dumps(metadata).replace(self.key, '[REDACTED]'))
            self._update(call_id, {**metadata, 'state': 'error',
                                   'error_type': type(exc).__name__,
                                   'seconds': time.monotonic() - started})
            raise RuntimeError(f'Qwen verification failed ({type(exc).__name__}); '
                               'no BLIP or silent navigation fallback') from None
