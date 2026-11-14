"""Recover only an unfinished episode with an explicitly linked network error."""

import hashlib
import json
from pathlib import Path
import shutil
import time

import requests

from basic_utils.record_episode.experiment import read_rows
from vlm.qwen_review import BASE_URL, RESERVE_CNY


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def eligible_failure(output, ledger, evaluation_started):
    """No success/failure CSV row, stale error or model answer can trigger a retry."""
    output = Path(output)
    number = len(read_rows(output)) + 1
    attempts = sorted((output / 'verification' / f'episode-{number:04d}').glob('attempt-*'))
    if not attempts:
        return None
    attempt = attempts[-1]
    events = [json.loads(line) for line in (attempt / 'events.jsonl').read_text().splitlines()]
    errors = [e for e in events if e['event'] == 'verification_transport_error']
    if any(e['event'] == 'episode_end' for e in events) or len(errors) != 1:
        return None
    error = errors[0]
    bill = json.loads(Path(ledger).read_text())['calls'].get(error['call_id'], {})
    diagnostic = bill.get('transport', {})
    if not (error['retryable'] and bill.get('state') == 'error' and
            bill.get('error_stage') == 'transport' and diagnostic.get('retryable') and
            diagnostic.get('category') in ('connection_interrupted', 'connect_timeout', 'response_timeout') and
            bill.get('started_unix', 0) >= evaluation_started and 'http_status' not in bill and
            bill.get('charged_cny') == RESERVE_CNY):
        return None
    crop = Path(error['crop']).resolve()
    if not crop.is_relative_to(attempt.resolve()) or digest(crop) != bill['image_sha256']:
        raise RuntimeError('Network error image does not match the unfinished episode')
    return {'episode': number, 'attempt': str(attempt), 'call_id': error['call_id'],
            'transport': diagnostic, 'events_sha256': digest(attempt / 'events.jsonl')}


def recover_episode(output, ledger, evaluation_started, limit, source_sha256,
                    wait=time.sleep, probe=None):
    """Archive first; reconnect without inference, then authorize one restart.

    The persistent per-episode limit survives manual process restarts. API
    billing reservations and original attempts are never removed or overwritten.
    """
    if not limit:
        return None
    output = Path(output)
    failure = eligible_failure(output, ledger, evaluation_started)
    if failure is None:
        return None
    number = failure['episode']
    parent = output / 'network-recovery' / f'episode-{number:04d}'
    prior = sorted(parent.glob('retry-*/recovery.json'))
    if len(prior) >= limit:
        return None
    ledger_data = json.loads(Path(ledger).read_text())
    if not 0 < ledger_data['limit_cny'] <= 20 or (
            ledger_data['spent_or_reserved_cny'] + RESERVE_CNY > ledger_data['limit_cny']):
        return None
    record = {**failure, 'retry': len(prior) + 1, 'limit': limit,
              'source_sha256': source_sha256, 'created_unix': time.time(),
              'completed_rows': read_rows(output), 'state': 'preserving'}
    archive = parent / f'retry-{record["retry"]:02d}'
    archive.mkdir(parents=True, exist_ok=False)
    for name in ('episodes.csv', 'summary.json', 'episode_set.json',
                 f'episode-{number:04d}.partial.mp4'):
        if (output / name).exists():
            shutil.copy2(output / name, archive / name)
    shutil.copy2(ledger, archive / 'budget.json')
    # The verification attempt itself stays at its original immutable path.
    record['preserved_hashes'] = {p.name: digest(p) for p in archive.iterdir() if p.is_file()}
    path = archive / 'recovery.json'

    def save():
        path.write_text(json.dumps(record, indent=2) + '\n')

    record.update(state='reconnecting', delay_seconds=30 * record['retry'])
    save()
    print(f'[Network recovery] episode {number}: preserved incomplete attempt; '
          f'reconnect in {record["delay_seconds"]}s, restart {record["retry"]}/{limit}', flush=True)
    wait(record['delay_seconds'])
    try:
        # No key, image or inference. Certificate verification remains enabled.
        if probe is None:
            with requests.get(BASE_URL + '/models', timeout=(10, 15),
                              allow_redirects=False, verify=True) as response:
                status = response.status_code
        else:
            status = probe()
        if status != 401:
            raise RuntimeError('Unauthenticated endpoint did not return expected 401')
        if read_rows(output) != record['completed_rows']:
            raise RuntimeError('Completed rows changed during network recovery')
        record.update(state='restart_ready', probe_http_status=status)
        save()
        return record
    except Exception as error:
        record.update(state='reconnect_failed', error_type=type(error).__name__)
        save()
        raise RuntimeError(f'Network reconnection failed ({type(error).__name__}); preserved attempt') from None
