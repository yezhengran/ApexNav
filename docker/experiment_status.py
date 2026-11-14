#!/usr/bin/env python3
"""Read local experiment progress and actual processes without loading models."""
import argparse
import csv
import datetime
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--target', type=int)
    args = parser.parse_args()
    run = args.run
    rows = list(csv.DictReader((run / 'episodes.csv').open())) if (run / 'episodes.csv').exists() else []
    count = len(rows)
    successes = sum(int(r['success']) for r in rows)
    print(datetime.datetime.now().astimezone().isoformat(timespec='seconds'))
    print(f'Run: {run}')
    print(f'Completed: {count}/{args.target or "?"}; successes: {successes}; SR: {successes/max(1,count):.2%}')
    if rows:
        print(f'Last: #{count} {rows[-1]["target"]}, {rows[-1]["result"]}, {rows[-1]["steps"]} steps')
    found = []
    for proc in Path('/proc').glob('[0-9]*'):
        try:
            words = (proc / 'cmdline').read_bytes().split(b'\0')
            names = {Path(w.decode(errors='replace')).name for w in words if w}
            for name in ('run_batch.py', 'habitat_evaluation.py', 'exploration_node'):
                if name in names:
                    found.append((proc.name, name))
        except (PermissionError, FileNotFoundError, ProcessLookupError):
            pass
    print('Processes: ' + (', '.join(f'{name} PID={pid}' for pid, name in found) or 'none'))
    if (run / 'status.json').exists():
        print('Last batch exit (may be from before resume): ' + (run / 'status.json').read_text().strip())


if __name__ == '__main__':
    main()
