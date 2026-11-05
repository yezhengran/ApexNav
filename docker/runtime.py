#!/usr/bin/env python3
"""Start, inspect, and stop only the ApexNav services owned by this launcher."""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "outputs" / "runtime"
SERVICES = {
    "blip2": ("vlm.itm.blip2itm", 12182, "blip2itm"),
    "yolo": ("vlm.detector.yolov7", 12184, "yolov7"),
    "sam": ("vlm.segmentor.sam", 12183, "mobile_sam"),
    "dino": ("vlm.detector.grounding_dino", 12181, "gdino"),
}


def ready(name: str) -> bool:
    _, port, model = SERVICES[name]
    try:
        token = os.environ.get("APEXNAV_VLM_SERVER_TOKEN", "")
        headers = {"Authorization": f"Bearer {token}"} if token else {}
        request = urllib.request.Request(f"http://127.0.0.1:{port}/healthz", headers=headers)
        with urllib.request.urlopen(request, timeout=2) as r:
            payload = json.load(r)
        return payload == {"status": "ready", "model": model}
    except (OSError, ValueError, urllib.error.URLError):
        return False


def identity(pid: int) -> str | None:
    try:
        # Field 22 is the kernel start time; it distinguishes recycled PIDs.
        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if stat[0] == "Z":
            return None
        return stat[19]
    except (OSError, IndexError):
        return None


def record_path(name: str) -> Path:
    return STATE / f"{name}.json"


def owned_pid(name: str) -> int | None:
    try:
        record = json.loads(record_path(name).read_text())
        pid = int(record["pid"])
        if identity(pid) == record["start_time"]:
            return pid
    except (OSError, ValueError, KeyError):
        pass
    return None


def start(name: str, timeout: int) -> None:
    module, port, _ = SERVICES[name]
    if ready(name):
        print(f"[READY] {name} port={port}", flush=True)
        return
    if owned_pid(name):
        raise RuntimeError(f"{name} is still starting; inspect {STATE / (name + '.log')}")
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", port)) == 0:
            raise RuntimeError(f"port {port} is occupied by another service")
    STATE.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.update(
        PYTHONUNBUFFERED="1", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
        MALLOC_ARENA_MAX="2", TOKENIZERS_PARALLELISM="false",
    )
    with (STATE / f"{name}.log").open("a") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", module, "--port", str(port)],
            cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    record_path(name).write_text(json.dumps({
        "pid": process.pid, "start_time": identity(process.pid), "module": module,
    }))
    print(f"[STARTING] {name} pid={process.pid} log={STATE / (name + '.log')}", flush=True)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"{name} exited ({process.returncode}); see {STATE / (name + '.log')}")
        if ready(name):
            print(f"[READY] {name} port={port}", flush=True)
            return
        time.sleep(1)
    raise RuntimeError(f"{name} startup timed out; see {STATE / (name + '.log')}")


def stop(name: str) -> None:
    pid = owned_pid(name)
    if pid is None:
        print(f"[STOPPED] {name} (no owned process)")
        return
    os.killpg(pid, signal.SIGTERM)
    for _ in range(50):
        if owned_pid(name) is None:
            record_path(name).unlink(missing_ok=True)
            print(f"[STOPPED] {name}")
            return
        time.sleep(0.1)
    raise RuntimeError(f"{name} did not exit after SIGTERM; pid={pid}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["start", "status", "stop"])
    parser.add_argument("--service", choices=list(SERVICES))
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()
    names = [args.service] if args.service else list(SERVICES)
    if args.action == "stop":
        names.reverse()
    failed = False
    for name in names:
        if args.action == "start":
            start(name, args.timeout)
        elif args.action == "stop":
            stop(name)
        else:
            is_ready = ready(name)
            failed |= not is_ready
            print(f"[{'READY' if is_ready else 'DOWN'}] {name} port={SERVICES[name][1]} owned_pid={owned_pid(name)}")
    return int(failed)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
