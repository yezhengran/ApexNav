#!/usr/bin/env python3
"""Preflight assets/APIs, start the planner, then evaluate one real episode."""

import argparse
import datetime
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", nargs="?", choices=["hm3dv1", "hm3dv2", "mp3d"], default="hm3dv2")
    parser.add_argument("episode", nargs="?", type=int, default=0)
    args = parser.parse_args()
    if args.episode < 0:
        parser.error("episode must be nonnegative")
    os.chdir(ROOT)
    os.environ["ROS_DOMAIN_ID"] = os.environ.get("APEXNAV_ROS_DOMAIN_ID", "79")
    os.environ["ROS_LOCALHOST_ONLY"] = "1"
    subprocess.run([sys.executable, "docker/check_dataset.py", "--dataset", args.dataset], check=True)
    subprocess.run([sys.executable, "docker/check_vlm_http.py"], check=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    output = ROOT / f"outputs/runs/{args.dataset}-{args.episode}-{stamp}"
    output.mkdir(parents=True)
    print(f"Episode logs/results: {output}", flush=True)
    planner = None
    try:
        with (output / "planner.log").open("w") as log:
            planner = subprocess.Popen(
                ["ros2", "launch", "exploration_manager", "exploration.launch.py"],
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
            )
        time.sleep(3)
        if planner.poll() is not None:
            raise RuntimeError(f"Planner failed; see {output / 'planner.log'}")
        return subprocess.call([
            sys.executable, "-u", "habitat_evaluation.py", "--dataset", args.dataset,
            f"test_epi_num={args.episode}", "llm.llm_client.llm_client=none",
            f"video_output_path={output}",
        ])
    finally:
        if planner and planner.poll() is None:
            os.killpg(planner.pid, signal.SIGINT)
            try:
                planner.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(planner.pid, signal.SIGTERM)
                planner.wait(timeout=5)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except subprocess.CalledProcessError as error:
        print("Preflight failed; evaluation was not started.", file=sys.stderr)
        raise SystemExit(error.returncode)
