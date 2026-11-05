#!/usr/bin/env python3
"""Prepare and run a bounded local experiment, preserving logs and results."""

import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def stop_owned(process):
    if process is None or process.poll() is not None:
        return
    for sig, seconds in ((signal.SIGINT, 10), (signal.SIGTERM, 5)):
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            process.wait()
            return
        try:
            process.wait(timeout=seconds)
            return
        except subprocess.TimeoutExpired:
            pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def evaluation_command(dataset, episodes, output, video, max_steps=None):
    command = [
        sys.executable, "-u", "habitat_evaluation.py", "--dataset", dataset,
        "test_epi_num=-1", f"+max_episodes={episodes}",
        "llm.llm_client.llm_client=none", f"need_video={str(video).lower()}",
        f"video_output_path={output}",
    ]
    if max_steps is not None:
        command.append(f"habitat.environment.max_episode_steps={max_steps}")
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", nargs="?", default="hm3dv2",
                        choices=["hm3dv1", "hm3dv2", "mp3d"])
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, help="Diagnostic step cap; omit for the benchmark's default")
    parser.add_argument("--video", action="store_true", help="Stream MP4 per episode (bounded frame RAM)")
    parser.add_argument("--output", help="New experiment directory inside outputs/runs")
    parser.add_argument("--resume", action="store_true", help="Continue the same --output directory")
    parser.add_argument("--dry-run", action="store_true", help="Print invocation; create/start nothing")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if args.resume and not args.output:
        parser.error("--resume requires --output")
    stamp = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime("%Y%m%d-%H%M%S")
    output = Path(args.output or f"outputs/runs/{args.dataset}-{args.episodes}-{stamp}")
    output = (ROOT / output).resolve()
    if not output.is_relative_to(ROOT / "outputs/runs") or output == ROOT / "outputs/runs":
        parser.error("--output must name a directory below outputs/runs")
    command = evaluation_command(args.dataset, args.episodes, output, args.video, args.max_steps)
    spec = {"dataset": args.dataset, "episodes": args.episodes, "video": args.video,
            "llm_mode": "cached answers; no Ollama model loading"}
    if args.max_steps is not None:
        spec["max_steps"] = args.max_steps
    if args.dry_run:
        print(json.dumps({"spec": spec, "output": str(output), "command": command}, indent=2))
        return 0

    os.chdir(ROOT)
    os.environ["ROS_DOMAIN_ID"] = os.environ.get("APEXNAV_ROS_DOMAIN_ID", "79")
    os.environ["ROS_LOCALHOST_ONLY"] = "1"
    os.environ.setdefault("HABITAT_SIM_LOG", "error")
    os.environ.setdefault("MAGNUM_LOG", "quiet")
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    os.environ.setdefault("MALLOC_ARENA_MAX", "2")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    lock_path = ROOT / "outputs/runtime/evaluation.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another batch evaluation is running; do not share its ROS domain")
        manifest = output / "experiment.json"
        if args.resume:
            if not manifest.is_file() or json.loads(manifest.read_text()) != spec:
                raise RuntimeError("Resume requires an existing experiment with identical dataset/count/video settings")
        else:
            output.mkdir(parents=True, exist_ok=False)
            manifest.write_text(json.dumps(spec, indent=2) + "\n")
        print(f"Experiment directory: {output}", flush=True)
        subprocess.run([sys.executable, "docker/check_dataset.py", "--dataset", args.dataset], check=True)
        subprocess.run([sys.executable, "docker/check_vlm_http.py"], check=True)

        from habitat.config.default_structured_configs import HabitatConfigPlugin, register_hydra_plugin
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf

        register_hydra_plugin(HabitatConfigPlugin)
        with initialize_config_dir(config_dir=str(ROOT / "config"), version_base=None):
            config = compose(config_name=f"habitat_eval_{args.dataset}", overrides=command[5:])
        (output / "config.yaml").write_text(OmegaConf.to_yaml(config, resolve=True))
        available_kib = next(int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines()
                             if line.startswith("MemAvailable:"))
        if available_kib < 3 * 1024**2:
            print("[WARN] Less than 3 GiB host RAM available. Close memory-heavy apps; RAM may run out.", flush=True)
        planner = evaluation = None
        code, state = 1, "failed"
        try:
            with (output / "planner.log").open("a") as log:
                planner = subprocess.Popen(
                    ["ros2", "launch", "exploration_manager", "exploration.launch.py"],
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                )
            time.sleep(3)
            if planner.poll() is not None:
                raise RuntimeError(f"Planner exited; inspect {output / 'planner.log'}")
            with (output / "console.log").open("a", buffering=1) as log:
                evaluation = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                              text=True, bufsize=1, start_new_session=True)
                for line in evaluation.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                code = evaluation.wait()
            if code == 0:
                summary_path = output / "summary.json"
                completed = json.loads(summary_path.read_text())["completed_episodes"] if summary_path.exists() else 0
                if completed != args.episodes:
                    code = 1
                    raise RuntimeError(f"Only {completed}/{args.episodes} completed; not reporting a successful batch")
                state = "completed"
        except KeyboardInterrupt:
            state, code = "interrupted", 130
            print("\nInterrupted. Completed episode results are preserved.", flush=True)
        finally:
            stop_owned(evaluation)
            stop_owned(planner)
            (output / "status.json").write_text(json.dumps({"state": state, "exit_code": code}, indent=2) + "\n")
        print(f"Experiment {state}: {output}", flush=True)
        return code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError, subprocess.CalledProcessError) as error:
        print(f"Batch not completed: {error}", file=sys.stderr)
        raise SystemExit(1)
