#!/usr/bin/env python3
"""Prepare and run a bounded local experiment, preserving logs and results."""

import argparse
import datetime
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
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


def evaluation_command(dataset, episodes, output, video, max_steps=None,
                       method="baseline", seed=100, confirm_threshold=0.65, reject_threshold=0.20,
                       uncertain_after_steps=350, episode_index=-1, review_profile="legacy",
                       review_backend="blip2", qwen_budget_ledger=""):
    command = [
        sys.executable, "-u", "habitat_evaluation.py", "--dataset", dataset,
        f"test_epi_num={episode_index}", f"+max_episodes={episodes}",
        "llm.llm_client.llm_client=none", f"need_video={str(video).lower()}",
        f"video_output_path={output}",
        f"habitat.seed={seed}",
    ]
    if method != "baseline":
        command.extend([f"active_view.mode={method}",
                        f"active_view.itm_confirm_threshold={confirm_threshold}",
                        f"active_view.itm_reject_threshold={reject_threshold}",
                        f"active_view.uncertain_after_steps={uncertain_after_steps}"])
        if review_profile != "legacy":
            command.extend(["active_view.recent_evidence=true",
                            f"active_view.turn_first={str(review_profile == 'evidence_turn').lower()}"])
        if review_profile in ("linked", "linked_stop", "linked_stop_preserve"):
            prefix = '+' if dataset != 'hm3dv2' else ''
            command.extend([f"{prefix}active_view.linked_evidence=true",
                            f"{prefix}active_view.stop_review={str(review_profile in ('linked_stop', 'linked_stop_preserve')).lower()}"])
        if review_backend == 'qwen':
            command.extend(['active_view.backend=qwen',
                            f'active_view.qwen_budget_ledger={qwen_budget_ledger}'])
    if max_steps is not None:
        command.append(f"habitat.environment.max_episode_steps={max_steps}")
    return command


def runtime_sources():
    """Read tracked and non-ignored runtime files, including new uncommitted code."""
    # The bind-mounted checkout is owned by the host user; trust only this
    # explicit path for this invocation, never change global Git configuration.
    git = ["git", "-c", f"safe.directory={ROOT}"]
    revision = subprocess.check_output([*git, "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    names = subprocess.check_output(
        [*git, "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT).split(b"\0")
    sources = {}
    for name in sorted(set(n for n in names if n)):
        path = ROOT / os.fsdecode(name)
        # Historical ROS1 catkin entry; absent in this ROS2 image and not used by
        # colcon. Match the exact inert link, never relax other escaping paths.
        if (name == b'src/CMakeLists.txt' and path.is_symlink() and
                os.readlink(path) == '/opt/ros/noetic/share/catkin/cmake/toplevel.cmake' and
                not path.exists()):
            continue
        if (path.suffix in (".py", ".cpp", ".h", ".hpp", ".yaml", ".patch", ".sh", ".msg") or
                path.name in ("CMakeLists.txt", "package.xml") or
                name == b"docker/apexnav"):
            if not path.resolve().is_relative_to(ROOT.resolve()):
                raise RuntimeError(f"Runtime source escapes checkout: {path}")
            if path.exists():  # Deleted tracked files are absent from this snapshot.
                sources[name] = path.read_bytes()
    dirty = subprocess.check_output(
        [*git, "status", "--porcelain", "--untracked-files=normal"], cwd=ROOT, text=True).strip()
    return revision, sources, bool(dirty)


def source_digest(sources):
    digest = hashlib.sha256()
    for name, data in sorted(sources.items()):
        digest.update(name + b"\0" + data + b"\0")
    return digest.hexdigest()


def source_provenance(snapshot_dir=None):
    """Require a commit or an explicit, complete local runtime-source snapshot."""
    revision, sources, dirty = runtime_sources()
    if dirty and snapshot_dir is None:
        raise RuntimeError("Commit source/config changes before starting a reproducible batch")
    record = {"code_sha": revision, "source_sha256": source_digest(sources)}
    if snapshot_dir is not None:
        snapshot_dir.mkdir(parents=True, exist_ok=False)
        archive = snapshot_dir / "runtime-source.tar.gz"
        with tarfile.open(archive, "w:gz") as stream:
            for name, data in sorted(sources.items()):
                info = tarfile.TarInfo(os.fsdecode(name))
                info.size = len(data)
                stream.addfile(info, io.BytesIO(data))
        record.update(source_kind="worktree_snapshot", uncommitted=dirty,
                      source_archive=str(archive.relative_to(ROOT)),
                      archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
        (snapshot_dir / "files.json").write_text(json.dumps(
            {os.fsdecode(n): hashlib.sha256(d).hexdigest() for n, d in sources.items()}, indent=2) + "\n")
    return record


def resume_migration(old, current, name=None):
    """Only permit an explicitly reviewed, exact source fingerprint pair."""
    if name is None:
        if old["source_sha256"] != current["source_sha256"]:
            raise RuntimeError("Runtime source changed; start a separate experiment or use "
                               "an explicitly reviewed --resume-migration")
        return None
    registry = json.loads((ROOT / "docker/resume_migrations.json").read_text())
    migration = registry.get(name)
    if (migration is None or migration["from_source_sha256"] != old["source_sha256"] or
            migration["to_source_sha256"] != current["source_sha256"]):
        raise RuntimeError("Resume migration does not match the exact old/current source pair")
    return {"name": name, **migration}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", nargs="?", default="hm3dv2",
                        choices=["hm3dv1", "hm3dv2", "mp3d"])
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--max-steps", type=int, help="Diagnostic step cap; omit for the benchmark's default")
    parser.add_argument("--video", action="store_true", help="Stream MP4 per episode (bounded frame RAM)")
    parser.add_argument("--output", help="New experiment directory inside outputs/runs")
    parser.add_argument("--resume", action="store_true", help="Continue the same --output directory")
    parser.add_argument("--resume-migration", help="Named, reviewed tooling-only source migration")
    parser.add_argument("--snapshot-source", action="store_true",
                        help="Archive exact runtime source locally; permits testing before committing")
    parser.add_argument("--episode-index", type=int, default=-1,
                        help="Zero-based dataset index for a single diagnostic episode")
    parser.add_argument("--dry-run", action="store_true", help="Print invocation; create/start nothing")
    parser.add_argument("--method", choices=["baseline", "selected_review"],
                        default="baseline")
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--review-profile", choices=["legacy", "evidence", "evidence_turn", "linked", "linked_stop", "linked_stop_preserve"],
                        default="legacy", help="Independent selected-review observation ablations")
    parser.add_argument("--confirm-threshold", type=float, default=0.65)
    parser.add_argument('--review-backend', choices=['blip2', 'qwen'], default='blip2')
    parser.add_argument('--qwen-budget-ledger', default='',
                        help='Shared initialized local JSON ledger; never contains credentials')
    parser.add_argument("--reject-threshold", type=float, default=0.20)
    parser.add_argument("--uncertain-after-steps", type=int, default=350)
    args = parser.parse_args()
    if not 0 <= args.reject_threshold < args.confirm_threshold <= 1:
        parser.error("Require 0 <= reject-threshold < confirm-threshold <= 1")
    if args.uncertain_after_steps < 1:
        parser.error("--uncertain-after-steps must be positive")
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    if args.max_steps is not None and args.max_steps < 1:
        parser.error("--max-steps must be positive")
    if args.resume and not args.output:
        parser.error("--resume requires --output")
    if args.resume_migration and not args.resume:
        parser.error("--resume-migration requires --resume")
    if args.episode_index < -1 or (args.episode_index >= 0 and (args.episodes != 1 or args.resume)):
        parser.error("--episode-index requires --episodes 1 and a new diagnostic directory")
    if args.review_profile != "legacy" and args.method != "selected_review":
        parser.error("--review-profile requires --method selected_review")
    if args.review_backend == 'qwen':
        if args.method != 'selected_review' or args.review_profile != 'linked_stop_preserve':
            parser.error('This Qwen comparison requires selected_review / linked_stop_preserve')
        ledger = (ROOT / args.qwen_budget_ledger).resolve()
        if not args.qwen_budget_ledger or not ledger.is_file() or not ledger.is_relative_to(ROOT):
            parser.error('Qwen requires an existing budget ledger within the checkout')
        args.qwen_budget_ledger = str(ledger)
    elif args.qwen_budget_ledger:
        parser.error('--qwen-budget-ledger requires --review-backend qwen')
    if args.method != "baseline" and args.dataset != "hm3dv2":
        parser.error("Active-view experiments currently require hm3dv2")
    stamp = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).strftime("%Y%m%d-%H%M%S")
    output = Path(args.output or f"outputs/runs/{args.dataset}-{args.episodes}-{stamp}")
    output = (ROOT / output).resolve()
    if not output.is_relative_to(ROOT / "outputs/runs") or output == ROOT / "outputs/runs":
        parser.error("--output must name a directory below outputs/runs")
    command = evaluation_command(args.dataset, args.episodes, output, args.video, args.max_steps,
                                 args.method, args.seed, args.confirm_threshold,
                                 args.reject_threshold, args.uncertain_after_steps,
                                 args.episode_index, args.review_profile,
                                 args.review_backend, args.qwen_budget_ledger)
    spec = {"dataset": args.dataset, "episodes": args.episodes, "video": args.video,
            "method": args.method, "seed": args.seed,
            "llm_mode": "cached answers; no Ollama model loading"}
    if args.method != "baseline":
        spec["review"] = {"confirm_threshold": args.confirm_threshold,
                          "reject_threshold": args.reject_threshold,
                          "uncertain_after_steps": args.uncertain_after_steps}
        if args.review_backend == 'qwen':
            spec['review'] = {'backend': 'qwen', 'model': 'qwen3-vl-plus',
                              'endpoint': 'https://dashscope.aliyuncs.com/compatible-mode/v1',
                              'decision': 'structured three-way; no ITM thresholds',
                              'budget_ledger': args.qwen_budget_ledger,
                              'uncertain_after_steps': args.uncertain_after_steps}
    if args.max_steps is not None:
        spec["max_steps"] = args.max_steps
    if args.episode_index >= 0:
        spec["episode_index"] = args.episode_index
        spec["diagnostic_subset"] = True
    if args.review_profile != "legacy":
        spec["review_profile"] = args.review_profile
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
        if not args.resume and output.exists():
            raise RuntimeError("New experiment output already exists; refusing to overwrite")
        if args.resume and not manifest.is_file():
            raise RuntimeError("Resume output has no experiment manifest")
        snapshot = None
        if args.snapshot_source:
            snapshot = (output / "resume-provenance" / f"source-{time.time_ns()}" if args.resume
                        else output / "source-snapshot")
        provenance = source_provenance(snapshot)
        if args.resume:
            if not manifest.is_file() or json.loads(manifest.read_text()) != spec:
                raise RuntimeError("Resume requires identical dataset/count/video/method/seed settings; "
                                   "legacy runs without provenance must use a separate directory")
            old = json.loads((output / "provenance.json").read_text())
            migration = resume_migration(old, provenance, args.resume_migration)
        else:
            output.mkdir(parents=True, exist_ok=args.snapshot_source)
            manifest.write_text(json.dumps(spec, indent=2) + "\n")
            (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        print(f"Experiment directory: {output}", flush=True)
        subprocess.run([sys.executable, "docker/check_dataset.py", "--dataset", args.dataset], check=True)
        subprocess.run([sys.executable, "docker/check_vlm_http.py"], check=True)

        from habitat.config.default_structured_configs import HabitatConfigPlugin, register_hydra_plugin
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf

        register_hydra_plugin(HabitatConfigPlugin)
        with initialize_config_dir(config_dir=str(ROOT / "config"), version_base=None):
            config = compose(config_name=f"habitat_eval_{args.dataset}", overrides=command[5:])
        config_text = OmegaConf.to_yaml(config, resolve=True)
        if args.resume:
            if (output / "config.yaml").read_text() != config_text:
                raise RuntimeError("Resolved config changed; refusing to resume")
            # Preserve the original provenance and each previous attempt. A
            # tooling migration is visible, never relabelled as identical code.
            sys.path.insert(0, str(ROOT))
            from basic_utils.record_episode.experiment import read_rows
            rows = read_rows(output)
            attempts = output / "resume-provenance"
            attempts.mkdir(exist_ok=True)
            record = {"resumed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      "completed_before_resume": len(rows), "original": old,
                      "current": provenance, "migration": migration,
                      "episodes_csv_sha256": hashlib.sha256(
                          (output / "episodes.csv").read_bytes()).hexdigest() if rows else None,
                      "config_sha256": hashlib.sha256(config_text.encode()).hexdigest()}
            with (attempts / f"attempt-{time.time_ns()}.json").open("x") as stream:
                stream.write(json.dumps(record, indent=2) + "\n")
        else:
            (output / "config.yaml").write_text(config_text)
        available_kib = next(int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines()
                             if line.startswith("MemAvailable:"))
        if available_kib < 3 * 1024**2:
            print("[WARN] Less than 3 GiB host RAM available. Close memory-heavy apps; RAM may run out.", flush=True)
        planner = evaluation = None
        code, state = 1, "failed"
        try:
            with (output / "planner.log").open("a") as log:
                planner = subprocess.Popen(
                    ["ros2", "launch", "exploration_manager", "exploration.launch.py",
                     f"review_enabled:={str(args.method == 'selected_review').lower()}",
                     f"review_late_step:={args.uncertain_after_steps}",
                     f"review_lineage:={str(args.review_profile in ('linked', 'linked_stop', 'linked_stop_preserve')).lower()}",
                     f"review_stop:={str(args.review_profile in ('linked_stop', 'linked_stop_preserve')).lower()}",
                     f"review_preserve_stop:={str(args.review_profile == 'linked_stop_preserve').lower()}"],
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
            final_digest = source_digest(runtime_sources()[1])
            if final_digest != provenance["source_sha256"]:
                code, state = 1, "source_changed"
            (output / "source-check.json").write_text(json.dumps({
                "initial": provenance["source_sha256"], "final": final_digest,
                "unchanged": final_digest == provenance["source_sha256"]}, indent=2) + "\n")
            (output / "status.json").write_text(json.dumps({"state": state, "exit_code": code}, indent=2) + "\n")
        print(f"Experiment {state}: {output}", flush=True)
        return code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError, subprocess.CalledProcessError) as error:
        print(f"Batch not completed: {error}", file=sys.stderr)
        raise SystemExit(1)
