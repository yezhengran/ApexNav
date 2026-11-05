#!/usr/bin/env python3
"""Profile the installed vision models one at a time, without HTTP services."""

import argparse
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "outputs/runtime/model-profile"
MODELS = ("sam", "yolo", "dino", "blip2")


def gpu_used():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=5,
    )
    return int(result.stdout.splitlines()[0].strip())


def worker(name):
    sys.path.insert(0, str(ROOT))
    baseline = gpu_used()
    samples = [baseline]
    stopped = threading.Event()

    def sample():
        while not stopped.wait(0.5):
            try:
                samples.append(gpu_used())
            except (OSError, ValueError, subprocess.SubprocessError):
                pass

    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    try:
        import cv2
        import numpy as np
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for this profile")
        image = cv2.imread(str(ROOT / "outputs/runtime/smoke-rgb.png"))
        if image is None:
            raise RuntimeError("Run the Habitat smoke test to create its RGB frame first")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        if name == "sam":
            from vlm.segmentor.sam import MobileSAM
            model = MobileSAM(str(ROOT / "data/mobile_sam.pt"))
            predict = lambda: model.segment_bbox(image, [160, 120, 480, 400])
        elif name == "yolo":
            from vlm.detector.yolov7 import YOLOv7
            model = YOLOv7(str(ROOT / "data/yolov7-e6e.pt"))
            predict = lambda: model.predict(image)
        elif name == "dino":
            from vlm.detector.grounding_dino import GroundingDINO
            model = GroundingDINO()
            predict = lambda: model.predict(image, caption="bed . chair . plant .")
        else:
            from vlm.itm.blip2itm import BLIP2ITM
            model = BLIP2ITM()
            predict = lambda: (model.cosine(image, "a bed in a bedroom"),
                               model.itm_scores(image, "a bed in a bedroom"))
        torch.cuda.synchronize()
        loaded = time.monotonic()
        load_gpu = gpu_used()
        load_allocated = torch.cuda.memory_allocated() / 2**20
        load_reserved = torch.cuda.memory_reserved() / 2**20
        timings = []
        for _ in range(3):
            tick = time.monotonic()
            output = predict()
            torch.cuda.synchronize()
            timings.append(round(time.monotonic() - tick, 3))
        if name == "blip2" and not np.isfinite(output).all():
            raise RuntimeError("BLIP2 returned a non-finite score")
        allocated_after_inference = torch.cuda.memory_allocated() / 2**20
        torch.cuda.empty_cache()
        gpu_after_cache_release = gpu_used()
        report = {
            "model": name, "image_shape": list(image.shape),
            "load_seconds": round(loaded - started, 3),
            "inference_seconds": timings,
            "gpu_total_baseline_mib": baseline,
            "gpu_total_after_load_mib": load_gpu,
            "gpu_total_after_cache_release_mib": gpu_after_cache_release,
            "gpu_total_peak_sampled_mib": max(samples + [gpu_used()]),
            "torch_reserved_after_load_mib": round(load_reserved, 1),
            "torch_allocated_after_load_mib": round(load_allocated, 1),
            "torch_allocated_after_inference_mib": round(allocated_after_inference, 1),
            "torch_peak_allocated_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
            "torch_peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1),
            "host_process_peak_rss_mib": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
            "note": "Isolated 640x480 smoke-frame test, not a worst-case navigation benchmark. GPU totals include desktop/other apps; do not add these totals across models.",
        }
        (REPORTS / f"{name}.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report), flush=True)
    finally:
        stopped.set()
        monitor.join(timeout=6)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", choices=MODELS, help=argparse.SUPPRESS)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    args = parser.parse_args()
    os.chdir(ROOT)
    REPORTS.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(args.worker)
        return
    environment = dict(os.environ, PYTHONUNBUFFERED="1", OMP_NUM_THREADS="2",
                       MKL_NUM_THREADS="2", MALLOC_ARENA_MAX="2",
                       TOKENIZERS_PARALLELISM="false", HF_HUB_OFFLINE="1",
                       TRANSFORMERS_OFFLINE="1")
    reports = []
    for name in args.models:
        log_path = REPORTS / f"{name}.log"
        print(f"[PROFILING] {name} log={log_path}", flush=True)
        with log_path.open("w") as log:
            result = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--worker", name],
                env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=360,
            )
        if result.returncode:
            raise RuntimeError(f"{name} failed ({result.returncode}); inspect {log_path}")
        report = json.loads((REPORTS / f"{name}.json").read_text())
        reports.append(report)
        print(json.dumps(report), flush=True)
    (REPORTS / "summary.json").write_text(json.dumps(reports, indent=2) + "\n")
    print("LOCAL_MODEL_PROFILE_OK; all worker processes exited", flush=True)


if __name__ == "__main__":
    main()
