#!/usr/bin/env python3
"""Compare identical completed episode prefixes; partial reports stay explicit."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path


def compare(baseline, method):
    directories = [Path(baseline), Path(method)]
    specs = [json.loads((p / "experiment.json").read_text()) for p in directories]
    if specs[0].get("method") != "baseline":
        raise ValueError("First run must be baseline")
    for field in ("dataset", "episodes", "seed", "video", "max_steps", "llm_mode"):
        if specs[0].get(field) != specs[1].get(field):
            raise ValueError(f"Experiment setting differs: {field}")
    provenance = [json.loads((p / "provenance.json").read_text()) for p in directories]
    if provenance[0]["source_sha256"] != provenance[1]["source_sha256"]:
        raise ValueError("Runtime source differs; this is not a controlled comparison")
    rows = []
    for directory in directories:
        with (directory / "episodes.csv").open(newline="") as stream:
            rows.append(list(csv.DictReader(stream)))
    count = min(map(len, rows))
    if count == 0:
        raise ValueError("No paired completed episodes")
    for index, (left, right) in enumerate(zip(*rows), 1):
        if any(left[k] != right[k] for k in ("scene_id", "episode_id", "target")):
            raise ValueError(f"Episode identities differ at {index}")
        if any(int(row["episode_number"]) != index for row in (left, right)):
            raise ValueError("Episode sequence is not contiguous")
    metrics = ("success", "spl", "soft_spl", "distance_to_goal", "steps", "elapsed_seconds")
    averages = [{k: sum(float(row[k]) for row in run[:count]) / count for k in metrics}
                for run in rows]
    return {
        "paired_episodes": count, "requested_episodes": specs[0]["episodes"],
        "complete": all(len(run) == specs[0]["episodes"] for run in rows),
        "diagnostic_step_cap": specs[0].get("max_steps"),
        "method": specs[1]["method"], "review_settings": specs[1].get("review"),
        "baseline": averages[0], "treatment": averages[1],
        "delta_treatment_minus_baseline": {k: averages[1][k] - averages[0][k] for k in metrics},
        "result_counts": [dict(Counter(row["result"] for row in run[:count])) for run in rows],
        "changed_outcomes": [
            {"scene": a["scene_id"], "episode": a["episode_id"],
             "baseline": a["result"], "treatment": b["result"]}
            for a, b in zip(*rows) if a["result"] != b["result"]],
        "code_shas": [p["code_sha"] for p in provenance],
        "paths": [str(p.resolve()) for p in directories],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline")
    parser.add_argument("method")
    args = parser.parse_args()
    print(json.dumps(compare(args.baseline, args.method), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
