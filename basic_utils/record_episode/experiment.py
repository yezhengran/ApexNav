"""Small, machine-readable experiment records and constant-memory video output."""

import csv
from collections import Counter
import json
import os
from pathlib import Path
import tempfile

FIELDS = [
    "episode_number", "scene_id", "episode_id", "target", "steps", "success",
    "spl", "soft_spl", "distance_to_goal", "distance_to_goal_reward", "result",
    "elapsed_seconds", "cumulative_seconds", "video_path",
]


def episode_plan(available, completed=0, limit=-1, single=-1):
    if limit == 0 or limit < -1:
        raise ValueError("Episode limit must be -1 or positive")
    if single != -1:
        if not 0 <= single < available:
            raise ValueError("Single episode index is outside the dataset")
        return 1, single, 1
    target = available if limit == -1 else min(available, limit)
    if completed < 0 or completed > available:
        raise ValueError("Invalid completed episode count")
    return target, completed, max(0, target - completed)


def read_rows(directory):
    path = Path(directory) / "episodes.csv"
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    for index, row in enumerate(rows, 1):
        if int(row["episode_number"]) != index:
            raise ValueError("Episode CSV is not a contiguous completed sequence")
    return rows


def resume_state(directory):
    if not (Path(directory) / "episodes.csv").exists():
        return None
    rows = read_rows(directory)
    return (
        len(rows), sum(int(row["success"]) for row in rows),
        sum(float(row["spl"]) for row in rows),
        sum(float(row["soft_spl"]) for row in rows),
        sum(float(row["distance_to_goal"]) for row in rows),
        sum(float(row["distance_to_goal_reward"]) for row in rows),
        float(rows[-1]["cumulative_seconds"]) if rows else 0.0,
    )


def save_result(directory, row, requested_episodes):
    directory = Path(directory)
    rows = read_rows(directory)
    if int(row["episode_number"]) != len(rows) + 1:
        raise ValueError("Refusing to overwrite or skip completed episode results")
    rows.append(row)
    with tempfile.NamedTemporaryFile(mode="w", dir=directory, newline="",
                                     encoding="utf-8", delete=False) as stream:
        temporary = Path(stream.name)
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.chmod(0o644)
    os.replace(temporary, directory / "episodes.csv")
    count = len(rows)
    summary = {
        "completed_episodes": count, "requested_episodes": requested_episodes,
        "successful_episodes": sum(int(r["success"]) for r in rows),
        "success_rate": sum(int(r["success"]) for r in rows) / count,
        "mean_spl": sum(float(r["spl"]) for r in rows) / count,
        "mean_soft_spl": sum(float(r["soft_spl"]) for r in rows) / count,
        "mean_distance_to_goal_m": sum(float(r["distance_to_goal"]) for r in rows) / count,
        "elapsed_seconds": float(rows[-1]["cumulative_seconds"]),
        "scenes_completed": len({r["scene_id"] for r in rows}),
        "mean_steps": sum(int(r["steps"]) for r in rows) / count,
        "mean_episode_seconds": sum(float(r["elapsed_seconds"]) for r in rows) / count,
        "result_counts": dict(Counter(r["result"] for r in rows)),
        "complete": count >= requested_episodes,
    }
    with tempfile.NamedTemporaryFile(mode="w", dir=directory, encoding="utf-8",
                                     delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(summary, stream, indent=2)
        stream.write("\n")
    temporary.chmod(0o644)
    os.replace(temporary, directory / "summary.json")
    if count % 10 == 0 or count == requested_episodes:
        (directory / f"checkpoint-{count:04d}.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def record_episode_identity(directory, number, scene, episode, target):
    """Append actual iterator order before execution; verify identity on retry."""
    path = Path(directory) / "episode_set.json"
    entries = json.loads(path.read_text()) if path.exists() else []
    entry = {"scene_id": scene, "episode_id": str(episode), "target": target}
    if number <= len(entries):
        if entries[number - 1] != entry:
            raise ValueError("Episode order changed; refusing to mix a different evaluation set")
        return
    if number != len(entries) + 1:
        raise ValueError("Episode manifest is not contiguous; start a new experiment")
    entries.append(entry)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, encoding="utf-8",
                                     delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(entries, stream, indent=2)
        stream.write("\n")
    temporary.chmod(0o644)
    os.replace(temporary, path)


class EpisodeVideo:
    """Encode each frame immediately, rather than retaining an episode in RAM."""

    def __init__(self, directory, number):
        import imageio.v2 as imageio

        directory = Path(directory)
        self.temporary = directory / f"episode-{number:04d}.partial.mp4"
        self.writer = imageio.get_writer(str(self.temporary), fps=6, quality=8)

    def append(self, frame):
        import cv2

        if frame.shape[1] > 960:
            height = max(16, round(frame.shape[0] * 960 / frame.shape[1] / 16) * 16)
            frame = cv2.resize(frame, (960, height), interpolation=cv2.INTER_AREA)
        self.writer.append_data(frame)

    def finish(self, directory, name):
        self.writer.close()
        destination = Path(directory) / f"{name}.mp4"
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(self.temporary, destination)
        return str(destination)
