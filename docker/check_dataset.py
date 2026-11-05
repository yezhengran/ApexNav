#!/usr/bin/env python3
"""Check task episodes, referenced scenes, navmeshes, and semantic assets."""

import argparse
import gzip
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {"hm3dv1": "hm3d/v1", "hm3dv2": "hm3d/v2", "mp3d": "mp3d/v1"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, default="hm3dv2")
    parser.add_argument("--geometry-only", action="store_true")
    args = parser.parse_args()
    root = ROOT / "data/datasets/objectnav" / DATASETS[args.dataset] / "val"
    metadata = root / "val.json.gz"
    if not metadata.is_file():
        print(f"[MISSING] Task metadata: {metadata}")
        return 1
    with gzip.open(metadata, "rt") as stream:
        episodes = json.load(stream).get("episodes", [])
    for content in sorted((root / "content").glob("*.json.gz")):
        with gzip.open(content, "rt") as stream:
            episodes.extend(json.load(stream).get("episodes", []))
    if not episodes:
        print("[MISSING] No validation episodes found")
        return 1
    scene_paths = {ROOT / "data/scene_datasets" / e["scene_id"] for e in episodes}
    geometry = scene_paths | {p.with_suffix(".navmesh") for p in scene_paths}
    semantics = set()
    for episode in episodes:
        config = episode.get("scene_dataset_config")
        if config:
            semantics.add(ROOT / config)
    for scene in scene_paths:
        if scene.name.endswith(".basis.glb"):
            semantics.add(scene.with_name(scene.name.replace(".basis.glb", ".semantic.glb")))
            semantics.add(scene.with_name(scene.name.replace(".basis.glb", ".semantic.txt")))
    missing_geometry = sorted(p for p in geometry if not p.is_file())
    missing_semantics = sorted(p for p in semantics if not p.is_file())
    print(f"Episodes={len(episodes)} Scenes={len(scene_paths)}")
    print(f"Missing geometry/navmesh={len(missing_geometry)}")
    print(f"Missing semantic/config={len(missing_semantics)}")
    missing = missing_geometry + ([] if args.geometry_only else missing_semantics)
    for path in missing[:6]:
        print(f"[MISSING] {path.relative_to(ROOT)}")
    if missing:
        return 1
    print("GEOMETRY_READY (not a complete evaluation dataset)" if args.geometry_only else "OBJECTNAV_DATASET_READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
