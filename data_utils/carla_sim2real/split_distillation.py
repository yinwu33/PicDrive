"""Create leak-free CARLA distillation splits from a collected artifact tree.

The source collector may write one flat split for convenient capture. This tool
assigns whole segments to training or validation, stratified by town, and
hardlinks the large processed NPZ files so no camera data is duplicated.

Example:
    python -m data_utils.carla_sim2real.split_distillation \
      --source artifacts/carla_sim2real/sample1k/training \
      --output artifacts/carla_sim2real/sample1k_dino \
      --train-per-town 27 --validation-per-town 7 --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path


def _town_seed(seed: int, town: str) -> int:
    digest = hashlib.sha256(f"{seed}:{town}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def split_segments(
    episodes: list[dict[str, object]],
    train_per_town: int,
    validation_per_town: int,
    seed: int,
) -> dict[str, set[str]]:
    """Return deterministic, town-stratified sets of complete segment ids."""

    by_town: dict[str, list[str]] = {}
    for episode in episodes:
        by_town.setdefault(str(episode["town"]), []).append(str(episode["segment_id"]))
    result = {"training": set(), "validation": set()}
    required = train_per_town + validation_per_town
    for town, segments in sorted(by_town.items()):
        if len(segments) < required:
            raise ValueError(f"{town} has {len(segments)} segments, need {required}")
        shuffled = sorted(segments)
        random.Random(_town_seed(seed, town)).shuffle(shuffled)
        result["training"].update(shuffled[:train_per_town])
        result["validation"].update(shuffled[train_per_town:required])
    if result["training"] & result["validation"]:
        raise AssertionError("training and validation segment sets overlap")
    return result


def _link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.link(source, destination)


def build_split(
    source: Path,
    output: Path,
    train_per_town: int,
    validation_per_town: int,
    seed: int,
) -> dict[str, dict[str, int]]:
    if output.exists():
        raise FileExistsError(f"{output} already exists; choose a new output directory")
    episode_manifest = source / "episodes.jsonl"
    processed_manifest = source / "processed" / "manifest.jsonl"
    if not episode_manifest.is_file() or not processed_manifest.is_file():
        raise FileNotFoundError(f"{source} is missing episodes.jsonl or processed/manifest.jsonl")
    episodes = [json.loads(line) for line in episode_manifest.read_text().splitlines() if line]
    processed = [json.loads(line) for line in processed_manifest.read_text().splitlines() if line]
    assignment = split_segments(episodes, train_per_town, validation_per_town, seed)
    episode_by_id = {str(row["segment_id"]): row for row in episodes}
    if len(episode_by_id) != len(episodes):
        raise ValueError("episodes.jsonl contains duplicate segment ids")
    split_for_segment = {
        segment: split_name for split_name, segments in assignment.items() for segment in segments
    }
    counts: dict[str, dict[str, int]] = {}
    output.mkdir(parents=True)
    for split_name, segments in assignment.items():
        split_root = output / split_name
        (split_root / "processed").mkdir(parents=True)
        (split_root / "ego_state").mkdir()
        selected_episodes = [episode_by_id[segment] for segment in sorted(segments)]
        (split_root / "episodes.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in selected_episodes)
        )
        counts[split_name] = {"segments": len(segments), "frames": 0}

    manifests: dict[str, list[dict[str, object]]] = {"training": [], "validation": []}
    for entry in processed:
        segment = str(entry["segment_id"])
        split_name = split_for_segment.get(segment)
        if split_name is None:
            continue
        name = str(entry["file"])
        _link(source / "processed" / name, output / split_name / "processed" / name)
        manifests[split_name].append(entry)
        counts[split_name]["frames"] += 1
    for split_name, rows in manifests.items():
        (output / split_name / "processed" / "manifest.jsonl").write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
        )
        for segment in sorted(assignment[split_name]):
            _link(
                source / "ego_state" / f"{segment}.npz",
                output / split_name / "ego_state" / f"{segment}.npz",
            )
        expected = sum(int(episode_by_id[segment]["frames"]) for segment in assignment[split_name])
        if counts[split_name]["frames"] != expected:
            raise ValueError(
                f"{split_name} has {counts[split_name]['frames']} processed frames, expected {expected}"
            )

    metadata = {
        "schema_version": 1,
        "source": str(source.resolve()),
        "seed": seed,
        "train_per_town": train_per_town,
        "validation_per_town": validation_per_town,
        "counts": counts,
    }
    (output / "split.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return counts


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-per-town", type=int, default=27)
    parser.add_argument("--validation-per-town", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    if args.train_per_town < 1 or args.validation_per_town < 1:
        parser.error("split sizes must be positive")
    counts = build_split(
        args.source,
        args.output,
        args.train_per_town,
        args.validation_per_town,
        args.seed,
    )
    print(json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
