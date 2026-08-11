"""Verify a complete Waymo sim-to-real artifact tree."""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
import json
import os
from pathlib import Path
from typing import Callable, TypeVar

import numpy as np
from PIL import Image

from .processed import load_feature, load_processed


T = TypeVar("T")


def _bounded_check(
    paths: list[Path], check: Callable[[Path], T], workers: int, label: str
) -> None:
    with ThreadPoolExecutor(max_workers=workers) as executor:
        iterator = iter(paths)
        pending: deque[tuple[Path, Future[T]]] = deque()

        def submit_one() -> bool:
            try:
                path = next(iterator)
            except StopIteration:
                return False
            pending.append((path, executor.submit(check, path)))
            return True

        for _ in range(workers * 4):
            if not submit_one():
                break
        checked = 0
        while pending:
            path, future = pending.popleft()
            try:
                future.result()
            except Exception as exc:
                raise RuntimeError(f"{label} validation failed for {path}") from exc
            checked += 1
            if checked % 5000 == 0 or checked == len(paths):
                print(f"{label}: {checked}/{len(paths)}", flush=True)
            submit_one()


def _check_feature(path: Path, checkpoint_hash: str) -> None:
    feature = load_feature(path)
    segment_id = str(np.asarray(feature["segment_id"]).item())
    timestamp = int(np.asarray(feature["timestamp_micros"]).item())
    if path.stem != f"{segment_id}__{timestamp}":
        raise ValueError("filename and embedded feature identity differ")
    if str(np.asarray(feature["checkpoint_sha256"]).item()) != checkpoint_hash:
        raise ValueError("checkpoint hash differs")
    if not np.isfinite(feature["teacher_feature"]).all():
        raise ValueError("teacher feature contains NaN or infinity")


def _check_png_header(path: Path) -> None:
    with Image.open(path) as image:
        if image.format != "PNG" or image.mode != "RGB" or image.size != (1264, 832):
            raise ValueError(
                f"expected 1264x832 RGB PNG, got {image.format} {image.mode} {image.size}"
            )


def _check_processed(path: Path) -> None:
    sample = load_processed(path)
    for key in ("agents", "roads", "ego", "rig"):
        if not np.isfinite(sample[key]).all():
            raise ValueError(f"{key} contains NaN or infinity")


def verify_split(split_root: Path, workers: int) -> dict[str, int]:
    processed_dir = split_root / "processed"
    features_dir = split_root / "teacher_features"
    png_dir = split_root / "png"
    processed_entries = [
        json.loads(line)
        for line in (processed_dir / "manifest.jsonl").read_text().splitlines()
        if line.strip()
    ]
    feature_manifest = json.loads((features_dir / "manifest.json").read_text())
    expected = len(processed_entries)
    if feature_manifest["num_samples"] != expected or len(feature_manifest["samples"]) != expected:
        raise ValueError("processed and feature manifest counts differ")
    checkpoint_hash = feature_manifest["checkpoint_sha256"]

    processed_names = [entry["file"] for entry in processed_entries]
    feature_names = [entry["file"] for entry in feature_manifest["samples"]]
    feature_sources = [entry["processed_file"] for entry in feature_manifest["samples"]]
    if len(set(processed_names)) != expected:
        raise ValueError("processed manifest contains duplicate files")
    if set(feature_names) != set(processed_names) or set(feature_sources) != set(processed_names):
        raise ValueError("feature manifest is not a one-to-one mapping of processed files")

    processed_files = sorted(processed_dir.glob("*.npz"))
    feature_files = sorted(features_dir.glob("*.npz"))
    png_files = sorted(png_dir.glob("*/*.png"))
    if not (len(processed_files) == len(feature_files) == len(png_files) == expected):
        raise ValueError(
            "artifact counts differ: "
            f"processed={len(processed_files)} feature={len(feature_files)} "
            f"png={len(png_files)} expected={expected}"
        )
    expected_png = {
        f"{Path(name).stem.rsplit('__', 1)[0]}/{Path(name).stem}.png"
        for name in processed_names
    }
    actual_png = {path.relative_to(png_dir).as_posix() for path in png_files}
    if actual_png != expected_png:
        raise ValueError("PNG paths are not a one-to-one mapping of processed files")

    _bounded_check(
        feature_files,
        lambda path: _check_feature(path, checkpoint_hash),
        workers,
        f"{split_root.name} features",
    )
    _bounded_check(png_files, _check_png_header, workers, f"{split_root.name} PNG headers")

    # Fully decode one processed sample and one PNG from every segment. This
    # complements the all-file structural checks without re-decoding all RGB.
    representative_processed: dict[str, Path] = {}
    representative_png: dict[str, Path] = {}
    for entry in processed_entries:
        segment = entry["segment_id"]
        representative_processed.setdefault(segment, processed_dir / entry["file"])
        representative_png.setdefault(
            segment, png_dir / segment / f"{Path(entry['file']).stem}.png"
        )
    _bounded_check(
        list(representative_processed.values()),
        _check_processed,
        workers,
        f"{split_root.name} representative processed",
    )

    def decode_png(path: Path) -> None:
        with Image.open(path) as image:
            image.load()

    _bounded_check(
        list(representative_png.values()),
        decode_png,
        workers,
        f"{split_root.name} representative PNG decode",
    )
    return {"frames": expected, "segments": len(representative_processed)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=("training", "validation"))
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    summaries = {
        split: verify_split(args.root / split, args.workers) for split in args.splits
    }
    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
