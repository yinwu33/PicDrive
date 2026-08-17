"""Reconstruct the simulator's ego observation vector from Waymo pose tracks.

The planning-loss term of :mod:`train_distillation` feeds the frozen teacher
head an ego vector alongside the scene feature, so distillation needs one 11-D
vector per processed frame.  ``preprocess`` keeps no ego pose -- the abstract
scene it stores is already expressed in the ego frame -- so this stage makes a
second pass over the raw TFRecords.

It decodes only ``timestamp_micros`` and ``pose``: no image is JPEG-decoded and
no lidar range image is touched, which is roughly twenty times cheaper than
``preprocess`` even though it reads the same bytes off disk.  A whole segment's
poses are needed at once because the goal is a point further along the driven
path, so this writes one table per segment rather than one file per frame.

Example:

    python -m data_utils.sim2real.waymo.extract_ego_state \
        --input /mnt/disk/data/public/waymo/perception_1_4_3/training \
        --output artifacts/waymo_sim2real/full/training/ego_state \
        --workers 8 --resume
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path

import numpy as np

from .ego_state import ego_observations
from .processed import (
    EGO_SCHEMA_VERSION,
    GOAL_TARGET_DISTANCE,
    WAYMO_SDC_LENGTH,
    WAYMO_SDC_WIDTH,
    atomic_savez,
    load_ego_state,
)
from .proto import ProtoDecodeError, first_bytes, first_int, iter_tfrecord, parse_transform


def _decode_text(value: bytes | None, fallback: str) -> str:
    return value.decode("utf-8", "replace") if value is not None else fallback


def process_segment(
    tfrecord: Path,
    output: Path,
    length: float,
    width: float,
    goal_distance: float,
    overwrite: bool,
    resume: bool,
) -> dict[str, int | str]:
    segment_id: str | None = None
    timestamps: list[int] = []
    poses: list[np.ndarray] = []

    for frame_index, frame in enumerate(iter_tfrecord(tfrecord)):
        if segment_id is None:
            context = first_bytes(frame, 1)
            if context is None:
                raise ProtoDecodeError(f"{tfrecord}: first frame has no Context")
            segment_id = _decode_text(first_bytes(context, 1), "unknown_segment")
            destination = output / f"{segment_id}.npz"
            if destination.exists() and resume and not overwrite:
                state = load_ego_state(destination)
                saved = str(np.asarray(state["segment_id"]).item())
                if saved != segment_id:
                    raise ValueError(f"{destination} holds segment {saved}, expected {segment_id}")
                return {"file": destination.name, "segment_id": segment_id, "frames": len(state["ego_obs"])}
            if destination.exists() and not overwrite:
                raise FileExistsError(
                    f"{destination} exists; pass --resume to keep it or --overwrite to replace it"
                )
        pose = first_bytes(frame, 3)
        if pose is None:
            raise ProtoDecodeError(f"{tfrecord}: frame {frame_index} has no pose")
        poses.append(parse_transform(pose))
        timestamps.append(int(first_int(frame, 2, -1)))

    if segment_id is None:
        raise ProtoDecodeError(f"{tfrecord}: contains no frames")
    stamps = np.asarray(timestamps, dtype=np.int64)
    obs = ego_observations(
        np.stack(poses), stamps, length=length, width=width, goal_distance=goal_distance
    )
    destination = output / f"{segment_id}.npz"
    atomic_savez(
        destination,
        schema_version=np.asarray(EGO_SCHEMA_VERSION, dtype=np.int32),
        segment_id=np.asarray(segment_id),
        timestamp_micros=stamps,
        ego_obs=obs,
        vehicle_length=np.asarray(length, dtype=np.float32),
        vehicle_width=np.asarray(width, dtype=np.float32),
        goal_distance=np.asarray(goal_distance, dtype=np.float32),
    )
    return {"file": destination.name, "segment_id": segment_id, "frames": len(obs)}


def _input_files(path: Path, max_segments: int | None) -> list[Path]:
    files = [path] if path.is_file() else sorted(path.glob("*.tfrecord"))
    if max_segments is not None:
        files = files[:max_segments]
    if not files:
        raise FileNotFoundError(f"no .tfrecord files found under {path}")
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Raw Waymo TFRecord file or directory")
    parser.add_argument("--output", type=Path, required=True, help="Ego-state directory")
    parser.add_argument("--vehicle-length", type=float, default=WAYMO_SDC_LENGTH)
    parser.add_argument("--vehicle-width", type=float, default=WAYMO_SDC_WIDTH)
    parser.add_argument(
        "--goal-distance",
        type=float,
        default=GOAL_TARGET_DISTANCE,
        help="Must match the simulator's goal_target_distance",
    )
    parser.add_argument("--max-segments", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Keep validated existing segments")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive")
    if args.goal_distance <= 0:
        parser.error("--goal-distance must be positive")

    args.output.mkdir(parents=True, exist_ok=True)
    files = _input_files(args.input, args.max_segments)
    worker_args = [
        (
            tfrecord,
            args.output,
            args.vehicle_length,
            args.vehicle_width,
            args.goal_distance,
            args.overwrite,
            args.resume,
        )
        for tfrecord in files
    ]
    entries: list[dict[str, int | str]] = []
    if args.workers == 1:
        results = (process_segment(*item) for item in worker_args)
    else:
        executor = ProcessPoolExecutor(max_workers=args.workers)
        results = executor.map(process_segment, *zip(*worker_args))
    try:
        for index, (tfrecord, entry) in enumerate(zip(files, results), 1):
            entries.append(entry)
            print(f"[{index}/{len(files)}] {tfrecord.name}: {entry['frames']} ego rows", flush=True)
    finally:
        if args.workers != 1:
            executor.shutdown()

    entries.sort(key=lambda entry: entry["segment_id"])
    metadata = {
        "schema_version": EGO_SCHEMA_VERSION,
        "num_segments": len(entries),
        "num_frames": int(sum(int(entry["frames"]) for entry in entries)),
        "vehicle_length": args.vehicle_length,
        "vehicle_width": args.vehicle_width,
        "goal_distance": args.goal_distance,
        "segments": entries,
    }
    (args.output / "manifest.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(
        f"reconstructed ego state for {len(entries)} segments "
        f"({metadata['num_frames']} frames) into {args.output}"
    )


if __name__ == "__main__":
    main()
