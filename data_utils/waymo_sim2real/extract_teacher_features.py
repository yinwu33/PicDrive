"""Render processed Waymo scenes and extract frozen DriveCam teacher features.

This stage never opens TFRecord files. It consumes only compact samples emitted
by :mod:`data_utils.waymo_sim2real.preprocess`, renders their abstract scene with
the existing CUDA rasterizer, and stores the frozen teacher's 256-D scene
embedding plus the three rendered views needed for audit visualization.

Example:

    python -m data_utils.waymo_sim2real.extract_teacher_features \
        --processed artifacts/waymo_sim2real/processed \
        --checkpoint experiments/puffer_drive_cam_gwvaxkmh/model_puffer_drive_cam_007800.pt \
        --output artifacts/waymo_sim2real/teacher_features
"""

from __future__ import annotations

import argparse
import json
import os
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from pufferlib.ocean.drive import raster_cuda
from pufferlib.ocean.drive.raster_ref import WAYMO_RIG
from pufferlib.ocean.torch import DriveCam

from .processed import (
    CAMERA_NAMES,
    FEATURE_SCHEMA_VERSION,
    TEACHER_FEATURE_DIM,
    atomic_savez,
    list_processed_files,
    load_feature,
    load_render_input,
)
from .render_roads import prepare_runtime_roads
from .teacher import load_teacher, scene_features, sha256_file


def _prefix_ranges(chunks: list[np.ndarray], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    offsets = [0]
    for chunk in chunks:
        offsets.append(offsets[-1] + len(chunk))
    if offsets[-1]:
        values = torch.from_numpy(np.ascontiguousarray(np.concatenate(chunks, axis=0))).to(device)
    else:
        width = chunks[0].shape[1]
        values = torch.empty((0, width), dtype=torch.float32, device=device)
    ranges = torch.tensor(offsets, dtype=torch.int32, device=device)
    return values, ranges


def _prefetched_samples(files: list[Path], workers: int) -> tuple[Path, dict[str, np.ndarray]]:
    """Yield ordered render inputs with a small, bounded read-ahead window."""
    if workers == 1:
        for source in files:
            yield source, load_render_input(source)
        return

    with ThreadPoolExecutor(max_workers=workers) as executor:
        iterator = iter(files)
        pending: deque[tuple[Path, Future[dict[str, np.ndarray]]]] = deque()

        def submit_one() -> bool:
            try:
                source = next(iterator)
            except StopIteration:
                return False
            pending.append((source, executor.submit(load_render_input, source)))
            return True

        for _ in range(workers * 2):
            if not submit_one():
                break
        while pending:
            source, future = pending.popleft()
            yield source, future.result()
            submit_one()


@torch.inference_mode()
def _extract_batch(
    teacher: DriveCam, samples: list[tuple[Path, dict[str, np.ndarray]]], device: torch.device
) -> tuple[np.ndarray, np.ndarray]:
    rigs = [sample["rig"] for _, sample in samples]
    if not all(np.allclose(rig, rigs[0], atol=1e-5, rtol=1e-5) for rig in rigs[1:]):
        raise ValueError("a render batch crossed camera calibrations; group samples by segment")
    agents, agent_ranges = _prefix_ranges([sample["agents"] for _, sample in samples], device)
    roads, road_ranges = _prefix_ranges([prepare_runtime_roads(sample["roads"]) for _, sample in samples], device)
    egos = torch.from_numpy(np.stack([sample["ego"] for _, sample in samples])).to(device)
    ego_scene = torch.arange(len(samples), dtype=torch.int32, device=device)
    rig = torch.from_numpy(np.ascontiguousarray(rigs[0])).to(device)

    images = raster_cuda.render(
        agents=agents,
        roads=roads,
        egos=egos,
        cameras=WAYMO_RIG,
        rig=rig,
        ego_scene=ego_scene,
        agent_ranges=agent_ranges,
        road_ranges=road_ranges,
    )
    features = scene_features(teacher, images)
    sim_rgb = images.permute(0, 1, 3, 4, 2).contiguous().cpu().numpy()
    return features.float().cpu().numpy(), sim_rgb


def _flush_batch(
    teacher: DriveCam,
    pending: list[tuple[Path, dict[str, np.ndarray]]],
    output: Path,
    checkpoint_hash: str,
    device: torch.device,
    overwrite: bool,
) -> list[dict[str, int | str]]:
    if not pending:
        return []
    features, sim_images = _extract_batch(teacher, pending, device)
    entries = []
    for index, (source, sample) in enumerate(pending):
        destination = output / source.name
        if destination.exists() and not overwrite:
            raise FileExistsError(f"{destination} exists; pass --resume to keep it or --overwrite to replace it")
        segment_id = str(np.asarray(sample["segment_id"]).item())
        timestamp = int(np.asarray(sample["timestamp_micros"]).item())
        atomic_savez(
            destination,
            schema_version=np.asarray(FEATURE_SCHEMA_VERSION, dtype=np.int32),
            segment_id=np.asarray(segment_id),
            timestamp_micros=np.asarray(timestamp, dtype=np.int64),
            camera_names=np.asarray(CAMERA_NAMES),
            teacher_feature=features[index],
            sim_images=sim_images[index],
            checkpoint_sha256=np.asarray(checkpoint_hash),
        )
        entries.append(
            {
                "file": destination.name,
                "processed_file": source.name,
                "segment_id": segment_id,
                "timestamp_micros": timestamp,
            }
        )
    return entries


@torch.inference_mode()
def _flush_cached_render_batch(
    teacher: DriveCam,
    sources: list[Path],
    cached_features: Path,
    output: Path,
    checkpoint_hash: str,
    device: torch.device,
    overwrite: bool,
) -> list[dict[str, int | str]]:
    """Recompute targets from audited sim renders saved by an older teacher.

    Rendering is checkpoint-independent. Reusing the saved uint8 renders is
    therefore exact and lets a target refresh run on CPU when CUDA is absent.
    The filename, segment, timestamp, camera order, shape and dtype are all
    validated before the new teacher sees a pixel.
    """

    cached = [load_feature(cached_features / source.name) for source in sources]
    images = torch.from_numpy(np.stack([row["sim_images"] for row in cached])).permute(0, 1, 4, 2, 3)
    features = scene_features(teacher, images.to(device)).float().cpu().numpy()
    entries: list[dict[str, int | str]] = []
    for index, (source, row) in enumerate(zip(sources, cached)):
        destination = output / source.name
        if destination.exists() and not overwrite:
            raise FileExistsError(f"{destination} exists; pass --resume to keep it or --overwrite to replace it")
        segment_id = str(np.asarray(row["segment_id"]).item())
        timestamp = int(np.asarray(row["timestamp_micros"]).item())
        expected_stem = f"{segment_id}__{timestamp}"
        if source.stem != expected_stem:
            raise ValueError(
                f"cached render identity {expected_stem} does not match processed file {source.name}"
            )
        atomic_savez(
            destination,
            schema_version=np.asarray(FEATURE_SCHEMA_VERSION, dtype=np.int32),
            segment_id=np.asarray(segment_id),
            timestamp_micros=np.asarray(timestamp, dtype=np.int64),
            camera_names=np.asarray(CAMERA_NAMES),
            teacher_feature=features[index],
            sim_images=row["sim_images"],
            checkpoint_sha256=np.asarray(checkpoint_hash),
        )
        entries.append(
            {
                "file": destination.name,
                "processed_file": source.name,
                "segment_id": segment_id,
                "timestamp_micros": timestamp,
            }
        )
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--loader-workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Bounded NPZ prefetch threads (default: min(8, CPU count))",
    )
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--reuse-sim-images",
        type=Path,
        help="Existing teacher_features directory whose audited sim_images are reused",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep validated features made with the same checkpoint",
    )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if args.loader_workers < 1:
        parser.error("--loader-workers must be >= 1")
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive")
    if not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    device = torch.device(args.device)
    if device.type != "cuda" and args.reuse_sim_images is None:
        parser.error("the production rasterizer is CUDA-only; use --device cuda")
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    if args.reuse_sim_images is not None and not args.reuse_sim_images.is_dir():
        parser.error(f"cached teacher features do not exist: {args.reuse_sim_images}")

    files = list_processed_files(args.processed, args.max_samples)
    if not files:
        parser.error(f"no processed .npz samples found under {args.processed}")
    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_hash = sha256_file(args.checkpoint)
    teacher = load_teacher(args.checkpoint, device)

    entries: list[dict[str, int | str]] = []
    todo: list[Path] = []
    for sample_index, source in enumerate(files, 1):
        destination = args.output / source.name
        if destination.exists() and args.resume:
            feature = load_feature(destination)
            saved_hash = str(np.asarray(feature["checkpoint_sha256"]).item())
            if saved_hash != checkpoint_hash:
                raise ValueError(f"{destination} uses checkpoint {saved_hash}, expected {checkpoint_hash}")
            segment_id = str(np.asarray(feature["segment_id"]).item())
            timestamp = int(np.asarray(feature["timestamp_micros"]).item())
            entries.append(
                {
                    "file": destination.name,
                    "processed_file": source.name,
                    "segment_id": segment_id,
                    "timestamp_micros": timestamp,
                }
            )
        elif destination.exists() and not args.overwrite:
            raise FileExistsError(f"{destination} exists; pass --resume to keep it or --overwrite to replace it")
        else:
            todo.append(source)
        if sample_index % 1000 == 0:
            print(
                f"[{sample_index}/{len(files)}] scanned; kept {len(entries)}, need {len(todo)}",
                flush=True,
            )

    kept = len(entries)
    if args.reuse_sim_images is not None:
        for start in range(0, len(todo), args.batch_size):
            batch = todo[start : start + args.batch_size]
            entries.extend(
                _flush_cached_render_batch(
                    teacher,
                    batch,
                    args.reuse_sim_images,
                    args.output,
                    checkpoint_hash,
                    device,
                    args.overwrite,
                )
            )
            if start % 500 == 0:
                print(f"[{kept + min(start + len(batch), len(todo))}/{len(files)}] refreshed features", flush=True)
    else:
        pending: list[tuple[Path, dict[str, np.ndarray]]] = []
        pending_segment: str | None = None
        for todo_index, (source, sample) in enumerate(_prefetched_samples(todo, args.loader_workers), 1):
            segment = str(np.asarray(sample["segment_id"]).item())
            if pending and (segment != pending_segment or len(pending) >= args.batch_size):
                entries.extend(_flush_batch(teacher, pending, args.output, checkpoint_hash, device, args.overwrite))
                pending = []
            pending.append((source, sample))
            pending_segment = segment
            if todo_index % 500 == 0:
                print(
                    f"[{kept + todo_index}/{len(files)}] kept/loaded features",
                    flush=True,
                )
        entries.extend(_flush_batch(teacher, pending, args.output, checkpoint_hash, device, args.overwrite))
    entries.sort(key=lambda entry: entry["processed_file"])

    metadata = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "num_samples": len(entries),
        "camera_names": list(CAMERA_NAMES),
        "feature_dim": TEACHER_FEATURE_DIM,
        "samples": entries,
    }
    (args.output / "manifest.json").write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"extracted {len(entries)} teacher features into {args.output}; checkpoint sha256={checkpoint_hash}")


if __name__ == "__main__":
    main()
