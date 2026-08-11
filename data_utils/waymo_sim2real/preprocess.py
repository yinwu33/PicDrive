"""Extract compact paired-scene inputs from Waymo Perception TFRecords.

This is the *only* sim-to-real stage allowed to read raw TFRecord files.  It
keeps just the three forward-facing RGB cameras, the scene primitives needed by
PufferDrive's abstract renderer, exact camera calibration, and frame identity.
Teacher extraction and visualization consume only the resulting ``.npz`` files.

Example (one segment, eight frames):

    python -m data_utils.waymo_sim2real.preprocess \
        --input /mnt/disk/data/public/waymo/perception_1_4_3/training \
        --output artifacts/waymo_sim2real/processed \
        --max-segments 1 --max-frames-per-segment 8
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from io import BytesIO
import json
import math
import os
from pathlib import Path

import numpy as np
from PIL import Image

from .processed import (
    CAMERA_ENUMS,
    CAMERA_NAMES,
    PROCESSED_SCHEMA_VERSION,
    RENDER_FAR,
    RENDER_NEAR,
    SIM_HEIGHT,
    SIM_WIDTH,
    WAYMO_REAR_AXLE_TO_BOX_CENTER,
    atomic_savez,
)
from .proto import (
    ProtoDecodeError,
    first_bytes,
    first_int,
    iter_fields,
    iter_tfrecord,
    parse_transform,
    parse_vector3d,
    repeated_bytes,
    repeated_doubles,
)


# PufferDrive RenderState type IDs.
VEHICLE = 1
PEDESTRIAN = 2
CYCLIST = 3
ROAD_LINE = 5
ROAD_EDGE = 6
CROSSWALK = 8
SPEED_BUMP = 9

LABEL_TYPE_TO_RENDER = {1: VEHICLE, 2: PEDESTRIAN, 4: CYCLIST}
MAP_FIELD_TO_RENDER = {4: ROAD_LINE, 5: ROAD_EDGE, 8: CROSSWALK, 9: SPEED_BUMP}
ROAD_WIDTH = {ROAD_LINE: 0.15, ROAD_EDGE: 0.25, CROSSWALK: 0.50, SPEED_BUMP: 0.40}

# Waymo's mounting rotation is expressed in an x-forward/y-left/z-up camera
# sensor frame. The CUDA rasterizer uses the CV x-right/y-down/z-forward frame.
SENSOR_TO_CV = np.asarray(
    [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]], dtype=np.float64
)


def _decode_text(value: bytes | None, fallback: str) -> str:
    return value.decode("utf-8", "replace") if value is not None else fallback


def _parse_context(context: bytes) -> tuple[str, dict[int, dict[str, np.ndarray | int]]]:
    segment_id = _decode_text(first_bytes(context, 1), "unknown_segment")
    calibrations: dict[int, dict[str, np.ndarray | int]] = {}
    for message in repeated_bytes(context, 2):
        name = first_int(message, 1)
        if name not in CAMERA_ENUMS:
            continue
        intrinsic = np.asarray(repeated_doubles(message, 2), dtype=np.float64)
        if intrinsic.size < 4:
            raise ProtoDecodeError(f"camera {name} has only {intrinsic.size} intrinsics")
        calibrations[int(name)] = {
            "intrinsic": intrinsic,
            "extrinsic": parse_transform(first_bytes(message, 3)),
            "width": int(first_int(message, 4, 0)),
            "height": int(first_int(message, 5, 0)),
        }
    missing = sorted(set(CAMERA_ENUMS) - calibrations.keys())
    if missing:
        raise ProtoDecodeError(f"segment {segment_id} lacks camera calibrations {missing}")
    return segment_id, calibrations


def _parse_images(frame: bytes, output_size: tuple[int, int]) -> np.ndarray:
    encoded: dict[int, bytes] = {}
    for message in repeated_bytes(frame, 4):
        name = first_int(message, 1)
        image = first_bytes(message, 2)
        if name in CAMERA_ENUMS and image is not None:
            encoded[int(name)] = image
    missing = sorted(set(CAMERA_ENUMS) - encoded.keys())
    if missing:
        raise ProtoDecodeError(f"frame lacks camera images {missing}")

    width, height = output_size
    images = []
    for name in CAMERA_ENUMS:
        with Image.open(BytesIO(encoded[name])) as image:
            image = image.convert("RGB")
            if image.size != (width, height):
                image = image.resize((width, height), Image.Resampling.BILINEAR)
            images.append(np.asarray(image, dtype=np.uint8))
    return np.stack(images)


def _parse_box(message: bytes) -> list[float] | None:
    values: dict[int, float] = {}
    for field in iter_fields(message):
        if field.wire_type == 1 and field.number in (1, 2, 3, 4, 5, 6, 7):
            values[field.number] = np.frombuffer(field.value, dtype="<f8", count=1)[0]
    needed = (1, 2, 4, 5, 6, 7)
    if any(field not in values for field in needed):
        return None
    return [values[field] for field in needed]


def _parse_agents(frame: bytes, radius: float) -> np.ndarray:
    rows: list[list[float]] = []
    for label in repeated_bytes(frame, 6):
        label_type = first_int(label, 3, 0)
        render_type = LABEL_TYPE_TO_RENDER.get(int(label_type))
        if render_type is None:
            continue
        box_data = first_bytes(label, 1)
        box = _parse_box(box_data) if box_data is not None else None
        if box is None:
            continue
        center_x, center_y, width, length, height, heading = box
        center_x -= WAYMO_REAR_AXLE_TO_BOX_CENTER
        if center_x * center_x + center_y * center_y > radius * radius:
            continue
        rows.append(
            [
                center_x,
                center_y,
                math.cos(heading),
                math.sin(heading),
                length,
                width,
                height,
                float(render_type),
            ]
        )
    if not rows:
        return np.zeros((0, 8), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


def _parse_point(message: bytes) -> np.ndarray | None:
    xyz = []
    for number in (1, 2, 3):
        value = repeated_doubles(message, number)
        xyz.append(value[0] if value else 0.0)
    if not np.isfinite(xyz).all():
        return None
    return np.asarray(xyz, dtype=np.float64)


def _parse_map_features(frame: bytes) -> list[tuple[int, np.ndarray]]:
    """Return global-frame polylines and polygons from the segment's first frame."""
    features: list[tuple[int, np.ndarray]] = []
    for map_feature in repeated_bytes(frame, 10):
        for field in iter_fields(map_feature):
            render_type = MAP_FIELD_TO_RENDER.get(field.number)
            if render_type is None or field.wire_type != 2:
                continue
            geometry = field.value
            point_field = 2 if field.number in (4, 5) else 1
            points = [
                point
                for point in (_parse_point(raw) for raw in repeated_bytes(geometry, point_field))
                if point is not None
            ]
            if len(points) < 2:
                continue
            polyline = np.stack(points)
            if field.number in (8, 9) and not np.allclose(polyline[0], polyline[-1]):
                polyline = np.concatenate([polyline, polyline[:1]], axis=0)
            features.append((render_type, polyline))
    return features


def _roads_in_ego_frame(
    features: list[tuple[int, np.ndarray]],
    vehicle_to_global: np.ndarray,
    map_pose_offset: np.ndarray,
    radius: float,
) -> np.ndarray:
    rotation = vehicle_to_global[:3, :3]
    translation = vehicle_to_global[:3, 3]
    rows: list[list[float]] = []
    for render_type, points_global in features:
        # Waymo specifies that map_pose_offset is added to transformed lidar
        # points to align them with map coordinates. Inverting that relation:
        # p_vehicle = R^T (p_map - offset - t).
        points_vehicle = (points_global - map_pose_offset - translation) @ rotation
        points_vehicle[:, 0] -= WAYMO_REAR_AXLE_TO_BOX_CENTER
        for p0, p1 in zip(points_vehicle[:-1], points_vehicle[1:]):
            # Keep segments intersecting the renderer's square far-range region.
            lo = np.minimum(p0[:2], p1[:2])
            hi = np.maximum(p0[:2], p1[:2])
            if (hi < -radius).any() or (lo > radius).any():
                continue
            rows.append([p0[0], p0[1], p1[0], p1[1], ROAD_WIDTH[render_type], render_type])
    if not rows:
        return np.zeros((0, 6), dtype=np.float32)
    return np.asarray(rows, dtype=np.float32)


def _calibration_arrays(
    calibrations: dict[int, dict[str, np.ndarray | int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rig_rows = []
    intrinsics, extrinsics, sizes = [], [], []
    for name in CAMERA_ENUMS:
        calibration = calibrations[name]
        intrinsic = np.asarray(calibration["intrinsic"], dtype=np.float64)
        extrinsic = np.asarray(calibration["extrinsic"], dtype=np.float64)
        source_width = int(calibration["width"])
        source_height = int(calibration["height"])
        if source_width <= 0 or source_height <= 0:
            raise ProtoDecodeError(f"camera {name} has invalid size {source_width}x{source_height}")

        # Camera sensor mounting frame -> vehicle is in the calibration. Convert
        # vehicle vectors into the rasterizer's CV camera convention.
        ego_to_camera = SENSOR_TO_CV @ extrinsic[:3, :3].T
        camera_position = extrinsic[:3, 3].copy()
        camera_position[0] -= WAYMO_REAR_AXLE_TO_BOX_CENTER
        scale_x, scale_y = SIM_WIDTH / source_width, SIM_HEIGHT / source_height
        fx, fy, cx, cy = intrinsic[:4]
        rig_rows.append(
            np.concatenate(
                [
                    ego_to_camera.reshape(-1),
                    camera_position,
                    np.asarray(
                        [
                            fx * scale_x,
                            fy * scale_y,
                            cx * scale_x,
                            cy * scale_y,
                            SIM_WIDTH,
                            SIM_HEIGHT,
                            RENDER_NEAR,
                            RENDER_FAR,
                        ]
                    ),
                ]
            )
        )
        padded = np.zeros(9, dtype=np.float64)
        padded[: min(9, intrinsic.size)] = intrinsic[:9]
        intrinsics.append(padded)
        extrinsics.append(extrinsic)
        sizes.append([source_height, source_width])
    return (
        np.asarray(rig_rows, dtype=np.float32),
        np.asarray(intrinsics, dtype=np.float64),
        np.asarray(extrinsics, dtype=np.float64),
        np.asarray(sizes, dtype=np.int32),
    )


def process_segment(
    tfrecord: Path,
    output: Path,
    real_size: tuple[int, int],
    frame_stride: int,
    max_frames: int | None,
    radius: float,
    overwrite: bool,
    resume: bool = False,
) -> list[dict[str, int | str]]:
    entries: list[dict[str, int | str]] = []
    segment_id: str | None = None
    calibrations = None
    map_features: list[tuple[int, np.ndarray]] | None = None
    rig = source_intrinsics = source_extrinsics = source_sizes = None
    emitted = 0

    for frame_index, frame in enumerate(iter_tfrecord(tfrecord)):
        if calibrations is None:
            context = first_bytes(frame, 1)
            if context is None:
                raise ProtoDecodeError(f"{tfrecord}: first frame has no Context")
            segment_id, calibrations = _parse_context(context)
            rig, source_intrinsics, source_extrinsics, source_sizes = _calibration_arrays(calibrations)
        if map_features is None:
            map_features = _parse_map_features(frame)
            if not map_features:
                raise ProtoDecodeError(f"{tfrecord}: first frame has no supported map geometry")
        if frame_index % frame_stride:
            continue
        if max_frames is not None and emitted >= max_frames:
            break

        timestamp = int(first_int(frame, 2, -1))
        assert segment_id is not None
        filename = f"{segment_id}__{timestamp}.npz"
        destination = output / filename
        if destination.exists() and resume and not overwrite:
            with np.load(destination, allow_pickle=False) as archive:
                saved_segment = str(np.asarray(archive["segment_id"]).item())
                saved_timestamp = int(np.asarray(archive["timestamp_micros"]).item())
                if (saved_segment, saved_timestamp) != (segment_id, timestamp):
                    raise ValueError(
                        f"{destination} identity is {saved_segment}@{saved_timestamp}, "
                        f"expected {segment_id}@{timestamp}"
                    )
                entries.append(
                    {
                        "file": filename,
                        "segment_id": segment_id,
                        "timestamp_micros": timestamp,
                        "frame_index": int(np.asarray(archive["frame_index"]).item()),
                        "agents": int(archive["agents"].shape[0]),
                        "roads": int(archive["roads"].shape[0]),
                    }
                )
            emitted += 1
            continue

        pose = parse_transform(first_bytes(frame, 3))
        map_offset = parse_vector3d(first_bytes(frame, 11))
        real_images = _parse_images(frame, real_size)
        agents = _parse_agents(frame, radius)
        roads = _roads_in_ego_frame(map_features, pose, map_offset, radius)
        ego = np.asarray([0.0, 0.0, 1.0, 0.0, -1.0], dtype=np.float32)

        if destination.exists() and not overwrite:
            raise FileExistsError(
                f"{destination} exists; pass --resume to keep it or --overwrite to replace it"
            )
        atomic_savez(
            destination,
            schema_version=np.asarray(PROCESSED_SCHEMA_VERSION, dtype=np.int32),
            segment_id=np.asarray(segment_id),
            timestamp_micros=np.asarray(timestamp, dtype=np.int64),
            frame_index=np.asarray(frame_index, dtype=np.int32),
            camera_names=np.asarray(CAMERA_NAMES),
            real_images=real_images,
            agents=agents,
            roads=roads,
            ego=ego,
            rig=rig,
            source_intrinsics=source_intrinsics,
            source_extrinsics=source_extrinsics,
            source_image_sizes=source_sizes,
            map_pose_offset=map_offset.astype(np.float64),
        )
        entries.append(
            {
                "file": filename,
                "segment_id": segment_id,
                "timestamp_micros": timestamp,
                "frame_index": frame_index,
                "agents": int(agents.shape[0]),
                "roads": int(roads.shape[0]),
            }
        )
        emitted += 1
    return entries


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
    parser.add_argument("--output", type=Path, required=True, help="Processed sample directory")
    parser.add_argument("--real-width", type=int, default=384)
    parser.add_argument("--real-height", type=int, default=256)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-segments", type=int)
    parser.add_argument("--max-frames-per-segment", type=int)
    parser.add_argument(
        "--scene-radius",
        type=float,
        default=220.0,
        help="Keep primitives within this radius; renderer far plane is 200 m",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep validated existing samples and rebuild the manifest",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Segment worker processes (default: min(8, CPU count))",
    )
    args = parser.parse_args()
    if args.frame_stride < 1:
        parser.error("--frame-stride must be >= 1")
    if args.real_width < 1 or args.real_height < 1:
        parser.error("real image dimensions must be positive")
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive")

    args.output.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, int | str]] = []
    files = _input_files(args.input, args.max_segments)
    worker_args = [
        (
            tfrecord,
            args.output,
            (args.real_width, args.real_height),
            args.frame_stride,
            args.max_frames_per_segment,
            args.scene_radius,
            args.overwrite,
            args.resume,
        )
        for tfrecord in files
    ]
    if args.workers == 1:
        results = (process_segment(*item) for item in worker_args)
        for index, (tfrecord, segment_entries) in enumerate(zip(files, results), 1):
            entries.extend(segment_entries)
            print(
                f"[{index}/{len(files)}] {tfrecord.name}: "
                f"{'kept/wrote' if args.resume else 'wrote'} {len(segment_entries)} samples",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            results = executor.map(process_segment, *zip(*worker_args))
            for index, (tfrecord, segment_entries) in enumerate(zip(files, results), 1):
                entries.extend(segment_entries)
                print(
                    f"[{index}/{len(files)}] {tfrecord.name}: "
                    f"{'kept/wrote' if args.resume else 'wrote'} {len(segment_entries)} samples",
                    flush=True,
                )

    with (args.output / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
    counts = Counter(entry["segment_id"] for entry in entries)
    print(
        f"processed {len(entries)} frames from {len(counts)} segments into {args.output} "
        f"at real resolution {args.real_width}x{args.real_height}"
    )


if __name__ == "__main__":
    main()
