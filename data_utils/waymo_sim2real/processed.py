"""Shared schema and helpers for compact Waymo sim-to-real samples."""

from __future__ import annotations

from pathlib import Path
import json
import os

import numpy as np


PROCESSED_SCHEMA_VERSION = 1
FEATURE_SCHEMA_VERSION = 1
EGO_SCHEMA_VERSION = 1

# Policy/checkpoint order. The visualization deliberately reorders this to
# left/front/right for human inspection.
CAMERA_ENUMS = (1, 2, 3)
CAMERA_NAMES = ("front", "front_left", "front_right")
DISPLAY_CAMERA_NAMES = ("front_left", "front", "front_right")

SIM_HEIGHT = 64
SIM_WIDTH = 96
WAYMO_REAR_AXLE_TO_BOX_CENTER = 1.44
RENDER_NEAR = 0.15
RENDER_FAR = 200.0

TEACHER_FEATURE_DIM = 256

# Ego observation vector under the JERK dynamics model, mirroring
# ``compute_observations`` in drive.h (EGO_FEATURES_JERK).  The planning head
# distilled alongside the scene feature consumes exactly this layout, so the
# normalizers below must track the #defines rather than be re-derived.
EGO_OBS_DIM = 11
MAX_SPEED = 100.0
MAX_VEH_LEN = 30.0
MAX_VEH_WIDTH = 15.0
JERK_LONG_MIN = -15.0  # JERK_LONG[0]
JERK_LONG_MAX = 4.0  # JERK_LONG[3]
JERK_LAT_MAX = 4.0  # JERK_LAT[2]
GOAL_OBS_SCALE = 0.005
# drive_3cam.ini `goal_target_distance`: the sim aims at a lane point this far
# ahead, so the Waymo analogue is the ego's own logged pose after the same
# travelled distance.
GOAL_TARGET_DISTANCE = 30.0
# Waymo's self-driving Chrysler Pacifica. WOD never labels the ego vehicle, so
# the box the sim would have read from a WOMD SDC track is supplied here.
WAYMO_SDC_LENGTH = 5.286
WAYMO_SDC_WIDTH = 2.332


def atomic_savez(path: str | Path, **arrays) -> None:
    """Write an npz atomically without numpy appending a surprise suffix."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temp.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def load_processed(path: str | Path) -> dict[str, np.ndarray]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        sample = {key: archive[key] for key in archive.files}
    validate_processed(sample, path)
    return sample


def load_render_input(path: str | Path) -> dict[str, np.ndarray]:
    """Load only arrays needed by the abstract renderer and frozen teacher.

    In particular, this deliberately avoids decompressing ``real_images``;
    those are large and are not an input to sim perception.
    """
    path = Path(path)
    keys = (
        "schema_version",
        "segment_id",
        "timestamp_micros",
        "agents",
        "roads",
        "ego",
        "rig",
    )
    with np.load(path, allow_pickle=False) as archive:
        missing = [key for key in keys if key not in archive]
        if missing:
            raise ValueError(f"{path} is missing render fields: {missing}")
        sample = {key: archive[key] for key in keys}
    version = int(np.asarray(sample["schema_version"]).item())
    if version != PROCESSED_SCHEMA_VERSION:
        raise ValueError(f"{path} schema {version}, expected {PROCESSED_SCHEMA_VERSION}")
    expected_shapes = {"agents": (8,), "roads": (6,), "ego": (5,), "rig": (3, 20)}
    for key, tail in expected_shapes.items():
        value = sample[key]
        if tuple(value.shape[-len(tail) :]) != tail:
            raise ValueError(f"{path} {key} has shape {value.shape}, expected tail {tail}")
    return sample


def validate_processed(sample: dict[str, np.ndarray], path: Path | None = None) -> None:
    label = str(path) if path is not None else "processed sample"
    required = {
        "schema_version",
        "segment_id",
        "timestamp_micros",
        "frame_index",
        "camera_names",
        "real_images",
        "agents",
        "roads",
        "ego",
        "rig",
        "source_intrinsics",
        "source_extrinsics",
        "source_image_sizes",
    }
    missing = sorted(required - sample.keys())
    if missing:
        raise ValueError(f"{label} is missing fields: {missing}")
    version = int(np.asarray(sample["schema_version"]).item())
    if version != PROCESSED_SCHEMA_VERSION:
        raise ValueError(f"{label} schema {version}, expected {PROCESSED_SCHEMA_VERSION}")
    names = tuple(str(v) for v in sample["camera_names"].tolist())
    if names != CAMERA_NAMES:
        raise ValueError(f"{label} camera order {names}, expected {CAMERA_NAMES}")
    real = sample["real_images"]
    if real.ndim != 4 or real.shape[0] != 3 or real.shape[-1] != 3 or real.dtype != np.uint8:
        raise ValueError(f"{label} real_images must be uint8 [3,H,W,3], got {real.shape} {real.dtype}")
    expected_shapes = {"agents": (8,), "roads": (6,), "ego": (5,), "rig": (3, 20)}
    for key, tail in expected_shapes.items():
        value = sample[key]
        if tuple(value.shape[-len(tail) :]) != tail:
            raise ValueError(f"{label} {key} has shape {value.shape}, expected tail {tail}")


def list_processed_files(directory: str | Path, max_samples: int | None = None) -> list[Path]:
    directory = Path(directory)
    manifest = directory / "manifest.jsonl"
    if manifest.exists():
        files = []
        for line in manifest.read_text().splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            files.append(directory / entry["file"])
    else:
        files = sorted(directory.glob("*.npz"))
    files = [path for path in files if path.is_file()]
    return files if max_samples is None else files[:max_samples]


def load_ego_state(path: str | Path) -> dict[str, np.ndarray]:
    """Load one segment's per-frame ego observation table."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        state = {key: archive[key] for key in archive.files}
    required = {"schema_version", "segment_id", "timestamp_micros", "ego_obs"}
    missing = sorted(required - state.keys())
    if missing:
        raise ValueError(f"{path} is missing fields: {missing}")
    version = int(np.asarray(state["schema_version"]).item())
    if version != EGO_SCHEMA_VERSION:
        raise ValueError(f"{path} schema {version}, expected {EGO_SCHEMA_VERSION}")
    obs = state["ego_obs"]
    if obs.ndim != 2 or obs.shape[1] != EGO_OBS_DIM or obs.dtype != np.float32:
        raise ValueError(f"{path} ego_obs must be float32 [N,{EGO_OBS_DIM}], got {obs.shape} {obs.dtype}")
    if len(state["timestamp_micros"]) != len(obs):
        raise ValueError(f"{path} has {len(obs)} rows for {len(state['timestamp_micros'])} timestamps")
    return state


def load_feature(path: str | Path) -> dict[str, np.ndarray]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        feature = {key: archive[key] for key in archive.files}
    required = {
        "schema_version",
        "segment_id",
        "timestamp_micros",
        "camera_names",
        "teacher_feature",
        "sim_images",
        "checkpoint_sha256",
    }
    missing = sorted(required - feature.keys())
    if missing:
        raise ValueError(f"{path} is missing fields: {missing}")
    version = int(np.asarray(feature["schema_version"]).item())
    if version != FEATURE_SCHEMA_VERSION:
        raise ValueError(f"{path} schema {version}, expected {FEATURE_SCHEMA_VERSION}")
    sim = feature["sim_images"]
    if sim.shape != (3, SIM_HEIGHT, SIM_WIDTH, 3) or sim.dtype != np.uint8:
        raise ValueError(f"{path} sim_images has invalid shape/dtype {sim.shape} {sim.dtype}")
    if feature["teacher_feature"].shape != (TEACHER_FEATURE_DIM,):
        raise ValueError(f"{path} teacher_feature must be [{TEACHER_FEATURE_DIM}]")
    return feature
