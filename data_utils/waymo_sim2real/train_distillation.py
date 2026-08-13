"""Adapt real-image perception onto a frozen recurrent PufferDrive policy.

The default DrivoR-style student uses a pretrained DINOv2 ViT-S/14, 16 learned
scene registers per camera, and rank-32 Q/V LoRA. It is trained to reproduce
the frozen simulation perception's 256-D output immediately before the LSTM.
The frozen teacher planner evaluates both latents with a fresh zero LSTM state,
which gives the exact first-frame policy requested for recurrent checkpoints.

    L = feature_weight * ||E_real - E_sim||^2
        + cosine_weight * (1 - cosine(E_real, E_sim))
        + plan_weight * KL(pi_sim || pi_real)

``E_sim`` never has to be recomputed: it was extracted once and pinned by
checkpoint SHA256 in ``teacher_features/manifest.json``, which also makes
accidental teacher updates impossible. The DINO base and complete recurrent
planner are frozen; LoRA, camera registers, fusion, and projection are trained.

Example:

    .venv/bin/python -m data_utils.waymo_sim2real.train_distillation \
        --root artifacts/carla_sim2real/sample1k_dino \
        --checkpoint experiments/skynet/model_puffer_giga_3cam_001400.pt \
        --output artifacts/carla_sim2real/runs/dino_carla1k
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .giga_conditioning import GIGA_EGO_OBS_DIM, append_giga_conditioning
from .processed import CAMERA_NAMES, EGO_OBS_DIM, TEACHER_FEATURE_DIM, load_ego_state
from .real_perception import (
    DistillationLoss,
    RealPerception,
    RealPerceptionConfig,
    ViTRealPerception,
    ViTRealPerceptionConfig,
)
from .teacher import load_frozen_planning_head, sha256_file


CHECKPOINT_SCHEMA_VERSION = 3
DEPLOYMENT_SCHEMA_VERSION = 1

# Changing any of these silently reshapes the learning-rate schedule or the
# sample stream, so a resume that disagrees is a different run wearing the same
# checkpoint. LambdaLR only stores its step counter, not the curve it indexes.
SCHEDULE_ARGS = (
    "epochs",
    "batch_size",
    "accumulation_steps",
    "learning_rate",
    "backbone_learning_rate",
    "warmup_fraction",
    "train_frame_stride",
    "max_train_samples",
    "conditioning_seed",
)


class PairedWaymoFeatureDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]):
    """Strict processed-image/teacher-feature pairing backed by manifests."""

    def __init__(
        self,
        split_root: str | Path,
        max_samples: int | None = None,
        frame_stride: int = 1,
        require_ego: bool = True,
        expected_ego_dim: int = EGO_OBS_DIM,
        conditioning_seed: int = 42,
    ):
        self.split_root = Path(split_root)
        self.processed_dir = self.split_root / "processed"
        self.features_dir = self.split_root / "teacher_features"
        self.ego_dir = self.split_root / "ego_state"
        processed_manifest = self.processed_dir / "manifest.jsonl"
        feature_manifest = self.features_dir / "manifest.json"
        if not processed_manifest.is_file():
            raise FileNotFoundError(f"missing processed manifest: {processed_manifest}")
        if not feature_manifest.is_file():
            raise FileNotFoundError(f"missing teacher manifest: {feature_manifest}")
        if frame_stride < 1:
            raise ValueError("frame_stride must be positive")

        processed_entries = [json.loads(line) for line in processed_manifest.read_text().splitlines() if line]
        processed_by_file = {entry["file"]: entry for entry in processed_entries}
        if len(processed_by_file) != len(processed_entries):
            raise ValueError(f"{processed_manifest} contains duplicate files")

        metadata = json.loads(feature_manifest.read_text())
        if tuple(metadata.get("camera_names", ())) != CAMERA_NAMES:
            raise ValueError(
                f"{feature_manifest} camera order {metadata.get('camera_names')} does not match {CAMERA_NAMES}"
            )
        if int(metadata.get("feature_dim", -1)) != TEACHER_FEATURE_DIM:
            raise ValueError(f"{feature_manifest} does not contain {TEACHER_FEATURE_DIM}-D teacher features")
        self.teacher_checkpoint_sha256 = str(metadata.get("checkpoint_sha256", ""))
        if len(self.teacher_checkpoint_sha256) != 64:
            raise ValueError(f"{feature_manifest} has an invalid checkpoint SHA256")

        samples: list[dict[str, object]] = []
        seen_processed: set[str] = set()
        for entry in metadata.get("samples", ()):
            processed_name = entry["processed_file"]
            if processed_name not in processed_by_file:
                raise ValueError(f"teacher sample references unknown processed file {processed_name}")
            if processed_name in seen_processed:
                raise ValueError(f"teacher manifest pairs {processed_name} more than once")
            seen_processed.add(processed_name)
            samples.append(entry)
        if len(samples) != len(processed_entries):
            raise ValueError(
                f"pair count {len(samples)} does not match processed count {len(processed_entries)}"
            )

        # Subsample in time inside each segment rather than across the flat list:
        # Waymo logs at 10 Hz and neighbouring frames are near-duplicates, so a
        # stride is the cheapest way to trade redundant frames for epochs.
        samples.sort(key=lambda entry: (entry["segment_id"], entry["timestamp_micros"]))
        kept: list[dict[str, object]] = []
        position = 0
        previous_segment: str | None = None
        for entry in samples:
            if entry["segment_id"] != previous_segment:
                previous_segment, position = entry["segment_id"], 0
            if position % frame_stride == 0:
                kept.append(entry)
            position += 1
        if max_samples is not None:
            if max_samples < 1:
                raise ValueError("max_samples must be positive")
            kept = kept[:max_samples]
        if not kept:
            raise ValueError(f"{self.split_root} yielded no samples")

        self.segment_ids = sorted({str(entry["segment_id"]) for entry in kept})
        segment_index = {segment: index for index, segment in enumerate(self.segment_ids)}
        self.pairs: list[tuple[Path, Path]] = []
        self.segments = np.empty(len(kept), dtype=np.int32)
        for index, entry in enumerate(kept):
            processed_path = self.processed_dir / str(entry["processed_file"])
            feature_path = self.features_dir / str(entry["file"])
            if not processed_path.is_file() or not feature_path.is_file():
                raise FileNotFoundError(f"missing pair: {processed_path}, {feature_path}")
            self.pairs.append((processed_path, feature_path))
            self.segments[index] = segment_index[str(entry["segment_id"])]

        if expected_ego_dim not in (EGO_OBS_DIM, GIGA_EGO_OBS_DIM):
            raise ValueError(
                f"unsupported planner ego width {expected_ego_dim}; expected {EGO_OBS_DIM} "
                f"or {GIGA_EGO_OBS_DIM}"
            )
        self.expected_ego_dim = expected_ego_dim
        self.conditioning_seed = conditioning_seed
        self.ego = self._load_ego_table(kept) if require_ego else None

    def _load_ego_table(self, entries: list[dict[str, object]]) -> np.ndarray:
        """Resolve every sample's ego vector once, into one resident array.

        A few megabytes for the whole split, so dataloader workers never touch
        the ego files and the join is validated up front instead of at step 40k.
        """
        if not self.ego_dir.is_dir():
            raise FileNotFoundError(
                f"missing {self.ego_dir}; the planning loss needs the reconstructed ego state. Run\n"
                f"  python -m data_utils.waymo_sim2real.extract_ego_state "
                f"--input <raw tfrecords> --output {self.ego_dir} --workers 8 --resume\n"
                f"or pass --plan-weight 0 to train on the feature term alone."
            )
        needed = sorted({str(entry["segment_id"]) for entry in entries})
        tables: dict[str, dict[int, np.ndarray]] = {}
        for segment in needed:
            path = self.ego_dir / f"{segment}.npz"
            if not path.is_file():
                raise FileNotFoundError(f"missing ego state for segment {segment}: {path}")
            state = load_ego_state(path)
            stamps = np.asarray(state["timestamp_micros"], dtype=np.int64)
            tables[segment] = dict(zip(stamps.tolist(), np.asarray(state["ego_obs"], dtype=np.float32)))
        ego = np.empty((len(entries), self.expected_ego_dim), dtype=np.float32)
        for index, entry in enumerate(entries):
            segment, timestamp = str(entry["segment_id"]), int(entry["timestamp_micros"])
            row = tables[segment].get(timestamp)
            if row is None:
                raise ValueError(
                    f"segment {segment} has no ego state at {timestamp}; the ego tables were "
                    f"built from different TFRecords than the processed samples"
                )
            ego[index] = (
                append_giga_conditioning(row, segment, self.conditioning_seed)
                if self.expected_ego_dim == GIGA_EGO_OBS_DIM
                else row
            )
        return ego

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        processed_path, feature_path = self.pairs[index]
        with np.load(processed_path, allow_pickle=False) as archive:
            images = np.asarray(archive["real_images"])
        with np.load(feature_path, allow_pickle=False) as archive:
            target = np.asarray(archive["teacher_feature"])
            saved_hash = str(np.asarray(archive["checkpoint_sha256"]).item())
        if images.shape != (3, 256, 384, 3) or images.dtype != np.uint8:
            raise ValueError(f"{processed_path} has invalid real_images {images.shape} {images.dtype}")
        if target.shape != (TEACHER_FEATURE_DIM,) or target.dtype not in (
            np.float16,
            np.float32,
            np.float64,
        ):
            raise ValueError(f"{feature_path} has invalid teacher_feature {target.shape} {target.dtype}")
        if saved_hash != self.teacher_checkpoint_sha256:
            raise ValueError(
                f"{feature_path} uses teacher {saved_hash}, expected {self.teacher_checkpoint_sha256}"
            )
        # Both arrays originate in read-only zip buffers. Copy before converting
        # so worker tensors own writable, contiguous storage.
        image_tensor = torch.from_numpy(images.copy()).permute(0, 3, 1, 2)
        target_tensor = torch.from_numpy(target.astype(np.float32, copy=True))
        ego = (
            self.ego[index]
            if self.ego is not None
            else np.zeros(self.expected_ego_dim, dtype=np.float32)
        )
        return image_tensor, target_tensor, torch.from_numpy(ego.copy()), int(self.segments[index])


def target_statistics(dataset: PairedWaymoFeatureDataset, sample: int, workers: int) -> dict[str, object]:
    """Mean and variance of the teacher features, cached beside them.

    Without these there is no scale to read the loss against: a constant
    predictor already scores a high cosine similarity because these features
    carry a large mean offset, so only variance explained says whether the
    student learned anything.
    """
    # The cache is keyed by the population it summarizes as well as the teacher:
    # a --max-train-samples smoke run must not leave two samples' statistics
    # behind for the next full run to read back as the reference variance.
    cache = dataset.features_dir / "target_stats.json"
    population = len(dataset.pairs)
    if cache.is_file():
        stored = json.loads(cache.read_text())
        if (
            stored.get("checkpoint_sha256") == dataset.teacher_checkpoint_sha256
            and stored.get("population") == population
        ):
            return stored

    paths = [feature for _, feature in dataset.pairs]
    if 0 < sample < len(paths):
        rows = np.linspace(0, len(paths) - 1, sample).astype(int)
        paths = [paths[index] for index in rows]

    def read(path: Path) -> np.ndarray:
        with np.load(path, allow_pickle=False) as archive:
            return np.asarray(archive["teacher_feature"], dtype=np.float64)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        features = np.stack(list(executor.map(read, paths)))
    stats = {
        "checkpoint_sha256": dataset.teacher_checkpoint_sha256,
        "population": population,
        "num_samples": len(features),
        "mean": features.mean(axis=0).tolist(),
        "std": features.std(axis=0).tolist(),
        "variance": float(features.var(axis=0).mean()),
    }
    temporary = cache.with_name(f".{cache.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(stats) + "\n")
    os.replace(temporary, cache)
    return stats


def _photometric_augment(images: torch.Tensor) -> torch.Tensor:
    """Mild geometry-preserving jitter shared across the three views."""

    images = images.float().div_(255.0)
    batch = images.shape[0]
    shape = (batch, 1, 1, 1, 1)
    brightness = torch.empty(shape, device=images.device).uniform_(-0.08, 0.08)
    contrast = torch.empty(shape, device=images.device).uniform_(0.85, 1.15)
    saturation = torch.empty(shape, device=images.device).uniform_(0.85, 1.15)
    mean = images.mean(dim=(-3, -2, -1), keepdim=True)
    images = (images - mean) * contrast + mean + brightness
    gray = images.mean(dim=2, keepdim=True)
    return ((images - gray) * saturation + gray).clamp_(0.0, 1.0)


def _parameter_groups(
    model: nn.Module,
    weight_decay: float,
    head_learning_rate: float,
    backbone_learning_rate: float,
) -> list[dict[str, object]]:
    groups: dict[tuple[str, bool], list[nn.Parameter]] = {
        (scope, decay): [] for scope in ("backbone", "head") for decay in (False, True)
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        scope = "backbone" if name.startswith(("backbone.", "view_norm.")) else "head"
        decay = parameter.ndim > 1 and not name.endswith("bias") and "camera_embedding" not in name
        groups[(scope, decay)].append(parameter)
    learning_rates = {"backbone": backbone_learning_rate, "head": head_learning_rate}
    return [
        {
            "params": parameters,
            "weight_decay": weight_decay if decay else 0.0,
            "lr": learning_rates[scope],
            "group_name": f"{scope}_{'decay' if decay else 'no_decay'}",
        }
        for (scope, decay), parameters in groups.items()
        if parameters
    ]


def _lr_factor(step: int, total_steps: int, warmup_steps: int) -> float:
    if warmup_steps and step < warmup_steps:
        return max(1e-8, (step + 1) / warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))


def _amp_context(device: torch.device, mode: str):
    if mode == "off":
        return nullcontext()
    dtype = torch.bfloat16 if mode == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


def _reduce_metrics(sums: dict[str, float], metrics: dict[str, torch.Tensor], batch: int) -> None:
    for name, value in metrics.items():
        sums[name] = sums.get(name, 0.0) + float(value) * batch
    sums["samples"] = sums.get("samples", 0.0) + batch


def _averages(sums: dict[str, float], variance: float | None = None) -> dict[str, float]:
    samples = sums.pop("samples")
    averages = {name: value / samples for name, value in sums.items()}
    # RMSE is the root of the epoch's mean square error. Averaging per-batch
    # roots instead is Jensen-biased low, which quietly flatters the run.
    if "mse" in averages:
        averages["rmse"] = math.sqrt(max(averages["mse"], 0.0))
        if variance:
            averages["r2"] = 1.0 - averages["mse"] / variance
    return averages


def _within_segment_r2(
    predictions: np.ndarray, targets: np.ndarray, segments: np.ndarray
) -> tuple[float, float]:
    """Variance explained after each segment's own mean is removed.

    Roughly half of the teacher feature's variance is which segment a frame came
    from -- the static look of the street -- and a student can capture that
    without ever resolving a vehicle. What perception is actually for is the
    other half, the frame-to-frame change, which this isolates by centring both
    sides per segment.
    """
    order = np.argsort(segments, kind="stable")
    bounds = np.flatnonzero(np.diff(segments[order])) + 1
    residual_error = 0.0
    residual_total = 0.0
    count = 0
    for block in np.split(order, bounds):
        if len(block) < 2:
            continue
        centred_target = targets[block] - targets[block].mean(axis=0)
        centred_prediction = predictions[block] - predictions[block].mean(axis=0)
        residual_error += float(((centred_prediction - centred_target) ** 2).sum())
        residual_total += float((centred_target**2).sum())
        count += len(block)
    if count == 0 or residual_total == 0.0:
        return float("nan"), 0.0
    return 1.0 - residual_error / residual_total, residual_total / (count * targets.shape[1])


def _wandb_epoch_payload(record: dict[str, object]) -> dict[str, float]:
    """Flatten the local JSON record into stable W&B metric names."""

    payload = {
        "epoch": float(record["epoch"]),
        "epoch/elapsed_seconds": float(record["elapsed_seconds"]),
        "validation/best_loss": float(record["best_validation_loss"]),
    }
    for split in ("train", "validation"):
        metrics = record[split]
        if not isinstance(metrics, dict):
            raise TypeError(f"{split} metrics must be a dictionary")
        for name, value in metrics.items():
            payload[f"{split}/{name}"] = float(value)
    if "peak_gpu_memory_gib" in record:
        payload["system/peak_gpu_memory_gib"] = float(record["peak_gpu_memory_gib"])
    return payload


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: DistillationLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp: str,
    accumulation_steps: int,
    max_grad_norm: float,
    epoch: int,
    global_step: int,
    log_interval: int,
    variance: float,
    step_logger: Callable[[dict[str, float], int], None] | None,
) -> tuple[dict[str, float], int]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    sums: dict[str, float] = {}
    interval_sums: dict[str, float] = {}
    skipped_steps = 0
    for batch_index, (images, targets, egos, _) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        egos = egos.to(device, non_blocking=True)
        images = _photometric_augment(images)
        group_start = (batch_index // accumulation_steps) * accumulation_steps
        group_size = min(accumulation_steps, len(loader) - group_start)
        with _amp_context(device, amp):
            predictions = model(images)
            loss, metrics = criterion(predictions, targets, egos)
            backward_loss = loss / group_size
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite training loss at batch {batch_index}: {float(loss)}")
        scaler.scale(backward_loss).backward()
        _reduce_metrics(sums, metrics, images.shape[0])
        _reduce_metrics(interval_sums, metrics, images.shape[0])
        should_step = (batch_index + 1) % accumulation_steps == 0 or batch_index + 1 == len(loader)
        if should_step:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            if not torch.isfinite(grad_norm):
                # Under fp16 an overflowing step is the ordinary event GradScaler
                # exists to absorb: it drops the step and halves the scale below.
                # Under bf16 or off there is no scale to retune, so it is a fault.
                if amp != "fp16":
                    raise FloatingPointError(f"non-finite gradient norm at batch {batch_index}")
                skipped_steps += 1
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            global_step += 1
            is_last_batch = batch_index + 1 == len(loader)
            if step_logger is not None and global_step % log_interval == 0 and not is_last_batch:
                interval_metrics = _averages(interval_sums, variance)
                interval_sums = {}
                payload = {
                    "epoch": float(epoch),
                    "train/grad_norm": float(grad_norm),
                    "train/skipped_steps": float(skipped_steps),
                }
                payload.update({f"train/step_{name}": value for name, value in interval_metrics.items()})
                for group in optimizer.param_groups:
                    scope = str(group["group_name"]).split("_", 1)[0]
                    payload[f"train/{scope}_lr"] = float(group["lr"])
                step_logger(payload, global_step)
    result = _averages(sums, variance)
    result["skipped_steps"] = float(skipped_steps)
    for group in optimizer.param_groups:
        scope = str(group["group_name"]).split("_", 1)[0]
        result[f"{scope}_lr"] = float(group["lr"])
    return result, global_step


@torch.inference_mode()
def _validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: DistillationLoss,
    device: torch.device,
    amp: str,
    variance: float,
) -> dict[str, float]:
    model.eval()
    sums: dict[str, float] = {}
    predictions: list[np.ndarray] = []
    targets_seen: list[np.ndarray] = []
    segments_seen: list[np.ndarray] = []
    for images, targets, egos, segments in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        egos = egos.to(device, non_blocking=True)
        with _amp_context(device, amp):
            prediction = model(images)
            loss, metrics = criterion(prediction, targets, egos)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite validation loss")
        _reduce_metrics(sums, metrics, images.shape[0])
        predictions.append(prediction.float().cpu().numpy())
        targets_seen.append(targets.float().cpu().numpy())
        segments_seen.append(segments.numpy())
    result = _averages(sums, variance)
    within_r2, within_variance = _within_segment_r2(
        np.concatenate(predictions), np.concatenate(targets_seen), np.concatenate(segments_seen)
    )
    result["r2_within_segment"] = within_r2
    result["within_segment_variance"] = within_variance
    return result


def _atomic_torch_save(payload: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    global_step: int,
    best_validation_loss: float,
    teacher_hash: str,
    wandb_run_id: str | None,
    args: argparse.Namespace,
) -> dict[str, object]:
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "model": model.state_dict(),
        "model_config": model.config.to_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_validation_loss": best_validation_loss,
        "teacher_checkpoint_sha256": teacher_hash,
        "backbone_initialization": model.pretrained_source,
        "backbone_checkpoint_sha256": model.pretrained_sha256,
        "backbone_revision": model.pretrained_revision,
        "wandb_run_id": wandb_run_id,
        "schedule": {name: getattr(args, name) for name in SCHEDULE_ARGS},
        "args": vars(args),
    }


def _deployment_bundle(
    model: nn.Module,
    teacher_hash: str,
    architecture: str,
) -> dict[str, object]:
    """Self-contained visual encoder artifact for replacing sim perception."""

    return {
        "schema_version": DEPLOYMENT_SCHEMA_VERSION,
        "architecture": architecture,
        "model_config": model.config.to_dict(),
        "model": model.state_dict(),
        "teacher_checkpoint_sha256": teacher_hash,
        "backbone_initialization": model.pretrained_source,
        "backbone_checkpoint_sha256": model.pretrained_sha256,
        "backbone_revision": model.pretrained_revision,
        "camera_names": list(CAMERA_NAMES),
        "output": {"name": "scene_feature", "dimension": TEACHER_FEATURE_DIM},
    }


def _build_model(args: argparse.Namespace, *, resume: bool) -> nn.Module:
    pretrained = not args.random_init and not resume
    if args.architecture == "convnext_tiny":
        return RealPerception(RealPerceptionConfig(), pretrained=pretrained)
    config = ViTRealPerceptionConfig(
        backbone_revision=args.backbone_revision,
        num_scene_tokens=args.num_scene_tokens,
        lora_rank=args.lora_rank,
        fusion_layers=args.fusion_layers,
        fusion_heads=args.fusion_heads,
        fusion_dropout=args.fusion_dropout,
    )
    return ViTRealPerception(
        config,
        pretrained=pretrained,
        weights_path=args.backbone_weights,
    )


def _loader(dataset: Dataset, args: argparse.Namespace, training: bool) -> DataLoader:
    kwargs: dict[str, object] = {
        "batch_size": args.batch_size,
        "shuffle": training,
        "num_workers": args.workers,
        "pin_memory": args.device.startswith("cuda"),
        "drop_last": False,
        "persistent_workers": args.workers > 0,
    }
    if args.workers > 0:
        kwargs["prefetch_factor"] = args.prefetch_factor
    return DataLoader(dataset, **kwargs)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts/waymo_sim2real/full"))
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/waymo_sim2real/runs/real_perception_dinov2")
    )
    parser.add_argument(
        "--architecture",
        choices=("dino_vit_small", "convnext_tiny"),
        default="dino_vit_small",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Frozen DriveCam checkpoint supplying the planning head; required unless --plan-weight 0",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--accumulation-steps", type=int, default=1)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--backbone-learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-fraction", type=float, default=0.05)
    parser.add_argument("--feature-weight", type=float, default=1.0, help="lambda on the feature L2 term")
    parser.add_argument("--cosine-weight", type=float, default=0.1)
    parser.add_argument(
        "--plan-weight",
        type=float,
        default=1.0,
        help="Weight on the frozen planner's action KL; 0 disables it and the ego-state requirement",
    )
    parser.add_argument("--plan-temperature", type=float, default=1.0)
    parser.add_argument(
        "--standardize-targets",
        action="store_true",
        help="Divide the feature residual by each dimension's standard deviation",
    )
    parser.add_argument(
        "--freeze-backbone-stages",
        type=int,
        default=0,
        help="Freeze this many leading ConvNeXt stages (0-4) against overfitting",
    )
    parser.add_argument(
        "--backbone-weights",
        type=Path,
        help="Local DINOv2 model.safetensors; otherwise download the pinned Hugging Face revision",
    )
    parser.add_argument("--backbone-revision", default="main")
    parser.add_argument("--num-scene-tokens", type=int, default=16)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--fusion-layers", type=int, default=2)
    parser.add_argument("--fusion-heads", type=int, default=6)
    parser.add_argument("--fusion-dropout", type=float, default=0.1)
    parser.add_argument(
        "--train-frame-stride",
        type=int,
        default=1,
        help="Keep every Nth frame of each training segment; Waymo logs at 10 Hz",
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--amp", choices=("bf16", "fp16", "off"), default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--conditioning-seed",
        type=int,
        default=42,
        help="Deterministic per-segment giga conditioning draws for a 24-D planning head",
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--overwrite", action="store_true", help="Discard an existing run in --output and start over"
    )
    parser.add_argument("--target-stats-samples", type=int, default=8192)
    parser.add_argument("--wandb-project", default="pufferdrive-sim2real")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-name")
    parser.add_argument("--wandb-group")
    parser.add_argument("--wandb-tags", nargs="*", default=())
    parser.add_argument("--wandb-run-id")
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
        help="W&B is online by default; use offline or disabled explicitly",
    )
    parser.add_argument("--wandb-log-interval", type=int, default=50)
    parser.add_argument(
        "--random-init",
        action="store_true",
        help="Disable pretrained backbone initialization (intended only as an ablation)",
    )
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    for name in (
        "epochs",
        "batch_size",
        "accumulation_steps",
        "prefetch_factor",
        "train_frame_stride",
        "num_scene_tokens",
        "lora_rank",
        "fusion_layers",
        "fusion_heads",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be >= 1")
    if args.workers < 0:
        parser.error("--workers must be >= 0")
    if not 0 <= args.warmup_fraction < 1:
        parser.error("--warmup-fraction must be in [0, 1)")
    if args.learning_rate <= 0 or args.backbone_learning_rate <= 0:
        parser.error("learning rates must be positive")
    if args.wandb_log_interval < 1:
        parser.error("--wandb-log-interval must be >= 1")
    if args.device == "cpu" and args.amp != "off":
        parser.error("CPU training requires --amp off")
    if min(args.feature_weight, args.cosine_weight, args.plan_weight) < 0:
        parser.error("loss weights must be non-negative")
    if not 0 <= args.freeze_backbone_stages <= 4:
        parser.error("--freeze-backbone-stages must be in [0, 4]")
    if args.architecture != "convnext_tiny" and args.freeze_backbone_stages:
        parser.error("--freeze-backbone-stages only applies to --architecture convnext_tiny")
    if not 0 <= args.fusion_dropout < 1:
        parser.error("--fusion-dropout must be in [0, 1)")
    if args.backbone_weights is not None and not args.backbone_weights.is_file():
        parser.error(f"DINOv2 weights do not exist: {args.backbone_weights}")
    if args.plan_weight > 0 and args.checkpoint is None:
        parser.error("--checkpoint is required for the planning loss; pass --plan-weight 0 to drop it")
    if args.checkpoint is not None and not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    if args.resume is not None and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive")

    # A run directory holds append-only metrics and a best-so-far checkpoint.
    # Starting fresh on top of one appends a second history to the same file and
    # resets the best-loss watermark, so the better checkpoint is overwritten by
    # a worse first epoch. Make that collision explicit instead.
    occupied = [name for name in ("last.pt", "best.pt", "metrics.jsonl") if (args.output / name).exists()]
    if occupied and args.resume is None:
        if not args.overwrite:
            parser.error(
                f"{args.output} already holds {', '.join(occupied)}; pass "
                f"--resume {args.output / 'last.pt'} to continue that run, --overwrite to discard "
                f"it, or choose a different --output"
            )
        for name in occupied:
            (args.output / name).unlink()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    _seed_everything(args.seed)

    plan_head = None
    expected_ego_dim = EGO_OBS_DIM
    if args.plan_weight > 0:
        plan_head = load_frozen_planning_head(args.checkpoint, device, require_recurrent=True)
        expected_ego_dim = plan_head.ego_features
    needs_ego = plan_head is not None
    training = PairedWaymoFeatureDataset(
        args.root / "training",
        args.max_train_samples,
        args.train_frame_stride,
        needs_ego,
        expected_ego_dim,
        args.conditioning_seed,
    )
    validation = PairedWaymoFeatureDataset(
        args.root / "validation",
        args.max_validation_samples,
        1,
        needs_ego,
        expected_ego_dim,
        args.conditioning_seed,
    )
    if training.teacher_checkpoint_sha256 != validation.teacher_checkpoint_sha256:
        raise ValueError("training and validation features were extracted from different teachers")
    teacher_hash = training.teacher_checkpoint_sha256
    if args.checkpoint is not None:
        supplied = sha256_file(args.checkpoint)
        if supplied != teacher_hash:
            raise ValueError(
                f"--checkpoint hashes to {supplied}, but the cached features came from {teacher_hash}"
            )
    training_loader = _loader(training, args, training=True)
    validation_loader = _loader(validation, args, training=False)

    train_stats = target_statistics(training, args.target_stats_samples, args.workers or 4)
    validation_stats = target_statistics(validation, args.target_stats_samples, args.workers or 4)
    target_scale = None
    if args.standardize_targets:
        target_scale = torch.tensor(train_stats["std"], dtype=torch.float32).clamp_min(1e-6)

    # A resume checkpoint overwrites the entire model, so it does not need a
    # second backbone download. Fresh runs use the selected official weights.
    model = _build_model(args, resume=args.resume is not None).to(device)
    if isinstance(model, RealPerception):
        model.freeze_backbone_stages(args.freeze_backbone_stages)
    frozen_parameters = model.frozen_parameters

    criterion = DistillationLoss(
        feature_weight=args.feature_weight,
        cosine_weight=args.cosine_weight,
        plan_weight=args.plan_weight,
        plan_head=plan_head,
        temperature=args.plan_temperature,
        target_scale=target_scale,
    ).to(device)

    optimizer = torch.optim.AdamW(
        _parameter_groups(
            model,
            args.weight_decay,
            args.learning_rate,
            args.backbone_learning_rate,
        ),
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    optimizer_steps_per_epoch = math.ceil(len(training_loader) / args.accumulation_steps)
    total_steps = args.epochs * optimizer_steps_per_epoch
    warmup_steps = int(args.warmup_fraction * total_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: _lr_factor(step, total_steps, warmup_steps)
    )
    scaler = torch.amp.GradScaler(device.type, enabled=args.amp == "fp16")

    start_epoch = 0
    global_step = 0
    best_validation_loss = math.inf
    resume_state: dict[str, object] | None = None
    if args.resume is not None:
        resume_state = torch.load(args.resume, map_location="cpu", weights_only=False)
        if int(resume_state.get("schema_version", 0)) != CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                f"{args.resume} is a schema {resume_state.get('schema_version')} checkpoint; this "
                f"trainer writes schema {CHECKPOINT_SCHEMA_VERSION} and its objective differs"
            )
        if resume_state.get("teacher_checkpoint_sha256") != teacher_hash:
            raise ValueError("resume checkpoint was trained against a different teacher")
        if resume_state.get("model_config") != model.config.to_dict():
            raise ValueError("resume checkpoint architecture does not match")
        stored_schedule = dict(resume_state.get("schedule", {}))
        current_schedule = {name: getattr(args, name) for name in SCHEDULE_ARGS}
        changed = {
            name: (stored_schedule.get(name), value)
            for name, value in current_schedule.items()
            if stored_schedule.get(name) != value
        }
        if changed:
            raise ValueError(
                "resume would reshape the schedule or the sample stream: "
                + ", ".join(f"{name} {was!r} -> {now!r}" for name, (was, now) in sorted(changed.items()))
            )
        model.load_state_dict(resume_state["model"], strict=True)
        if isinstance(model, RealPerception):
            model.freeze_backbone_stages(args.freeze_backbone_stages)
        model.pretrained_source = str(resume_state.get("backbone_initialization", "unknown"))
        model.pretrained_sha256 = resume_state.get("backbone_checkpoint_sha256")
        model.pretrained_revision = resume_state.get("backbone_revision")
        optimizer.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])
        scaler.load_state_dict(resume_state["scaler"])
        start_epoch = int(resume_state["epoch"]) + 1
        global_step = int(resume_state["global_step"])
        best_validation_loss = float(resume_state["best_validation_loss"])

    args.output.mkdir(parents=True, exist_ok=True)
    checkpoint_wandb_id = None if resume_state is None else resume_state.get("wandb_run_id")
    if args.wandb_run_id and checkpoint_wandb_id and args.wandb_run_id != checkpoint_wandb_id:
        raise ValueError(
            f"--wandb-run-id {args.wandb_run_id} does not match checkpoint run {checkpoint_wandb_id}"
        )
    wandb_run_id = args.wandb_run_id or checkpoint_wandb_id
    run_metadata = {
        "model_config": model.config.to_dict(),
        "trainable_parameters": model.trainable_parameters,
        "frozen_backbone_parameters": frozen_parameters,
        "teacher_checkpoint_sha256": teacher_hash,
        "backbone_initialization": model.pretrained_source,
        "backbone_checkpoint_sha256": model.pretrained_sha256,
        "backbone_revision": model.pretrained_revision,
        "training_samples": len(training),
        "training_segments": len(training.segment_ids),
        "validation_samples": len(validation),
        "validation_segments": len(validation.segment_ids),
        "camera_names": list(CAMERA_NAMES),
        "objective": {
            "feature_weight": args.feature_weight,
            "cosine_weight": args.cosine_weight,
            "plan_weight": args.plan_weight,
            "plan_temperature": args.plan_temperature,
            "standardize_targets": args.standardize_targets,
            "action_dims": None if plan_head is None else plan_head.action_dims,
            "planner_mode": None if plan_head is None else plan_head.planner_mode,
        },
        # The reference every metric should be read against: a constant
        # predictor scores mse = variance and r2 = 0.
        "target_variance": {
            "train": train_stats["variance"],
            "validation": validation_stats["variance"],
            "sampled_from": train_stats["num_samples"],
        },
        "wandb": {
            "project": args.wandb_project,
            "entity": args.wandb_entity,
            "mode": args.wandb_mode,
            "run_id": wandb_run_id,
        },
        "args": {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()},
    }

    wandb_run = None
    if args.wandb_mode != "disabled":
        try:
            import wandb
        except ImportError as error:
            raise ImportError("W&B tracking is enabled by default; install the 'wandb' package") from error
        if wandb_run_id is None:
            wandb_run_id = wandb.util.generate_id()
        fixed_tags = ("sim2real", "waymo", args.architecture, "feature-distillation")
        try:
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_name or args.output.name,
                group=args.wandb_group,
                tags=tuple(dict.fromkeys(fixed_tags + tuple(args.wandb_tags))),
                id=wandb_run_id,
                resume="allow" if args.wandb_mode == "online" else None,
                mode=args.wandb_mode,
                job_type="feature-distillation",
                dir=str(args.output),
                config=run_metadata,
            )
        except Exception as error:
            raise RuntimeError(
                "failed to initialize W&B; authenticate with 'wandb login' or pass "
                "--wandb-mode offline/disabled"
            ) from error
        if wandb_run is None:
            raise RuntimeError("wandb.init returned no run")
        wandb_run.define_metric("global_step")
        wandb_run.define_metric("train/*", step_metric="global_step")
        wandb_run.define_metric("validation/*", step_metric="global_step")
        wandb_run.define_metric("epoch/*", step_metric="global_step")
        wandb_run.define_metric("system/*", step_metric="global_step")
        run_metadata["wandb"]["run_id"] = wandb_run.id
        run_metadata["wandb"]["url"] = wandb_run.url
        wandb_run_id = wandb_run.id

    (args.output / "run.json").write_text(json.dumps(run_metadata, indent=2, sort_keys=True) + "\n")
    print(
        f"real perception parameters={model.trainable_parameters:,} trainable "
        f"({frozen_parameters:,} frozen); train={len(training):,} frames / "
        f"{len(training.segment_ids)} segments, validation={len(validation):,} frames / "
        f"{len(validation.segment_ids)} segments"
    )
    print(
        f"initialization={model.pretrained_source}; teacher={teacher_hash}; "
        f"objective=lambda*{args.feature_weight}*||dE||^2 + {args.plan_weight}*KL_plan; "
        f"constant-predictor mse: train={train_stats['variance']:.5f}, "
        f"validation={validation_stats['variance']:.5f}"
    )
    if wandb_run is not None:
        print(f"wandb run id={wandb_run.id}; url={wandb_run.url}; mode={args.wandb_mode}")

    metrics_path = args.output / "metrics.jsonl"

    def log_training_step(payload: dict[str, float], step: int) -> None:
        if wandb_run is not None:
            wandb_run.log({"global_step": step, **payload}, step=step)

    run_failed = True
    try:
        for epoch in range(start_epoch, args.epochs):
            started = time.monotonic()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            train_metrics, global_step = _train_epoch(
                model,
                training_loader,
                criterion,
                optimizer,
                scheduler,
                scaler,
                device,
                args.amp,
                args.accumulation_steps,
                args.max_grad_norm,
                epoch,
                global_step,
                args.wandb_log_interval,
                train_stats["variance"],
                log_training_step if wandb_run is not None else None,
            )
            validation_metrics = _validate(
                model, validation_loader, criterion, device, args.amp, validation_stats["variance"]
            )
            elapsed = time.monotonic() - started
            improved = validation_metrics["loss"] < best_validation_loss
            if improved:
                best_validation_loss = validation_metrics["loss"]
            record = {
                "epoch": epoch,
                "global_step": global_step,
                "elapsed_seconds": elapsed,
                "train": train_metrics,
                "validation": validation_metrics,
                "best_validation_loss": best_validation_loss,
            }
            if device.type == "cuda":
                record["peak_gpu_memory_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
            with metrics_path.open("a") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
            payload = _checkpoint(
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                global_step,
                best_validation_loss,
                teacher_hash,
                wandb_run_id,
                args,
            )
            _atomic_torch_save(payload, args.output / "last.pt")
            if improved:
                _atomic_torch_save(payload, args.output / "best.pt")
                _atomic_torch_save(
                    _deployment_bundle(model, teacher_hash, args.architecture),
                    args.output / "deployment.pt",
                )
            if wandb_run is not None:
                wandb_run.log(
                    {"global_step": global_step, **_wandb_epoch_payload(record)},
                    step=global_step,
                )
                wandb_run.summary["best_validation_loss"] = best_validation_loss
                wandb_run.summary["best_checkpoint"] = str((args.output / "best.pt").resolve())
            print(json.dumps(record, sort_keys=True), flush=True)
        run_failed = False
    finally:
        if wandb_run is not None:
            wandb_run.finish(exit_code=1 if run_failed else 0)


if __name__ == "__main__":
    main()
