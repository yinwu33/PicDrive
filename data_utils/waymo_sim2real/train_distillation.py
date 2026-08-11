"""Train real-image perception against frozen PufferDrive scene features.

The simulation teacher is intentionally absent from the optimizer and from the
training process. Its frozen outputs have already been extracted and pinned by
checkpoint SHA256 in ``teacher_features/manifest.json``. This makes accidental
teacher updates impossible and avoids rendering abstract scenes during student
training.

Example:

    .venv/bin/python -m data_utils.waymo_sim2real.train_distillation \
        --root artifacts/waymo_sim2real/full \
        --output artifacts/waymo_sim2real/runs/real_perception_convnext_tiny
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .processed import CAMERA_NAMES
from .real_perception import FeatureAlignmentLoss, RealPerception, RealPerceptionConfig


class PairedWaymoFeatureDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Strict processed-image/teacher-feature pairing backed by manifests."""

    def __init__(self, split_root: str | Path, max_samples: int | None = None):
        self.split_root = Path(split_root)
        self.processed_dir = self.split_root / "processed"
        self.features_dir = self.split_root / "teacher_features"
        processed_manifest = self.processed_dir / "manifest.jsonl"
        feature_manifest = self.features_dir / "manifest.json"
        if not processed_manifest.is_file():
            raise FileNotFoundError(f"missing processed manifest: {processed_manifest}")
        if not feature_manifest.is_file():
            raise FileNotFoundError(f"missing teacher manifest: {feature_manifest}")

        processed_entries = [json.loads(line) for line in processed_manifest.read_text().splitlines() if line]
        processed_by_file = {entry["file"]: entry for entry in processed_entries}
        if len(processed_by_file) != len(processed_entries):
            raise ValueError(f"{processed_manifest} contains duplicate files")

        metadata = json.loads(feature_manifest.read_text())
        if tuple(metadata.get("camera_names", ())) != CAMERA_NAMES:
            raise ValueError(
                f"{feature_manifest} camera order {metadata.get('camera_names')} does not match {CAMERA_NAMES}"
            )
        if int(metadata.get("feature_dim", -1)) != 256:
            raise ValueError(f"{feature_manifest} does not contain 256-D teacher features")
        self.teacher_checkpoint_sha256 = str(metadata.get("checkpoint_sha256", ""))
        if len(self.teacher_checkpoint_sha256) != 64:
            raise ValueError(f"{feature_manifest} has an invalid checkpoint SHA256")

        pairs: list[tuple[Path, Path]] = []
        seen_processed: set[str] = set()
        for entry in metadata.get("samples", ()):
            processed_name = entry["processed_file"]
            feature_name = entry["file"]
            if processed_name not in processed_by_file:
                raise ValueError(f"teacher sample references unknown processed file {processed_name}")
            if processed_name in seen_processed:
                raise ValueError(f"teacher manifest pairs {processed_name} more than once")
            seen_processed.add(processed_name)
            processed_path = self.processed_dir / processed_name
            feature_path = self.features_dir / feature_name
            if not processed_path.is_file() or not feature_path.is_file():
                raise FileNotFoundError(f"missing pair: {processed_path}, {feature_path}")
            pairs.append((processed_path, feature_path))
        if len(pairs) != len(processed_entries):
            raise ValueError(
                f"pair count {len(pairs)} does not match processed count {len(processed_entries)}"
            )
        if max_samples is not None:
            if max_samples < 1:
                raise ValueError("max_samples must be positive")
            pairs = pairs[:max_samples]
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        processed_path, feature_path = self.pairs[index]
        with np.load(processed_path, allow_pickle=False) as archive:
            images = np.asarray(archive["real_images"])
        with np.load(feature_path, allow_pickle=False) as archive:
            target = np.asarray(archive["teacher_feature"])
            saved_hash = str(np.asarray(archive["checkpoint_sha256"]).item())
        if images.shape != (3, 256, 384, 3) or images.dtype != np.uint8:
            raise ValueError(f"{processed_path} has invalid real_images {images.shape} {images.dtype}")
        if target.shape != (256,) or target.dtype not in (np.float16, np.float32, np.float64):
            raise ValueError(f"{feature_path} has invalid teacher_feature {target.shape} {target.dtype}")
        if saved_hash != self.teacher_checkpoint_sha256:
            raise ValueError(
                f"{feature_path} uses teacher {saved_hash}, expected {self.teacher_checkpoint_sha256}"
            )
        # Both arrays originate in read-only zip buffers. Copy before converting
        # so worker tensors own writable, contiguous storage.
        image_tensor = torch.from_numpy(images.copy()).permute(0, 3, 1, 2)
        target_tensor = torch.from_numpy(target.astype(np.float32, copy=True))
        return image_tensor, target_tensor


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


def _averages(sums: dict[str, float]) -> dict[str, float]:
    samples = sums.pop("samples")
    return {name: value / samples for name, value in sums.items()}


def _train_epoch(
    model: RealPerception,
    loader: DataLoader,
    criterion: FeatureAlignmentLoss,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp: str,
    accumulation_steps: int,
    max_grad_norm: float,
) -> dict[str, float]:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    sums: dict[str, float] = {}
    for batch_index, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        images = _photometric_augment(images)
        group_start = (batch_index // accumulation_steps) * accumulation_steps
        group_size = min(accumulation_steps, len(loader) - group_start)
        with _amp_context(device, amp):
            predictions = model(images)
            loss, metrics = criterion(predictions, targets)
            backward_loss = loss / group_size
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite training loss at batch {batch_index}: {float(loss)}")
        scaler.scale(backward_loss).backward()
        should_step = (batch_index + 1) % accumulation_steps == 0 or batch_index + 1 == len(loader)
        if should_step:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            if not torch.isfinite(grad_norm):
                raise FloatingPointError(f"non-finite gradient norm at batch {batch_index}")
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
        _reduce_metrics(sums, metrics, images.shape[0])
    result = _averages(sums)
    for group in optimizer.param_groups:
        scope = str(group["group_name"]).split("_", 1)[0]
        result[f"{scope}_lr"] = float(group["lr"])
    return result


@torch.inference_mode()
def _validate(
    model: RealPerception,
    loader: DataLoader,
    criterion: FeatureAlignmentLoss,
    device: torch.device,
    amp: str,
) -> dict[str, float]:
    model.eval()
    sums: dict[str, float] = {}
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        with _amp_context(device, amp):
            predictions = model(images)
            loss, metrics = criterion(predictions, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite validation loss")
        _reduce_metrics(sums, metrics, images.shape[0])
    return _averages(sums)


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
    model: RealPerception,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    global_step: int,
    best_validation_loss: float,
    teacher_hash: str,
    args: argparse.Namespace,
) -> dict[str, object]:
    return {
        "schema_version": 1,
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
        "args": vars(args),
    }


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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("artifacts/waymo_sim2real/full"))
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/waymo_sim2real/runs/real_perception_convnext_tiny")
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--backbone-learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--warmup-fraction", type=float, default=0.05)
    parser.add_argument("--cosine-weight", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--amp", choices=("bf16", "fp16", "off"), default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--random-init",
        action="store_true",
        help="Disable ImageNet initialization (intended only as an ablation)",
    )
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-validation-samples", type=int)
    args = parser.parse_args(argv)

    for name in ("epochs", "batch_size", "accumulation_steps", "prefetch_factor"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be >= 1")
    if args.workers < 0:
        parser.error("--workers must be >= 0")
    if not 0 <= args.warmup_fraction < 1:
        parser.error("--warmup-fraction must be in [0, 1)")
    if args.learning_rate <= 0 or args.backbone_learning_rate <= 0:
        parser.error("learning rates must be positive")
    if args.device == "cpu" and args.amp != "off":
        parser.error("CPU training requires --amp off")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    _seed_everything(args.seed)

    training = PairedWaymoFeatureDataset(args.root / "training", args.max_train_samples)
    validation = PairedWaymoFeatureDataset(args.root / "validation", args.max_validation_samples)
    if training.teacher_checkpoint_sha256 != validation.teacher_checkpoint_sha256:
        raise ValueError("training and validation features were extracted from different teachers")
    teacher_hash = training.teacher_checkpoint_sha256
    training_loader = _loader(training, args, training=True)
    validation_loader = _loader(validation, args, training=False)

    # A resume checkpoint overwrites the entire model, so it must not require a
    # second network download. Fresh runs use official ImageNet weights unless
    # the explicit ablation flag is provided.
    model = RealPerception(pretrained=not args.random_init and args.resume is None).to(device)
    criterion = FeatureAlignmentLoss(args.cosine_weight)
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
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp == "fp16")

    start_epoch = 0
    global_step = 0
    best_validation_loss = math.inf
    if args.resume is not None:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        if state.get("teacher_checkpoint_sha256") != teacher_hash:
            raise ValueError("resume checkpoint was trained against a different teacher")
        if state.get("model_config") != model.config.to_dict():
            raise ValueError("resume checkpoint architecture does not match")
        model.load_state_dict(state["model"], strict=True)
        model.pretrained_source = str(state.get("backbone_initialization", "unknown"))
        model.pretrained_sha256 = state.get("backbone_checkpoint_sha256")
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = int(state["epoch"]) + 1
        global_step = int(state["global_step"])
        best_validation_loss = float(state["best_validation_loss"])

    args.output.mkdir(parents=True, exist_ok=True)
    run_metadata = {
        "model_config": model.config.to_dict(),
        "trainable_parameters": model.trainable_parameters,
        "teacher_checkpoint_sha256": teacher_hash,
        "backbone_initialization": model.pretrained_source,
        "backbone_checkpoint_sha256": model.pretrained_sha256,
        "training_samples": len(training),
        "validation_samples": len(validation),
        "camera_names": list(CAMERA_NAMES),
        "args": {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()},
    }
    (args.output / "run.json").write_text(json.dumps(run_metadata, indent=2, sort_keys=True) + "\n")
    print(
        f"real perception parameters={model.trainable_parameters:,}; "
        f"train={len(training):,}, validation={len(validation):,}; "
        f"initialization={model.pretrained_source}; teacher={teacher_hash}"
    )

    metrics_path = args.output / "metrics.jsonl"
    for epoch in range(start_epoch, args.epochs):
        started = time.monotonic()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        train_metrics = _train_epoch(
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
        )
        global_step += optimizer_steps_per_epoch
        validation_metrics = _validate(model, validation_loader, criterion, device, args.amp)
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
            args,
        )
        _atomic_torch_save(payload, args.output / "last.pt")
        if improved:
            _atomic_torch_save(payload, args.output / "best.pt")
        print(json.dumps(record, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
