"""Render processed Waymo samples as inspectable 3x3 PNG grids.

Rows are fixed by the audit contract:

1. PufferDrive abstract render for front-left/front/front-right.
2. The paired real Waymo images in the same order.
3. An alpha overlay of the two rows.

The sim row is rendered directly from the compact scene arrays in each processed
NPZ.  A teacher-feature directory may still be supplied to reuse its cached sim
images, but teacher features are not required for visualization.  The bulk path
uses Pillow directly so tens of thousands of PNGs can be created without
starting a Matplotlib figure for every frame. This script never reads raw
TFRecord data.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .processed import DISPLAY_CAMERA_NAMES, list_processed_files, load_feature, load_processed
from .render_roads import prepare_runtime_roads


class _PackedCamera:
    """Camera adapter for one row of the exact per-segment packed rig."""

    def __init__(self, name: str, row: np.ndarray) -> None:
        if row.shape != (20,):
            raise ValueError(f"camera {name} has packed rig shape {row.shape}, expected (20,)")
        self.name = name
        self._rotation = np.asarray(row[:9], dtype=np.float32).reshape(3, 3).copy()
        self.pos = tuple(float(value) for value in row[9:12])
        self._intrinsics = tuple(float(value) for value in row[12:16])
        self.width = round(float(row[16]))
        self.height = round(float(row[17]))
        self.near = float(row[18])
        self.far = float(row[19])

    def rotation(self):
        import torch

        return torch.from_numpy(self._rotation)

    def intrinsics(self) -> tuple[float, float, float, float]:
        return self._intrinsics


def _render_sim_images(processed: dict[str, np.ndarray], device: str = "auto") -> np.ndarray:
    """Render uint8 [camera,H,W,RGB] views directly from a processed sample."""
    import torch

    if device not in {"auto", "cpu", "cuda"}:
        raise ValueError(f"device must be auto, cpu, or cuda; got {device!r}")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA rendering was requested, but CUDA is unavailable")
    render_device = torch.device(
        "cuda" if device == "cuda" or (device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    names = [str(value) for value in processed["camera_names"].tolist()]
    rig_array = np.ascontiguousarray(processed["rig"], dtype=np.float32)
    cameras = [_PackedCamera(name, row) for name, row in zip(names, rig_array)]
    agents = torch.from_numpy(np.ascontiguousarray(processed["agents"])).to(render_device)
    roads = torch.from_numpy(prepare_runtime_roads(processed["roads"])).to(render_device)
    egos = torch.from_numpy(np.ascontiguousarray(processed["ego"][None])).to(render_device)

    with torch.inference_mode():
        if render_device.type == "cuda":
            from pufferlib.ocean.drive import raster_cuda

            images = raster_cuda.render(
                agents=agents,
                roads=roads,
                egos=egos,
                cameras=cameras,
                rig=torch.from_numpy(rig_array).to(render_device),
            )
        else:
            from pufferlib.ocean.drive import raster_ref

            images = raster_ref.render(agents, roads, egos, cameras=cameras)
    return images[0].permute(0, 2, 3, 1).contiguous().cpu().numpy()


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _centered_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> None:
    bounds = draw.textbbox((0, 0), text, font=font)
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    draw.text((xy[0] - width / 2, xy[1] - height / 2), text, fill="black", font=font)


def compose_grid(
    processed_path: Path,
    feature_path: Path | None = None,
    alpha: float = 0.5,
    *,
    device: str = "auto",
) -> tuple[Image.Image, str, int]:
    """Load a processed sample and return its RGB audit grid and identity."""
    processed = load_processed(processed_path)
    processed_id = str(np.asarray(processed["segment_id"]).item())
    processed_time = int(np.asarray(processed["timestamp_micros"]).item())
    if feature_path is None:
        sim_arrays = _render_sim_images(processed, device)
    else:
        feature = load_feature(feature_path)
        feature_id = str(np.asarray(feature["segment_id"]).item())
        feature_time = int(np.asarray(feature["timestamp_micros"]).item())
        if (processed_id, processed_time) != (feature_id, feature_time):
            raise ValueError(
                f"processed/feature identity mismatch: {processed_id}@{processed_time} vs {feature_id}@{feature_time}"
            )
        sim_arrays = feature["sim_images"]

    policy_names = [str(value) for value in processed["camera_names"].tolist()]
    order = [policy_names.index(name) for name in DISPLAY_CAMERA_NAMES]
    real_arrays = processed["real_images"][order]
    sim_arrays = sim_arrays[order]
    height, width = real_arrays.shape[1:3]
    real = [Image.fromarray(image, mode="RGB") for image in real_arrays]
    sim = [
        Image.fromarray(image, mode="RGB").resize((width, height), Image.Resampling.BILINEAR) for image in sim_arrays
    ]
    overlay = [Image.blend(real_image, sim_image, alpha) for real_image, sim_image in zip(real, sim)]

    left_margin = 128
    top_margin = 64
    canvas = Image.new("RGB", (left_margin + 3 * width, top_margin + 3 * height), "white")
    for row, images in enumerate((sim, real, overlay)):
        for column, image in enumerate(images):
            canvas.paste(image, (left_margin + column * width, top_margin + row * height))

    draw = ImageDraw.Draw(canvas)
    label_font = _font(16)
    title_font = _font(14)
    metadata_font = _font(13)
    draw.text(
        (8, 5),
        f"{processed_id}  timestamp={processed_time}",
        fill="black",
        font=metadata_font,
    )
    for column, title in enumerate(("Front Left", "Front", "Front Right")):
        _centered_text(
            draw,
            (left_margin + (column + 0.5) * width, 46),
            title,
            title_font,
        )
    for row, label in enumerate(("Sim render", "Waymo real", f"Overlay α={alpha:g}")):
        _centered_text(draw, (left_margin / 2, top_margin + (row + 0.5) * height), label, label_font)
    return canvas, processed_id, processed_time


def plot_sample(
    processed_path: Path,
    feature_path: Path | None,
    output: Path,
    alpha: float = 0.5,
    *,
    verbose: bool = True,
    device: str = "auto",
) -> None:
    canvas, _, _ = compose_grid(processed_path, feature_path, alpha, device=device)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    try:
        canvas.save(temporary, format="PNG", compress_level=3)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    if verbose:
        print(f"wrote {output}")


def _feature_pairs(processed: Path, features: Path, max_samples: int | None) -> list[tuple[Path, Path]]:
    manifest = features / "manifest.json"
    pairs: list[tuple[Path, Path]] = []
    if manifest.is_file():
        metadata = json.loads(manifest.read_text())
        for entry in metadata.get("samples", []):
            pairs.append((processed / entry["processed_file"], features / entry["file"]))
    else:
        for feature_path in sorted(features.rglob("*.npz")):
            relative = feature_path.relative_to(features)
            processed_path = processed / relative
            if not processed_path.is_file():
                processed_path = processed / feature_path.name
            pairs.append((processed_path, feature_path))
    pairs = [pair for pair in pairs if pair[0].is_file() and pair[1].is_file()]
    return pairs if max_samples is None else pairs[:max_samples]


def _render_bulk_job(job: tuple[Path, Path | None, Path, float, bool, bool, bool, str]) -> tuple[Path, bool]:
    processed_path, feature_path, output_dir, alpha, resume, overwrite, flat, device = job
    source = feature_path if feature_path is not None else processed_path
    segment_id = source.stem.rsplit("__", 1)[0]
    destination = output_dir / f"{source.stem}.png"
    if not flat:
        destination = output_dir / segment_id / destination.name
    if destination.exists() and resume and not overwrite:
        return destination, False
    if destination.exists() and not overwrite:
        raise FileExistsError(f"{destination} exists; pass --resume to keep it or --overwrite to replace it")
    plot_sample(processed_path, feature_path, destination, alpha, verbose=False, device=device)
    return destination, True


def _sample_pairs(processed: Path, features: Path | None, max_samples: int | None) -> list[tuple[Path, Path | None]]:
    if features is not None:
        return _feature_pairs(processed, features, max_samples)
    files = list_processed_files(processed, max_samples)
    return [(path, None) for path in files]


def _resolve_sample(processed: Path, features: Path | None, sample: str | None) -> tuple[Path, Path | None]:
    if sample is None:
        pairs = _sample_pairs(processed, features, 1)
        if not pairs:
            raise FileNotFoundError(f"no processed .npz files under {processed}")
        return pairs[0]
    name = Path(sample).name
    if not name.endswith(".npz"):
        name += ".npz"
    return processed / name, features / name if features is not None else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, required=True)
    parser.add_argument(
        "--features",
        type=Path,
        help="Optional teacher-feature cache containing sim_images",
    )
    parser.add_argument("--sample", help="Sample stem or .npz filename; defaults to first sample")
    outputs = parser.add_mutually_exclusive_group(required=True)
    outputs.add_argument("--output", type=Path, help="Write one PNG")
    outputs.add_argument("--output-dir", type=Path, help="Write every matched frame as PNG")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--flat",
        action="store_true",
        help="Write bulk PNGs directly in output-dir instead of per-segment directories",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        parser.error("--alpha must be between 0 and 1")
    workers = args.workers
    if workers is None:
        workers = 1 if args.features is None else min(8, os.cpu_count() or 1)
    if workers < 1:
        parser.error("--workers must be >= 1")
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive")

    if args.output is not None:
        processed_path, feature_path = _resolve_sample(args.processed, args.features, args.sample)
        if not processed_path.is_file():
            parser.error(f"processed sample does not exist: {processed_path}")
        if feature_path is not None and not feature_path.is_file():
            parser.error(f"feature sample does not exist: {feature_path}")
        if args.output.exists() and args.resume:
            print(f"kept {args.output}")
            return
        if args.output.exists() and not args.overwrite:
            parser.error(f"{args.output} exists; pass --resume or --overwrite")
        plot_sample(processed_path, feature_path, args.output, args.alpha, device=args.device)
        return

    if args.sample is not None:
        parser.error("--sample is only valid with --output")
    pairs = _sample_pairs(args.processed, args.features, args.max_samples)
    if not pairs:
        parser.error("no processed samples found")
    assert args.output_dir is not None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        (
            processed,
            feature,
            args.output_dir,
            args.alpha,
            args.resume,
            args.overwrite,
            args.flat,
            args.device,
        )
        for processed, feature in pairs
    ]
    written = 0
    if workers == 1:
        results = map(_render_bulk_job, jobs)
        for index, (_, changed) in enumerate(results, 1):
            written += int(changed)
            if index % 500 == 0 or index == len(jobs):
                print(f"[{index}/{len(jobs)}] PNGs; wrote {written}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = executor.map(_render_bulk_job, jobs, chunksize=8)
            for index, (_, changed) in enumerate(results, 1):
                written += int(changed)
                if index % 500 == 0 or index == len(jobs):
                    print(f"[{index}/{len(jobs)}] PNGs; wrote {written}", flush=True)
    print(f"created {written} PNGs, kept {len(jobs) - written}, under {args.output_dir}")


if __name__ == "__main__":
    main()
