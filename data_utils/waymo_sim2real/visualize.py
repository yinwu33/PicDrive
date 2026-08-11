"""Render processed/teacher pairs as inspectable 3x3 PNG grids.

Rows are fixed by the audit contract:

1. PufferDrive abstract render for front-left/front/front-right.
2. The paired real Waymo images in the same order.
3. An alpha overlay of the two rows.

The bulk path uses Pillow directly so tens of thousands of PNGs can be created
without starting a Matplotlib figure for every frame. This script never reads
raw TFRecord data.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .processed import DISPLAY_CAMERA_NAMES, load_feature, load_processed


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
    processed_path: Path, feature_path: Path, alpha: float = 0.5
) -> tuple[Image.Image, str, int]:
    """Load a matched pair and return its RGB audit grid and identity."""
    processed = load_processed(processed_path)
    feature = load_feature(feature_path)
    processed_id = str(np.asarray(processed["segment_id"]).item())
    feature_id = str(np.asarray(feature["segment_id"]).item())
    processed_time = int(np.asarray(processed["timestamp_micros"]).item())
    feature_time = int(np.asarray(feature["timestamp_micros"]).item())
    if (processed_id, processed_time) != (feature_id, feature_time):
        raise ValueError(
            "processed/feature identity mismatch: "
            f"{processed_id}@{processed_time} vs {feature_id}@{feature_time}"
        )

    policy_names = [str(value) for value in processed["camera_names"].tolist()]
    order = [policy_names.index(name) for name in DISPLAY_CAMERA_NAMES]
    real_arrays = processed["real_images"][order]
    sim_arrays = feature["sim_images"][order]
    height, width = real_arrays.shape[1:3]
    real = [Image.fromarray(image, mode="RGB") for image in real_arrays]
    sim = [
        Image.fromarray(image, mode="RGB").resize((width, height), Image.Resampling.BILINEAR)
        for image in sim_arrays
    ]
    overlay = [Image.blend(real_image, sim_image, alpha) for real_image, sim_image in zip(real, sim)]

    left_margin = 112
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
    feature_path: Path,
    output: Path,
    alpha: float = 0.5,
    *,
    verbose: bool = True,
) -> None:
    canvas, _, _ = compose_grid(processed_path, feature_path, alpha)
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


def _feature_pairs(
    processed: Path, features: Path, max_samples: int | None
) -> list[tuple[Path, Path]]:
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


def _render_bulk_job(
    job: tuple[Path, Path, Path, float, bool, bool]
) -> tuple[Path, bool]:
    processed_path, feature_path, output_dir, alpha, resume, overwrite = job
    segment_id = feature_path.stem.rsplit("__", 1)[0]
    destination = output_dir / segment_id / f"{feature_path.stem}.png"
    if destination.exists() and resume and not overwrite:
        return destination, False
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"{destination} exists; pass --resume to keep it or --overwrite to replace it"
        )
    plot_sample(processed_path, feature_path, destination, alpha, verbose=False)
    return destination, True


def _resolve_sample(processed: Path, features: Path, sample: str | None) -> tuple[Path, Path]:
    if sample is None:
        pairs = _feature_pairs(processed, features, 1)
        if not pairs:
            raise FileNotFoundError(f"no matched feature .npz files under {features}")
        return pairs[0]
    name = Path(sample).name
    if not name.endswith(".npz"):
        name += ".npz"
    return processed / name, features / name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--sample", help="Sample stem or .npz filename; defaults to first feature")
    outputs = parser.add_mutually_exclusive_group(required=True)
    outputs.add_argument("--output", type=Path, help="Write one PNG")
    outputs.add_argument("--output-dir", type=Path, help="Write every matched frame as PNG")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        parser.error("--alpha must be between 0 and 1")
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive")

    if args.output is not None:
        processed_path, feature_path = _resolve_sample(args.processed, args.features, args.sample)
        if not processed_path.is_file():
            parser.error(f"processed sample does not exist: {processed_path}")
        if not feature_path.is_file():
            parser.error(f"feature sample does not exist: {feature_path}")
        if args.output.exists() and args.resume:
            print(f"kept {args.output}")
            return
        if args.output.exists() and not args.overwrite:
            parser.error(f"{args.output} exists; pass --resume or --overwrite")
        plot_sample(processed_path, feature_path, args.output, args.alpha)
        return

    if args.sample is not None:
        parser.error("--sample is only valid with --output")
    pairs = _feature_pairs(args.processed, args.features, args.max_samples)
    if not pairs:
        parser.error("no matched processed/feature samples found")
    assert args.output_dir is not None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        (processed, feature, args.output_dir, args.alpha, args.resume, args.overwrite)
        for processed, feature in pairs
    ]
    written = 0
    if args.workers == 1:
        results = map(_render_bulk_job, jobs)
        for index, (_, changed) in enumerate(results, 1):
            written += int(changed)
            if index % 500 == 0 or index == len(jobs):
                print(f"[{index}/{len(jobs)}] PNGs; wrote {written}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            results = executor.map(_render_bulk_job, jobs, chunksize=8)
            for index, (_, changed) in enumerate(results, 1):
                written += int(changed)
                if index % 500 == 0 or index == len(jobs):
                    print(f"[{index}/{len(jobs)}] PNGs; wrote {written}", flush=True)
    print(f"created {written} PNGs, kept {len(jobs) - written}, under {args.output_dir}")


if __name__ == "__main__":
    main()
