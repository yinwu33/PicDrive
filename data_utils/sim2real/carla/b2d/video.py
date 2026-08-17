"""Render the agent's frame dumps into annotated per-route videos.

``b2d.agent`` writes one JPEG and one JSON per *policy* tick when
``PICDRIVE_VIZ=1``, so a video at ``--fps 10`` runs at wall-clock speed even
though the simulator ticked at 20 Hz.  The overlay is the same telemetry
``demo.closed_loop``'s live preview draws, which is what makes a failure readable:
a route that ends badly almost always shows the reason in the commanded action
some seconds earlier, not in the frame where the run stopped.

Runs in the repository venv like everything else::

    source scripts/define_env.sh
    "$PICDRIVE_PYTHON" -m data_utils.sim2real.carla.b2d.video \
        --dump artifacts/carla_sim2real/bench2drive/dev10/dump
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


FONT = cv2.FONT_HERSHEY_SIMPLEX
HUD_HEIGHT = 76
# JERK_LONG x JERK_LAT, the 12-way action laid out as (longitudinal, lateral).
JERK_LONG_LABELS = ("brake--", "brake", "coast", "accel")
JERK_LAT_LABELS = ("right", "straight", "left")


def _fourcc(code: str) -> int:
    # OpenCV moved the helper onto VideoWriter in 5.0 and kept the old module
    # level name only in the 4.x line.
    factory = getattr(cv2.VideoWriter, "fourcc", None) or cv2.VideoWriter_fourcc
    return factory(*code)


def _bar(canvas: np.ndarray, x: int, y: int, width: int, value: float, color) -> None:
    """A signed [-1, 1] bar filled from its centre, for steer/throttle/brake."""

    cv2.rectangle(canvas, (x, y), (x + width, y + 10), (60, 60, 60), -1)
    middle = x + width // 2
    end = int(middle + 0.5 * width * float(np.clip(value, -1.0, 1.0)))
    cv2.rectangle(canvas, (min(middle, end), y), (max(middle, end), y + 10), color, -1)


def _annotate(frame: np.ndarray, meta: dict, index: int, total: int) -> np.ndarray:
    canvas = np.zeros((frame.shape[0] + HUD_HEIGHT, frame.shape[1], 3), dtype=np.uint8)
    canvas[: frame.shape[0]] = frame
    top = frame.shape[0]

    action = int(meta.get("action", 0))
    label = f"{JERK_LONG_LABELS[action // 3]}/{JERK_LAT_LABELS[action % 3]}"
    line = (
        f"t {index:4d}/{total}  speed {meta.get('speed', 0.0):5.2f} m/s"
        f"  target {meta.get('target_speed', 0.0):5.2f}"
        f"  action {action:2d} {label}"
    )
    cv2.putText(canvas, line, (8, top + 20), FONT, 0.45, (235, 235, 235), 1, cv2.LINE_AA)

    route = (
        f"route {100.0 * meta.get('route_completion', 0.0):5.1f}%"
        f"  goal {meta.get('goal_index', 0)}/{max(0, meta.get('num_goals', 1) - 1)}"
        f"  deviation {meta.get('route_deviation', 0.0):5.2f} m"
        f"  collisions {meta.get('collisions', 0)}"
    )
    colour = (120, 120, 255) if meta.get("collisions", 0) else (200, 200, 200)
    cv2.putText(canvas, route, (8, top + 40), FONT, 0.45, colour, 1, cv2.LINE_AA)

    cv2.putText(canvas, "steer", (8, top + 64), FONT, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
    _bar(canvas, 58, top + 54, 120, meta.get("steer", 0.0), (255, 200, 90))
    cv2.putText(canvas, "throt", (190, top + 64), FONT, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
    _bar(canvas, 240, top + 54, 120, meta.get("throttle", 0.0), (110, 230, 110))
    cv2.putText(canvas, "brake", (372, top + 64), FONT, 0.4, (180, 180, 180), 1, cv2.LINE_AA)
    _bar(canvas, 422, top + 54, 120, meta.get("brake", 0.0), (110, 110, 240))
    if meta.get("reverse"):
        cv2.putText(canvas, "REVERSE", (556, top + 64), FONT, 0.45, (90, 90, 255), 1, cv2.LINE_AA)
    return canvas


def render_route(route_dir: Path, output: Path, fps: float, source: str) -> int:
    frames = sorted((route_dir / source).glob("*.jpg"))
    if not frames:
        return 0
    first = _annotate(cv2.imread(str(frames[0])), {}, 0, len(frames))
    writer = cv2.VideoWriter(str(output), _fourcc("mp4v"), fps, (first.shape[1], first.shape[0]))
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV refused to open {output} for writing")
    try:
        for index, frame_path in enumerate(frames):
            meta_path = route_dir / "meta" / f"{frame_path.stem}.json"
            meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
            writer.write(_annotate(cv2.imread(str(frame_path)), meta, index, len(frames)))
    finally:
        writer.release()
    return len(frames)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump", required=True, help="SAVE_PATH root, or one route directory")
    parser.add_argument("--out", default=None, help="output directory (default: alongside the dump)")
    parser.add_argument("--fps", type=float, default=10.0, help="policy rate; 10 Hz is wall clock")
    parser.add_argument(
        "--source",
        default="rgb_strip",
        choices=("rgb_strip", "rgb_front"),
        help="rgb_strip is front_left|front|front_right side by side",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    dump = Path(args.dump)
    routes = [dump] if (dump / "meta").is_dir() else sorted(p for p in dump.iterdir() if (p / "meta").is_dir())
    if not routes:
        raise SystemExit(f"no route dumps under {dump}; was the agent run with PICDRIVE_VIZ=1?")
    output_dir = Path(args.out) if args.out else dump
    output_dir.mkdir(parents=True, exist_ok=True)
    for route in routes:
        target = output_dir / f"{route.name}.mp4"
        count = render_route(route, target, args.fps, args.source)
        print(f"{route.name}: {count} frames -> {target}" if count else f"{route.name}: no frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
