"""Preview a lane-derived drivable-area estimate for map-only binaries.

This is an audit tool, not a dataset migration.  It buffers lane centerlines,
optionally fills driveway polygons, and writes three-panel BEV PNGs without
modifying the input binaries.

Example::

    .venv/bin/python data_utils/womd/preview_drivable_area.py \
        --input resources/drive/binaries/training_map_only \
        --output artifacts/drivable_area_preview \
        --map-ids 0 37 284 999
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import struct
from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


ROAD_LANE = 4
ROAD_LINE = 5
ROAD_EDGE = 6
STOP_SIGN = 7
CROSSWALK = 8
SPEED_BUMP = 9
DRIVEWAY = 10

HEADER = struct.Struct("<16siiii")
ENTITY_HEADER = struct.Struct("<iiii")
SCALARS = struct.Struct("<ffffffi")

COLORS = {
    "background": (0, 0, 0),
    "drivable": (82, 84, 88),
    "lane": (76, 255, 105),
    "line": (245, 245, 245),
    "edge": (255, 78, 78),
    "driveway": (65, 155, 255),
    "crosswalk": (255, 178, 62),
    "speed_bump": (255, 235, 75),
}


@dataclass(frozen=True)
class RoadFeature:
    type_id: int
    feature_id: int
    x: np.ndarray
    y: np.ndarray


@dataclass(frozen=True)
class BinaryMap:
    scenario_id: str
    roads: tuple[RoadFeature, ...]


@dataclass(frozen=True)
class RasterTransform:
    min_x: float
    max_y: float
    resolution: float
    width: int
    height: int

    def points(self, feature: RoadFeature) -> list[tuple[int, int]]:
        px = np.rint((feature.x - self.min_x) / self.resolution).astype(np.int32)
        py = np.rint((self.max_y - feature.y) / self.resolution).astype(np.int32)
        return list(zip(px.tolist(), py.tolist()))


def read_map(path: Path) -> BinaryMap:
    """Read the road portion of a loader-compatible map-only binary."""
    data = path.read_bytes()
    if len(data) < HEADER.size:
        raise ValueError(f"{path}: truncated map header")
    scenario_raw, _sdc, num_predict, num_objects, num_roads = HEADER.unpack_from(data)
    if num_predict != 0 or num_objects != 0:
        raise ValueError(
            f"{path}: expected map-only data, found {num_predict} predictions and "
            f"{num_objects} objects"
        )
    if num_roads < 0:
        raise ValueError(f"{path}: negative road count {num_roads}")

    offset = HEADER.size
    roads: list[RoadFeature] = []
    for road_index in range(num_roads):
        if offset + ENTITY_HEADER.size > len(data):
            raise ValueError(f"{path}: truncated road header {road_index}")
        _scenario_index, type_id, feature_id, size = ENTITY_HEADER.unpack_from(data, offset)
        offset += ENTITY_HEADER.size
        if size < 0:
            raise ValueError(f"{path}: negative point count for road {road_index}")
        arrays_size = size * 4
        scalars_end = offset + 3 * arrays_size + SCALARS.size
        if scalars_end > len(data):
            raise ValueError(f"{path}: truncated road {road_index}")
        x = np.frombuffer(data, dtype="<f4", count=size, offset=offset).copy()
        y = np.frombuffer(data, dtype="<f4", count=size, offset=offset + arrays_size).copy()
        roads.append(RoadFeature(type_id, feature_id, x, y))
        offset = scalars_end

    if offset != len(data):
        raise ValueError(f"{path}: {len(data) - offset} trailing bytes")
    scenario_id = scenario_raw.rstrip(b"\0").decode("utf-8", errors="replace")
    return BinaryMap(scenario_id, tuple(roads))


def make_transform(
    roads: Iterable[RoadFeature], resolution: float, margin_m: float
) -> RasterTransform:
    nonempty = [road for road in roads if len(road.x)]
    if not nonempty:
        raise ValueError("map contains no road points")
    min_x = min(float(road.x.min()) for road in nonempty) - margin_m
    max_x = max(float(road.x.max()) for road in nonempty) + margin_m
    min_y = min(float(road.y.min()) for road in nonempty) - margin_m
    max_y = max(float(road.y.max()) for road in nonempty) + margin_m
    width = max(1, int(np.ceil((max_x - min_x) / resolution)) + 1)
    height = max(1, int(np.ceil((max_y - min_y) / resolution)) + 1)
    return RasterTransform(min_x, max_y, resolution, width, height)


def _draw_round_polyline(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[int, int]],
    *,
    fill: int | tuple[int, int, int],
    width: int,
) -> None:
    if not points:
        return
    if len(points) == 1:
        radius = width // 2
        x, y = points[0]
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill)
        return
    draw.line(points, fill=fill, width=width, joint="curve")
    radius = width // 2
    for x, y in points:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=fill)


def build_drivable_mask(
    road_map: BinaryMap,
    transform: RasterTransform,
    *,
    lane_half_width_m: float,
    closing_radius_m: float,
    include_driveways: bool,
) -> Image.Image:
    """Rasterize the deliberately simple first-pass drivable-area heuristic."""
    mask = Image.new("L", (transform.width, transform.height), 0)
    draw = ImageDraw.Draw(mask)
    lane_width_px = max(1, int(round(2 * lane_half_width_m / transform.resolution)))
    for road in road_map.roads:
        points = transform.points(road)
        if road.type_id == ROAD_LANE:
            _draw_round_polyline(draw, points, fill=255, width=lane_width_px)
        elif include_driveways and road.type_id == DRIVEWAY and len(points) >= 3:
            draw.polygon(points, fill=255)

    radius_px = int(round(closing_radius_m / transform.resolution))
    if radius_px:
        filter_size = 2 * radius_px + 1
        mask = mask.filter(ImageFilter.MaxFilter(filter_size))
        mask = mask.filter(ImageFilter.MinFilter(filter_size))
    return mask


def _feature_image(
    road_map: BinaryMap,
    transform: RasterTransform,
    background: Image.Image | None = None,
    *,
    final_style: bool = False,
) -> Image.Image:
    image = background.copy() if background is not None else Image.new(
        "RGB", (transform.width, transform.height), COLORS["background"]
    )
    draw = ImageDraw.Draw(image)
    width = max(1, int(round(0.8 / transform.resolution)))
    layers = (ROAD_LANE, DRIVEWAY, ROAD_EDGE, ROAD_LINE, CROSSWALK, SPEED_BUMP)
    for type_id in layers:
        if final_style and type_id in (ROAD_LANE, DRIVEWAY):
            continue
        for road in road_map.roads:
            if road.type_id != type_id:
                continue
            points = transform.points(road)
            if not points:
                continue
            color = {
                ROAD_LANE: COLORS["lane"],
                ROAD_LINE: COLORS["line"],
                ROAD_EDGE: COLORS["edge"] if not final_style else (155, 158, 164),
                DRIVEWAY: COLORS["driveway"],
                CROSSWALK: COLORS["crosswalk"] if not final_style else COLORS["line"],
                SPEED_BUMP: COLORS["speed_bump"],
            }[type_id]
            if type_id in (CROSSWALK, SPEED_BUMP) and len(points) >= 3:
                draw.polygon(points, outline=color, width=width)
            else:
                draw.line(points, fill=color, width=width, joint="curve")
    return image


def _mask_rgb(mask: Image.Image) -> Image.Image:
    pixels = np.zeros((mask.height, mask.width, 3), dtype=np.uint8)
    pixels[np.asarray(mask) > 0] = COLORS["drivable"]
    return Image.fromarray(pixels, mode="RGB")


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _fit_panel(image: Image.Image, size: int) -> Image.Image:
    scale = min(size / image.width, size / image.height)
    resized = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.NEAREST,
    )
    panel = Image.new("RGB", (size, size), (17, 18, 21))
    panel.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2))
    return panel


def render_audit_png(
    road_map: BinaryMap,
    transform: RasterTransform,
    mask: Image.Image,
    destination: Path,
    *,
    map_name: str,
    lane_half_width_m: float,
    closing_radius_m: float,
    include_driveways: bool,
    panel_size: int,
) -> None:
    mask_rgb = _mask_rgb(mask)
    panels = (
        _feature_image(road_map, transform),
        _feature_image(road_map, transform, mask_rgb),
        _feature_image(road_map, transform, mask_rgb, final_style=True),
    )
    labels = ("1  INPUT MAP FEATURES", "2  ESTIMATE + AUDIT", "3  PROPOSED BEV")
    title_h, label_h, footer_h, gap = 62, 38, 82, 12
    width = 3 * panel_size + 4 * gap
    height = title_h + label_h + panel_size + footer_h
    output = Image.new("RGB", (width, height), (17, 18, 21))
    draw = ImageDraw.Draw(output)
    title_font, label_font, footer_font = _font(22), _font(16), _font(14)
    title = (
        f"{map_name}   scenario={road_map.scenario_id}   "
        f"grid={transform.resolution:g} m   lane half-width={lane_half_width_m:g} m   "
        f"closing={closing_radius_m:g} m   driveways={'on' if include_driveways else 'off'}"
    )
    draw.text((gap, 17), title, font=title_font, fill=(238, 239, 242))
    for index, (panel, label) in enumerate(zip(panels, labels)):
        x = gap + index * (panel_size + gap)
        draw.text((x, title_h + 8), label, font=label_font, fill=(210, 213, 219))
        output.paste(_fit_panel(panel, panel_size), (x, title_h + label_h))

    legend_y = title_h + label_h + panel_size + 17
    legend = (
        ("lane center (buffer source)", COLORS["lane"]),
        ("road line", COLORS["line"]),
        ("road edge (audit only)", COLORS["edge"]),
        ("driveway", COLORS["driveway"]),
        ("drivable estimate", COLORS["drivable"]),
    )
    cursor = gap
    for text, color in legend:
        draw.rectangle((cursor, legend_y + 2, cursor + 18, legend_y + 16), fill=color)
        draw.text((cursor + 25, legend_y), text, font=footer_font, fill=(210, 213, 219))
        cursor += 25 + int(draw.textlength(text, font=footer_font)) + 24
    coverage = 100.0 * np.count_nonzero(np.asarray(mask)) / mask.size[0] / mask.size[1]
    draw.text(
        (gap, legend_y + 30),
        f"Gray coverage inside displayed bounds: {coverage:.1f}%  |  "
        "Road edges are shown for visual validation and do not clip the estimate.",
        font=footer_font,
        fill=(151, 156, 166),
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.save(destination, format="PNG", optimize=True)


def _resolve_maps(input_dir: Path, ids: list[int]) -> list[Path]:
    paths = [input_dir / f"map_{map_id:03d}.bin" for map_id in ids]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("map files not found: " + ", ".join(missing))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="map-only binary directory")
    parser.add_argument("--output", type=Path, required=True, help="PNG output directory")
    parser.add_argument("--map-ids", type=int, nargs="+", default=[0])
    parser.add_argument("--resolution", type=float, default=0.5, help="meters per mask pixel")
    parser.add_argument("--lane-half-width", type=float, default=2.25, help="meters")
    parser.add_argument("--closing-radius", type=float, default=1.0, help="meters")
    parser.add_argument("--margin", type=float, default=8.0, help="meters around map bounds")
    parser.add_argument("--panel-size", type=int, default=620, help="pixels per audit panel")
    parser.add_argument(
        "--no-driveways", action="store_true", help="do not add driveway polygons to the mask"
    )
    args = parser.parse_args()
    if args.resolution <= 0 or args.lane_half_width <= 0:
        parser.error("--resolution and --lane-half-width must be > 0")
    if args.closing_radius < 0 or args.margin < 0 or args.panel_size < 128:
        parser.error("closing/margin must be >= 0 and panel-size must be >= 128")

    paths = _resolve_maps(args.input, args.map_ids)
    for path in paths:
        road_map = read_map(path)
        transform = make_transform(road_map.roads, args.resolution, args.margin)
        mask = build_drivable_mask(
            road_map,
            transform,
            lane_half_width_m=args.lane_half_width,
            closing_radius_m=args.closing_radius,
            include_driveways=not args.no_driveways,
        )
        destination = args.output / f"{path.stem}_drivable_preview.png"
        render_audit_png(
            road_map,
            transform,
            mask,
            destination,
            map_name=path.stem,
            lane_half_width_m=args.lane_half_width,
            closing_radius_m=args.closing_radius,
            include_driveways=not args.no_driveways,
            panel_size=args.panel_size,
        )
        print(destination)


if __name__ == "__main__":
    main()
