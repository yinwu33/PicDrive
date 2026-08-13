"""CARLA road networks as the simulator's canonical road primitives.

The repository has CARLA road geometry checked in under ``data_utils/carla``,
but it was baked by an external tool, carries no crosswalks, and -- decisively --
comes from a different coordinate pipeline than the actors a live server
reports.  Extracting the network from the same ``carla.Map`` that the recorded
actors come from is what makes the pairing exact, so that is what this does.

Output rows are the canonical ``[x0, y0, x1, y1, width, type]`` of
``drive.h:151`` with type ids ``ROAD_LANE 4 / ROAD_LINE 5 / ROAD_EDGE 6 /
CROSSWALK 8`` -- *not* the renderer-only 11/12 tags, which
``render_roads.prepare_runtime_roads`` applies later, exactly as it does for
Waymo.  The width column is written for completeness and then overwritten by
that same pass, which is why CARLA's real 3.5-4.0 m lanes still render as the
4.5 m strips the policy trained on.

CARLA's world is left-handed with y pointing right; OpenDRIVE and this
repository are right-handed.  Every world quantity therefore negates y, and
``world_to_ego`` below is the only place that convention is applied.
"""

from __future__ import annotations

import math

import numpy as np

from data_utils.waymo_sim2real.render_roads import CROSSWALK, ROAD_EDGE, ROAD_LANE, ROAD_LINE

# Canonical widths at extraction time, matching preprocess.ROAD_WIDTH. Only the
# painted ones survive: prepare_runtime_roads rewrites every width it renders.
ROAD_WIDTH = {ROAD_LANE: 4.5, ROAD_LINE: 0.15, ROAD_EDGE: 0.25, CROSSWALK: 0.50}

# A boundary marking that carries no paint. Anything else is a road line.
UNPAINTED_MARKINGS = frozenset({"NONE", "Other"})
# Markings that *are* the edge of the drivable surface regardless of what lane
# sits beyond them.
EDGE_MARKINGS = frozenset({"Curb", "Grass"})

# Consecutive waypoints in one lane are `resolution` apart; a larger jump means
# the lane was interrupted rather than continued, and joining across it would
# paint a stripe through a junction.
GAP_FACTOR = 3.0

# Endpoint quantisation for deduplicating the boundary two adjacent lanes share.
DEDUP_QUANTUM = 0.05


def world_to_ego(points: np.ndarray, center: np.ndarray, yaw: float) -> np.ndarray:
    """Rotate/translate right-handed world xy into the ego box-centre frame.

    Mirrors the kernel's own transform (``raster.cu:179``) and the Waymo reader's
    ``_roads_in_ego_frame``: x forward along the ego heading, y to its left.
    """
    points = np.asarray(points, dtype=np.float64)
    delta = points - np.asarray(center, dtype=np.float64)
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    return np.stack(
        [
            delta[..., 0] * cos_yaw + delta[..., 1] * sin_yaw,
            -delta[..., 0] * sin_yaw + delta[..., 1] * cos_yaw,
        ],
        axis=-1,
    )


def _chain(points: np.ndarray, type_id: int, max_gap: float) -> list[list[float]]:
    """Expand an ordered polyline into segments, breaking at discontinuities."""
    width = ROAD_WIDTH[type_id]
    rows = []
    for start, end in zip(points[:-1], points[1:]):
        step = math.hypot(end[0] - start[0], end[1] - start[1])
        if step < 1e-6 or step > max_gap:
            continue
        rows.append([start[0], start[1], end[0], end[1], width, float(type_id)])
    return rows


def _boundary_type(neighbour, marking) -> int | None:
    """Classify one side of a lane: drivable-surface edge, painted line, or nothing.

    Neighbour type decides first.  In CARLA the outer side of a driving lane
    typically has ``LaneMarkingType.NONE`` yet borders a Shoulder -- that is the
    edge of the drivable surface and must render as one, or the policy sees a
    road that runs off into the black ground with no boundary.
    """
    marking_type = str(marking.type) if marking is not None else "NONE"
    if marking_type in EDGE_MARKINGS:
        return ROAD_EDGE
    if neighbour is None or str(neighbour.lane_type) != "Driving":
        return ROAD_EDGE
    if marking_type in UNPAINTED_MARKINGS:
        return None
    return ROAD_LINE


def town_roads(carla_map, resolution: float = 1.0) -> np.ndarray:
    """Return one town's ``[M, 6]`` road primitives in the right-handed world frame.

    ``carla_map`` is a ``carla.Map``.  It can be built from an ``.xodr`` string
    without a running server (``carla.Map(name, xodr_content)``), which is what
    makes this testable offline.
    """
    if resolution <= 0:
        raise ValueError(f"resolution must be positive, got {resolution}")
    max_gap = resolution * GAP_FACTOR

    lanes: dict[tuple[int, int, int], list] = {}
    for waypoint in carla_map.generate_waypoints(resolution):
        if str(waypoint.lane_type) != "Driving":
            continue
        lanes.setdefault((waypoint.road_id, waypoint.section_id, waypoint.lane_id), []).append(waypoint)

    rows: list[list[float]] = []
    # Painted features and edges first, lane areas last: the CUDA renderer reads
    # only the final road row to decide the ground is black (raster.cu:619), and
    # coplanar primitives tie-break on buffer order. prepare_runtime_roads
    # re-sorts into the same order, but keeping it here makes the raw array
    # renderable as-is.
    boundary_rows: list[list[float]] = []
    lane_rows: list[list[float]] = []
    seen: set[tuple] = set()

    for waypoints in lanes.values():
        waypoints.sort(key=lambda w: w.s)
        centre = np.asarray(
            [[w.transform.location.x, -w.transform.location.y] for w in waypoints], dtype=np.float64
        )
        lane_rows.extend(_chain(centre, ROAD_LANE, max_gap))

        # The lateral offset to each boundary. get_right_vector is in CARLA's
        # left-handed frame, so its y negates with everything else.
        right = np.asarray(
            [
                [w.transform.get_right_vector().x, -w.transform.get_right_vector().y]
                for w in waypoints
            ],
            dtype=np.float64,
        )
        half = np.asarray([[0.5 * w.lane_width] for w in waypoints], dtype=np.float64)
        sides = (
            (centre - right * half, [w.get_left_lane() for w in waypoints], "left_lane_marking"),
            (centre + right * half, [w.get_right_lane() for w in waypoints], "right_lane_marking"),
        )
        for points, neighbours, attribute in sides:
            # Classified per waypoint, not once per lane: a lane's neighbour and
            # its markings change along it, and a junction lane has neither.
            types = [
                None
                if waypoint.is_junction
                else _boundary_type(neighbour, getattr(waypoint, attribute))
                for waypoint, neighbour in zip(waypoints, neighbours)
            ]
            for index in range(len(points) - 1):
                type_id = types[index]
                # A boundary belongs to a segment only if both of its ends agree
                # on what it is; that also drops the junction entry and exit.
                if type_id is None or types[index + 1] != type_id:
                    continue
                for row in _chain(points[index : index + 2], type_id, max_gap):
                    # Adjacent driving lanes share a boundary and each emit it.
                    key = (
                        type_id,
                        *sorted(
                            (
                                (round(row[0] / DEDUP_QUANTUM), round(row[1] / DEDUP_QUANTUM)),
                                (round(row[2] / DEDUP_QUANTUM), round(row[3] / DEDUP_QUANTUM)),
                            )
                        ),
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    boundary_rows.append(row)

    for polygon in _crosswalk_polygons(carla_map):
        boundary_rows.extend(_chain(polygon, CROSSWALK, math.inf))

    rows = boundary_rows + lane_rows
    if not rows:
        return np.zeros((0, 6), dtype=np.float32)
    return np.ascontiguousarray(np.asarray(rows, dtype=np.float32))


def _crosswalk_polygons(carla_map) -> list[np.ndarray]:
    """Split ``Map.get_crosswalks()`` into closed polygons.

    CARLA returns one flat list in which each polygon repeats its first point to
    mark where it ends, so the repeat is the delimiter rather than a vertex to
    drop -- keeping it is what closes the ring.
    """
    locations = list(carla_map.get_crosswalks())
    polygons, current = [], []
    for location in locations:
        point = (location.x, -location.y)
        current.append(point)
        if len(current) > 2 and point == current[0]:
            polygons.append(np.asarray(current, dtype=np.float64))
            current = []
    if len(current) > 2:
        polygons.append(np.asarray(current, dtype=np.float64))
    return polygons


class RoadIndex:
    """A town's roads, croppable to the neighbourhood of one ego pose.

    Built once per town, then queried once per frame.  ``preprocess.py`` culls
    Waymo map features with a per-segment bounding-box test against the radius in
    the ego frame; this reproduces that exactly, so the two sources hand the
    rasterizer arrays of comparable size and extent.

    A town is only tens of thousands of segments, so transforming all of them and
    rejecting afterwards costs tens of microseconds -- cheaper than the spatial
    index it would take to avoid, and it keeps the cull semantics identical to
    the Waymo reader's rather than merely similar.
    """

    def __init__(self, roads: np.ndarray):
        roads = np.asarray(roads, dtype=np.float32)
        if roads.ndim != 2 or roads.shape[1:] != (6,):
            raise ValueError(f"roads must have shape [M,6], got {roads.shape}")
        self.roads = np.ascontiguousarray(roads)

    def crop(self, center: np.ndarray, yaw: float, radius: float) -> np.ndarray:
        """Return the roads within ``radius`` of ``center``, in the ego frame."""
        if not len(self.roads):
            return np.zeros((0, 6), dtype=np.float32)
        rows = self.roads.copy()
        rows[:, 0:2] = world_to_ego(rows[:, 0:2], center, yaw)
        rows[:, 2:4] = world_to_ego(rows[:, 2:4], center, yaw)
        low = np.minimum(rows[:, 0:2], rows[:, 2:4])
        high = np.maximum(rows[:, 0:2], rows[:, 2:4])
        keep = ~((high < -radius).any(axis=1) | (low > radius).any(axis=1))
        rows = rows[keep]
        # Lane areas must stay last: the CUDA renderer decides the ground is
        # black by reading only the final road row (raster.cu:619).
        lane = rows[:, 5] == ROAD_LANE
        return np.ascontiguousarray(np.concatenate([rows[~lane], rows[lane]]), dtype=np.float32)
