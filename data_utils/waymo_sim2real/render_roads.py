"""Convert canonical Waymo map segments into the camera-policy RenderState.

Processed samples retain semantic map types, including ROAD_LANE centerlines.
The live ocean/teddy/giga environments perform a second, renderer-only pass in
``fill_render_roads`` before rasterization.  This module mirrors that pass for
offline paired samples so they reach the shared rasterizer with the same widths,
tags, ordering, and lane-strip overlap as live simulation.
"""

from __future__ import annotations

import numpy as np


ROAD_LANE = 4
ROAD_LINE = 5
ROAD_EDGE = 6
CROSSWALK = 8
SPEED_BUMP = 9

RENDER_LANE_AREA = 11
RENDER_YELLOW_ROAD_EDGE = 12

DEFAULT_LANE_WIDTH = 4.5
RENDER_WIDTHS = {
    ROAD_LINE: 0.15,
    ROAD_EDGE: 0.25,
    CROSSWALK: 0.50,
    SPEED_BUMP: 0.40,
}
DEFAULT_RENDER_TYPES = tuple(RENDER_WIDTHS)


def prepare_runtime_roads(roads: np.ndarray, lane_width: float = DEFAULT_LANE_WIDTH) -> np.ndarray:
    """Return the exact road primitive semantics consumed by camera policies.

    Input rows are canonical ``[x0,y0,x1,y1,width,type]`` map segments. Painted
    features are emitted first. ROAD_EDGE becomes the shared yellow-edge tag;
    ROAD_LANE is emitted last as an overlapping lane-area strip.
    """
    roads = np.asarray(roads, dtype=np.float32)
    if roads.ndim != 2 or roads.shape[1:] != (6,):
        raise ValueError(f"roads must have shape [N,6], got {roads.shape}")
    if lane_width <= 0:
        raise ValueError(f"lane_width must be positive, got {lane_width}")
    if not len(roads):
        return np.zeros((0, 6), dtype=np.float32)
    if np.isin(roads[:, 5], (RENDER_LANE_AREA, RENDER_YELLOW_ROAD_EDGE)).any():
        raise ValueError("roads already contain renderer-only type 11/12 tags")

    painted_mask = np.isin(roads[:, 5], DEFAULT_RENDER_TYPES)
    painted = roads[painted_mask].copy()
    for type_id, width in RENDER_WIDTHS.items():
        painted[painted[:, 5] == type_id, 4] = width
    painted[painted[:, 5] == ROAD_EDGE, 5] = RENDER_YELLOW_ROAD_EDGE

    lane_area = roads[roads[:, 5] == ROAD_LANE].copy()
    if len(lane_area):
        lane_area[:, 4] = lane_width
        lane_area[:, 5] = RENDER_LANE_AREA
        delta = lane_area[:, 2:4] - lane_area[:, 0:2]
        length = np.linalg.norm(delta, axis=1)
        valid = length > 1e-6
        extension = np.zeros_like(delta)
        extension[valid] = delta[valid] / length[valid, None] * (lane_width * 0.5)
        lane_area[:, 0:2] -= extension
        lane_area[:, 2:4] += extension

    return np.ascontiguousarray(np.concatenate((painted, lane_area), axis=0), dtype=np.float32)
