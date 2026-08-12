import numpy as np

from data_utils.womd.preview_drivable_area import (
    BinaryMap,
    DRIVEWAY,
    ROAD_EDGE,
    ROAD_LANE,
    RasterTransform,
    RoadFeature,
    build_drivable_mask,
)


def _road(type_id, points):
    x, y = zip(*points)
    return RoadFeature(type_id, 0, np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32))


def test_mask_uses_lanes_and_driveways_but_not_road_edges():
    road_map = BinaryMap(
        "test",
        (
            _road(ROAD_LANE, [(2, 3), (8, 3)]),
            _road(ROAD_EDGE, [(2, 7), (8, 7)]),
            _road(DRIVEWAY, [(12, 2), (16, 2), (16, 5), (12, 5)]),
        ),
    )
    transform = RasterTransform(0, 10, 1, 20, 12)
    mask = np.asarray(
        build_drivable_mask(
            road_map,
            transform,
            lane_half_width_m=1,
            closing_radius_m=0,
            include_driveways=True,
        )
    )

    assert mask[7, 5] == 255  # lane at world y=3
    assert mask[3, 5] == 0  # road edge at world y=7 is audit-only
    assert mask[7, 14] == 255  # driveway polygon


def test_driveways_can_be_excluded():
    road_map = BinaryMap("test", (_road(DRIVEWAY, [(2, 2), (6, 2), (6, 5), (2, 5)]),))
    transform = RasterTransform(0, 10, 1, 10, 10)
    mask = build_drivable_mask(
        road_map,
        transform,
        lane_half_width_m=1,
        closing_radius_m=0,
        include_driveways=False,
    )
    assert np.count_nonzero(np.asarray(mask)) == 0
