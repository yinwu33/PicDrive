import struct

import numpy as np

from data_utils.sim2real.waymo.preprocess import _parse_map_features
from data_utils.sim2real.waymo.render_roads import RENDER_LANE_AREA, RENDER_YELLOW_ROAD_EDGE, prepare_runtime_roads


def _varint(value):
    out = bytearray()
    while value >= 0x80:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)
    return bytes(out)


def _message_field(number, payload):
    return _varint((number << 3) | 2) + _varint(len(payload)) + payload


def _point(x, y, z=0.0):
    return b"".join(_varint((number << 3) | 1) + struct.pack("<d", value) for number, value in ((1, x), (2, y), (3, z)))


def test_map_parser_retains_lane_centerlines():
    lane = _message_field(8, _point(1.0, 2.0)) + _message_field(8, _point(4.0, 6.0))
    road_line = _message_field(2, _point(2.0, 3.0)) + _message_field(2, _point(5.0, 7.0))
    frame = _message_field(10, _message_field(3, lane)) + _message_field(10, _message_field(4, road_line))

    features = _parse_map_features(frame)

    assert [type_id for type_id, _ in features] == [4, 5]
    np.testing.assert_allclose(features[0][1], [[1, 2, 0], [4, 6, 0]])
    np.testing.assert_allclose(features[1][1], [[2, 3, 0], [5, 7, 0]])


def test_runtime_roads_match_live_lane_area_semantics():
    canonical = np.asarray(
        [
            [0, 0, 10, 0, 4.5, 4],
            [0, 2, 10, 2, 0.25, 6],
            [0, 1, 10, 1, 0.15, 5],
            [0, -2, 10, -2, 0.5, 8],
            [0, 3, 10, 3, 1.0, 10],
        ],
        dtype=np.float32,
    )

    rendered = prepare_runtime_roads(canonical)

    np.testing.assert_array_equal(rendered[:, 5], [RENDER_YELLOW_ROAD_EDGE, 5, 8, RENDER_LANE_AREA])
    np.testing.assert_allclose(rendered[:, 4], [0.25, 0.15, 0.5, 4.5])
    np.testing.assert_allclose(rendered[-1, :4], [-2.25, 0, 12.25, 0])
