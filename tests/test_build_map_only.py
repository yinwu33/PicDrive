import struct

import pytest

from data_utils.womd.build_map_only import HEADER, strip_map_bytes


def _entity(type_id: int, size: int, marker: float) -> bytes:
    arrays = 8 if type_id in (1, 2, 3) else 3
    return b"".join(
        [
            struct.pack("<iiii", 7, type_id, 123, size),
            struct.pack(f"<{arrays * size}f", *([marker] * arrays * size)),
            struct.pack("<6fi", *([marker] * 6), 1),
        ]
    )


def test_strip_map_bytes_removes_agents_and_keeps_roads_exactly():
    scenario = b"scenario-123".ljust(16, b"\0")
    agents = _entity(1, 3, 1.25) + _entity(2, 2, 2.25)
    roads = _entity(4, 4, 4.25) + _entity(6, 3, 6.25)
    original = b"".join(
        [scenario, struct.pack("<ii2iii", 1, 2, 0, 1, 2, 2), agents, roads]
    )

    stripped, objects, road_count = strip_map_bytes(original)

    assert (objects, road_count) == (2, 2)
    assert stripped[: HEADER.size] == HEADER.pack(scenario, -1, 0, 0, 2)
    assert stripped[HEADER.size :] == roads


def test_strip_map_bytes_rejects_truncation():
    scenario = b"scenario-123".ljust(16, b"\0")
    original = HEADER.pack(scenario, -1, 0, 0, 1) + _entity(4, 4, 1.0)
    with pytest.raises(ValueError, match="truncated"):
        strip_map_bytes(original[:-1])
