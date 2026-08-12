from pathlib import Path

from data_utils.waymo_sim2real.strip_lidar import (
    _field_spans,
    _read_record,
    _scan_file,
    _strip_top_level_lidar,
    _write_record,
    strip_file_in_place,
)


def _varint(value: int) -> bytes:
    output = bytearray()
    while value >= 0x80:
        output.append((value & 0x7F) | 0x80)
        value >>= 7
    output.append(value)
    return bytes(output)


def _bytes_field(number: int, payload: bytes) -> bytes:
    return _varint((number << 3) | 2) + _varint(len(payload)) + payload


def _frame(with_lidar: bool = True) -> tuple[bytes, bytes]:
    context = _bytes_field(1, _bytes_field(5, b"nested calibration stays"))
    image = _bytes_field(4, b"jpeg bytes stay exactly")
    labels = _bytes_field(6, b"3D laser labels stay")
    lidar = _bytes_field(5, b"range image A") + _bytes_field(5, b"range image B")
    kept = context + image + labels
    return kept + lidar if with_lidar else kept, kept


def test_lidar_stripper_removes_only_raw_top_level_field_five():
    frame, kept = _frame()
    stripped, count, removed = _strip_top_level_lidar(frame)

    assert stripped == kept
    assert count == 2
    assert removed == len(frame) - len(kept)
    assert [number for number, _, _ in _field_spans(stripped)] == [1, 4, 6]


def test_tfrecord_is_atomically_stripped_and_rerun_is_idempotent(tmp_path: Path):
    source = tmp_path / "segment.tfrecord"
    first, first_kept = _frame()
    second, second_kept = _frame(with_lidar=False)
    with source.open("wb") as handle:
        _write_record(handle, first)
        _write_record(handle, second)
    source.chmod(0o640)
    before = source.stat().st_size

    projection = _scan_file(source)
    result = strip_file_in_place(source)

    assert projection.removed_bytes == result.removed_bytes
    assert result.changed
    assert result.records == 2
    assert result.lidar_fields == 2
    assert result.before_bytes == before
    assert result.after_bytes == before - result.removed_bytes
    assert source.stat().st_mode & 0o777 == 0o640
    with source.open("rb") as handle:
        assert _read_record(handle, source, 0, None) == first_kept
        assert _read_record(handle, source, 1, None) == second_kept
        assert _read_record(handle, source, 2, None) is None

    stripped_bytes = source.read_bytes()
    repeated = strip_file_in_place(source)
    assert not repeated.changed
    assert repeated.removed_bytes == 0
    assert source.read_bytes() == stripped_bytes
