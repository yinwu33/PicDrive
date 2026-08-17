"""Dependency-light reader for the legacy Waymo Perception TFRecord format.

PufferDrive's Python 3.10 environment is newer than the official legacy Waymo
wheel.  Pulling TensorFlow into the training environment just to decode proto
messages would also make the offline preprocessing unnecessarily fragile.  This
module therefore implements the small protobuf/TFRecord subset needed by the
sim-to-real preprocessor.  It is intentionally private to the raw-data stage.

The field numbers mirror Waymo Open Dataset v1.4.x ``dataset.proto``,
``label.proto`` and ``protos/map.proto``.  Unknown fields are skipped, so the
reader remains tolerant of additive schema changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct
from typing import BinaryIO, Iterator

import numpy as np


class ProtoDecodeError(ValueError):
    """Raised when a protobuf or TFRecord payload is structurally invalid."""


@dataclass(frozen=True)
class Field:
    number: int
    wire_type: int
    value: int | bytes


def _read_varint(data: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while offset < len(data) and shift < 70:
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, offset
        shift += 7
    raise ProtoDecodeError("unterminated protobuf varint")


def iter_fields(data: bytes) -> Iterator[Field]:
    """Iterate top-level protobuf fields without materializing unknown data."""
    offset = 0
    while offset < len(data):
        key, offset = _read_varint(data, offset)
        number, wire_type = key >> 3, key & 7
        if number == 0:
            raise ProtoDecodeError("protobuf field number 0 is invalid")
        if wire_type == 0:
            value, offset = _read_varint(data, offset)
        elif wire_type == 1:
            end = offset + 8
            if end > len(data):
                raise ProtoDecodeError("truncated fixed64 field")
            value, offset = data[offset:end], end
        elif wire_type == 2:
            length, offset = _read_varint(data, offset)
            end = offset + length
            if end > len(data):
                raise ProtoDecodeError("truncated length-delimited field")
            value, offset = data[offset:end], end
        elif wire_type == 5:
            end = offset + 4
            if end > len(data):
                raise ProtoDecodeError("truncated fixed32 field")
            value, offset = data[offset:end], end
        else:
            raise ProtoDecodeError(f"unsupported protobuf wire type {wire_type}")
        yield Field(number, wire_type, value)


def first_bytes(data: bytes, number: int, default: bytes | None = None) -> bytes | None:
    for field in iter_fields(data):
        if field.number == number and field.wire_type == 2:
            return field.value  # type: ignore[return-value]
    return default


def first_int(data: bytes, number: int, default: int | None = None) -> int | None:
    for field in iter_fields(data):
        if field.number == number and field.wire_type == 0:
            return int(field.value)
    return default


def repeated_bytes(data: bytes, number: int) -> list[bytes]:
    return [
        field.value  # type: ignore[misc]
        for field in iter_fields(data)
        if field.number == number and field.wire_type == 2
    ]


def repeated_doubles(data: bytes, number: int) -> list[float]:
    """Decode repeated doubles in packed or unpacked proto2 representation."""
    values: list[float] = []
    for field in iter_fields(data):
        if field.number != number:
            continue
        if field.wire_type == 1:
            values.append(struct.unpack("<d", field.value)[0])  # type: ignore[arg-type]
        elif field.wire_type == 2:
            raw = field.value  # type: ignore[assignment]
            if len(raw) % 8:
                raise ProtoDecodeError(f"packed double field {number} has invalid length")
            values.extend(struct.unpack(f"<{len(raw) // 8}d", raw))
    return values


def parse_transform(data: bytes | None) -> np.ndarray:
    if data is None:
        raise ProtoDecodeError("missing Transform message")
    values = repeated_doubles(data, 1)
    if len(values) != 16:
        raise ProtoDecodeError(f"Transform contains {len(values)} values, expected 16")
    return np.asarray(values, dtype=np.float64).reshape(4, 4)


def parse_vector3d(data: bytes | None) -> np.ndarray:
    if data is None:
        return np.zeros(3, dtype=np.float64)
    xyz = [repeated_doubles(data, field) for field in (1, 2, 3)]
    if any(len(v) > 1 for v in xyz):
        raise ProtoDecodeError("Vector3d field is repeated unexpectedly")
    return np.asarray([v[0] if v else 0.0 for v in xyz], dtype=np.float64)


def iter_tfrecord(path: str | Path) -> Iterator[bytes]:
    """Yield record payloads from an uncompressed TFRecord file.

    Length and payload boundaries are checked. CRC verification is omitted to
    avoid introducing TensorFlow or a CRC32C dependency; malformed/truncated
    records still fail deterministically at their first invalid boundary.
    """
    with Path(path).open("rb") as handle:
        yield from _iter_tfrecord_handle(handle)


def _iter_tfrecord_handle(handle: BinaryIO) -> Iterator[bytes]:
    record_index = 0
    while True:
        header = handle.read(12)
        if not header:
            return
        if len(header) != 12:
            raise ProtoDecodeError(f"truncated TFRecord header at record {record_index}")
        length = struct.unpack("<Q", header[:8])[0]
        payload = handle.read(length)
        crc = handle.read(4)
        if len(payload) != length or len(crc) != 4:
            raise ProtoDecodeError(f"truncated TFRecord payload at record {record_index}")
        record_index += 1
        yield payload
