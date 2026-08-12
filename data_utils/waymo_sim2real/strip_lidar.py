"""Remove LiDAR sensor payloads from Waymo Perception TFRecords in place.

Only the top-level repeated ``Frame.lasers`` field (protobuf field 5) is
removed. Camera images, camera and laser calibrations, poses, maps, and
``laser_labels`` (field 6, the 3D boxes used by our renderer) remain byte-for-
byte identical.

"In place" means one input file at a time is streamed into a temporary file in
the same directory, validated, and atomically replaced with :func:`os.replace`.
It is not possible to shrink a TFRecord by overwriting bytes in the existing
inode: every record stores its payload length and CRC32C checksum. At most one
temporary TFRecord is present, so peak scratch space is bounded by one segment.

Install the accelerated CRC32C implementation before running this over the
full dataset::

    uv pip install --python .venv/bin/python google-crc32c

Dry-run one file, then strip the directory::

    python -m data_utils.waymo_sim2real.strip_lidar \
        --input /path/to/training --dry-run --max-files 1
    python -m data_utils.waymo_sim2real.strip_lidar \
        --input /path/to/training --in-place

Both the overall file bar and the per-file rewrite/verification bars show an
ETA. Re-running after an interruption is safe: a source stays untouched until
its replacement validates, and files already stripped are reported unchanged.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import struct
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO

from tqdm.auto import tqdm

from .proto import ProtoDecodeError


LIDAR_FIELD_NUMBER = 5
TFRECORD_OVERHEAD = 16
_MASK_DELTA = 0xA282EAD8
_Progress = Callable[[int], None]

try:
    import google_crc32c
except ImportError:  # Covered by the CLI prerequisite check.
    google_crc32c = None


@dataclass(frozen=True)
class FileResult:
    path: str
    changed: bool
    records: int
    lidar_fields: int
    before_bytes: int
    after_bytes: int
    removed_bytes: int


@dataclass
class Summary:
    files: int = 0
    changed: int = 0
    already_stripped: int = 0
    records: int = 0
    lidar_fields: int = 0
    before_bytes: int = 0
    after_bytes: int = 0
    removed_bytes: int = 0
    elapsed_seconds: float = 0.0

    def add(self, result: FileResult) -> None:
        self.files += 1
        self.changed += int(result.changed)
        self.already_stripped += int(not result.changed)
        self.records += result.records
        self.lidar_fields += result.lidar_fields
        self.before_bytes += result.before_bytes
        self.after_bytes += result.after_bytes
        self.removed_bytes += result.removed_bytes


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


def _field_spans(data: bytes) -> Iterator[tuple[int, int, int]]:
    """Yield ``(field_number, start, end)`` without re-encoding protobuf data."""
    offset = 0
    while offset < len(data):
        start = offset
        key, offset = _read_varint(data, offset)
        number, wire_type = key >> 3, key & 7
        if number == 0:
            raise ProtoDecodeError("protobuf field number 0 is invalid")
        if wire_type == 0:
            _, offset = _read_varint(data, offset)
        elif wire_type == 1:
            offset += 8
        elif wire_type == 2:
            length, offset = _read_varint(data, offset)
            offset += length
        elif wire_type == 5:
            offset += 4
        else:
            raise ProtoDecodeError(f"unsupported protobuf wire type {wire_type}")
        if offset > len(data):
            raise ProtoDecodeError(f"truncated protobuf field {number}")
        yield number, start, offset


def _strip_top_level_lidar(payload: bytes) -> tuple[bytes, int, int]:
    """Copy every raw field span except top-level Frame field 5."""
    output = bytearray()
    removed_fields = 0
    removed_bytes = 0
    for number, start, end in _field_spans(payload):
        if number == LIDAR_FIELD_NUMBER:
            removed_fields += 1
            removed_bytes += end - start
        else:
            output.extend(payload[start:end])
    return bytes(output), removed_fields, removed_bytes


def _crc32c(data: bytes) -> int:
    if google_crc32c is None:
        raise RuntimeError(
            "google-crc32c is required; install it with: uv pip install --python .venv/bin/python google-crc32c"
        )
    return int(google_crc32c.value(data))


def _masked_crc32c(data: bytes) -> int:
    checksum = _crc32c(data)
    return (((checksum >> 15) | (checksum << 17)) + _MASK_DELTA) & 0xFFFFFFFF


def _read_record(
    handle: BinaryIO,
    path: Path,
    record_index: int,
    progress: _Progress | None,
) -> bytes | None:
    header = handle.read(12)
    if not header:
        return None
    if len(header) != 12:
        raise ProtoDecodeError(f"{path}: truncated TFRecord header at record {record_index}")
    length_bytes = header[:8]
    stored_length_crc = struct.unpack("<I", header[8:])[0]
    expected_length_crc = _masked_crc32c(length_bytes)
    if stored_length_crc != expected_length_crc:
        raise ProtoDecodeError(
            f"{path}: invalid length CRC at record {record_index}: "
            f"stored {stored_length_crc:#010x}, expected {expected_length_crc:#010x}"
        )
    length = struct.unpack("<Q", length_bytes)[0]
    payload = handle.read(length)
    payload_crc = handle.read(4)
    if len(payload) != length or len(payload_crc) != 4:
        raise ProtoDecodeError(f"{path}: truncated TFRecord payload at record {record_index}")
    stored_payload_crc = struct.unpack("<I", payload_crc)[0]
    expected_payload_crc = _masked_crc32c(payload)
    if stored_payload_crc != expected_payload_crc:
        raise ProtoDecodeError(
            f"{path}: invalid payload CRC at record {record_index}: "
            f"stored {stored_payload_crc:#010x}, expected {expected_payload_crc:#010x}"
        )
    if progress is not None:
        progress(length + TFRECORD_OVERHEAD)
    return payload


def _write_record(handle: BinaryIO, payload: bytes) -> None:
    length_bytes = struct.pack("<Q", len(payload))
    handle.write(length_bytes)
    handle.write(struct.pack("<I", _masked_crc32c(length_bytes)))
    handle.write(payload)
    handle.write(struct.pack("<I", _masked_crc32c(payload)))


def _scan_file(path: Path, progress: _Progress | None = None) -> FileResult:
    records = lidar_fields = removed_bytes = 0
    with path.open("rb") as handle:
        while (payload := _read_record(handle, path, records, progress)) is not None:
            _, count, removed = _strip_top_level_lidar(payload)
            lidar_fields += count
            removed_bytes += removed
            records += 1
    before = path.stat().st_size
    after = before - removed_bytes
    return FileResult(
        path=str(path),
        changed=lidar_fields > 0,
        records=records,
        lidar_fields=lidar_fields,
        before_bytes=before,
        after_bytes=after,
        removed_bytes=removed_bytes,
    )


def _verify_stripped_file(
    path: Path,
    expected_records: int,
    progress: _Progress | None = None,
) -> None:
    records = 0
    with path.open("rb") as handle:
        while (payload := _read_record(handle, path, records, progress)) is not None:
            for number, _, _ in _field_spans(payload):
                if number == LIDAR_FIELD_NUMBER:
                    raise ProtoDecodeError(f"{path}: LiDAR field remains in output record {records}")
            records += 1
    if records != expected_records:
        raise ProtoDecodeError(f"{path}: output contains {records} records, expected {expected_records}")


def _temporary_path(source: Path) -> Path:
    return source.with_name(f".{source.name}.strip-lidar.tmp")


def _fsync_directory(directory: Path) -> None:
    descriptor = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def strip_file_in_place(
    source: Path,
    rewrite_progress: _Progress | None = None,
    verify_progress_factory: Callable[[int], _Progress | None] | None = None,
) -> FileResult:
    """Rewrite, verify, and atomically replace one TFRecord."""
    source = source.absolute()
    if source.is_symlink():
        raise ValueError(f"refusing to replace symlink: {source}")
    before_stat = source.stat()
    if not stat.S_ISREG(before_stat.st_mode):
        raise ValueError(f"not a regular file: {source}")
    if before_stat.st_nlink != 1:
        raise ValueError(
            f"refusing {source}: it has {before_stat.st_nlink} hard links; replacing only one "
            "name would not reclaim the original blocks"
        )
    temporary = _temporary_path(source)
    if temporary.exists():
        raise FileExistsError(f"stale temporary file exists: {temporary}; inspect and remove it before retrying")
    free = shutil.disk_usage(source.parent).free
    if free < before_stat.st_size:
        raise OSError(
            f"{source.parent} has only {free / 2**30:.2f} GiB free; safely rewriting "
            f"{source.name} requires at least {before_stat.st_size / 2**30:.2f} GiB"
        )

    records = lidar_fields = removed_bytes = 0
    replaced = False
    try:
        with source.open("rb") as input_handle, temporary.open("xb") as output_handle:
            while (payload := _read_record(input_handle, source, records, rewrite_progress)) is not None:
                stripped, count, removed = _strip_top_level_lidar(payload)
                _write_record(output_handle, stripped)
                lidar_fields += count
                removed_bytes += removed
                records += 1
            output_handle.flush()
            os.fsync(output_handle.fileno())

        # Preserve mode, timestamps, flags and (where supported) xattrs. The
        # local Waymo mirror is owned by the executing user, so its original
        # group can also be retained rather than inheriting the directory's.
        shutil.copystat(source, temporary, follow_symlinks=False)
        os.chown(temporary, before_stat.st_uid, before_stat.st_gid)
        temporary_stat = temporary.stat()
        verify_tracker = (
            verify_progress_factory(temporary_stat.st_size) if verify_progress_factory is not None else None
        )
        if hasattr(verify_tracker, "update"):
            verify_progress = verify_tracker.update
        else:
            verify_progress = verify_tracker
        try:
            _verify_stripped_file(temporary, records, verify_progress)
        finally:
            close = getattr(verify_tracker, "close", None)
            if close is not None:
                close()

        current_stat = source.stat()
        identity_before = (
            before_stat.st_dev,
            before_stat.st_ino,
            before_stat.st_size,
            before_stat.st_mtime_ns,
        )
        identity_now = (
            current_stat.st_dev,
            current_stat.st_ino,
            current_stat.st_size,
            current_stat.st_mtime_ns,
        )
        if identity_now != identity_before:
            raise RuntimeError(f"{source} changed while it was being rewritten; source left intact")

        if lidar_fields == 0:
            temporary.unlink()
            return FileResult(
                path=str(source),
                changed=False,
                records=records,
                lidar_fields=0,
                before_bytes=before_stat.st_size,
                after_bytes=before_stat.st_size,
                removed_bytes=0,
            )

        after = temporary_stat.st_size
        if before_stat.st_size - after != removed_bytes:
            raise RuntimeError(
                f"{temporary}: size changed by {before_stat.st_size - after}, "
                f"but removed protobuf spans total {removed_bytes}"
            )

        os.replace(temporary, source)
        replaced = True
        _fsync_directory(source.parent)
        return FileResult(
            path=str(source),
            changed=True,
            records=records,
            lidar_fields=lidar_fields,
            before_bytes=before_stat.st_size,
            after_bytes=after,
            removed_bytes=removed_bytes,
        )
    except BaseException:
        # Before os.replace, any failure leaves the source untouched. A rare
        # directory-fsync failure happens after a fully validated replacement;
        # do not claim that valid new source was rolled back.
        if not replaced:
            temporary.unlink(missing_ok=True)
        raise


def _input_files(path: Path, max_files: int | None) -> list[Path]:
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = sorted(item for item in path.glob("*.tfrecord") if item.is_file())
    else:
        raise FileNotFoundError(path)
    if max_files is not None:
        files = files[:max_files]
    if not files:
        raise FileNotFoundError(f"no .tfrecord files found at {path}")
    return files


def _format_gib(value: int) -> str:
    return f"{value / 2**30:.2f} GiB"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="TFRecord file or directory")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="scan and verify CRCs without writing or replacing files",
    )
    mode.add_argument(
        "--in-place",
        action="store_true",
        help="destructively replace each TFRecord after its stripped copy verifies",
    )
    parser.add_argument("--max-files", type=int, help="limit sorted inputs, useful for a trial")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.max_files is not None and args.max_files < 1:
        parser.error("--max-files must be positive")
    if google_crc32c is None:
        parser.error(
            "google-crc32c is required for fast, standards-compliant TFRecord checksums. Run:\n"
            "  uv pip install --python .venv/bin/python google-crc32c"
        )

    files = _input_files(args.input, args.max_files)
    summary = Summary()
    started = time.monotonic()
    action = "Checking" if args.dry_run else "Stripping"
    with tqdm(
        total=len(files),
        desc=f"{action} TFRecords",
        unit="file",
        dynamic_ncols=True,
    ) as overall:
        for source in files:
            overall.set_postfix_str(f"current={source.name[:48]}")
            source_size = source.stat().st_size
            stage = "scan" if args.dry_run else "rewrite"
            with tqdm(
                total=source_size,
                desc=f"  {stage}",
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                leave=False,
                dynamic_ncols=True,
            ) as current:
                if args.dry_run:
                    result = _scan_file(source, current.update)
                else:

                    def verify_bar(size: int):
                        return tqdm(
                            total=size,
                            desc="  verify",
                            unit="B",
                            unit_scale=True,
                            unit_divisor=1024,
                            leave=False,
                            dynamic_ncols=True,
                        )

                    result = strip_file_in_place(
                        source,
                        rewrite_progress=current.update,
                        verify_progress_factory=verify_bar,
                    )
            summary.add(result)
            overall.set_postfix(
                saved=_format_gib(summary.removed_bytes),
                unchanged=summary.already_stripped,
            )
            overall.update()

    summary.elapsed_seconds = time.monotonic() - started
    payload = asdict(summary)
    payload["mode"] = "dry-run" if args.dry_run else "in-place"
    payload["before_gib"] = summary.before_bytes / 2**30
    payload["after_gib"] = summary.after_bytes / 2**30
    payload["removed_gib"] = summary.removed_bytes / 2**30
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.dry_run:
        print("Dry run only: no files were changed.", file=sys.stderr)


if __name__ == "__main__":
    main()
