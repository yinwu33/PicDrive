"""Strip logged agent tracks from PufferDrive binary maps.

The teddy/giga environments use the WOMD road entities but replace every logged
agent with a synthetic one.  The legacy binary files nevertheless contain all 91
steps of every logged track, so loading them wastes disk space and I/O.

This script writes files in the same format expected by ``load_map_binary``.  It
keeps the scenario id and road entity records byte-for-byte, while resetting all
metadata that indexes logged objects::

    sdc_track_index = -1
    tracks_to_predict = []
    num_objects = 0

The output can therefore be selected by changing only ``map_dir`` in a teddy or
giga config.

Example::

    .venv/bin/python data_utils/womd/build_map_only.py \
        --input resources/drive/binaries/training \
        --output resources/drive/binaries/training_map_only
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import os
from pathlib import Path
import struct


HEADER = struct.Struct("<16siiii")
ENTITY_HEADER = struct.Struct("<iiii")
INT32 = struct.Struct("<i")
SCALARS_SIZE = 6 * 4 + 4  # width/length/height, goal xyz, mark_as_expert
AGENT_TYPES = {1, 2, 3}


@dataclass(frozen=True)
class MapStats:
    source_bytes: int
    output_bytes: int
    objects_removed: int
    roads_kept: int


def _read_i32(data: bytes, offset: int, label: str) -> tuple[int, int]:
    if offset + INT32.size > len(data):
        raise ValueError(f"truncated while reading {label} at byte {offset}")
    return INT32.unpack_from(data, offset)[0], offset + INT32.size


def _entity_end(data: bytes, offset: int, *, expect_agent: bool) -> int:
    if offset + ENTITY_HEADER.size > len(data):
        raise ValueError(f"truncated entity header at byte {offset}")
    _, type_id, _, array_size = ENTITY_HEADER.unpack_from(data, offset)
    if array_size < 0:
        raise ValueError(f"negative entity array_size {array_size} at byte {offset}")
    is_agent = type_id in AGENT_TYPES
    if is_agent != expect_agent:
        expected = "agent" if expect_agent else "road"
        raise ValueError(f"expected {expected} entity at byte {offset}, found type {type_id}")
    arrays = 8 if is_agent else 3
    end = offset + ENTITY_HEADER.size + arrays * array_size * 4 + SCALARS_SIZE
    if end > len(data):
        raise ValueError(f"truncated type {type_id} entity at byte {offset}")
    return end


def strip_map_bytes(data: bytes) -> tuple[bytes, int, int]:
    """Return a loader-compatible map containing roads and no logged objects."""
    if len(data) < 24:
        raise ValueError("file is shorter than the binary map header")

    scenario_id = data[:16]
    offset = 16
    _, offset = _read_i32(data, offset, "sdc_track_index")
    num_tracks, offset = _read_i32(data, offset, "num_tracks_to_predict")
    if num_tracks < 0:
        raise ValueError(f"negative tracks_to_predict count {num_tracks}")
    offset += num_tracks * INT32.size
    num_objects, offset = _read_i32(data, offset, "num_objects")
    num_roads, offset = _read_i32(data, offset, "num_roads")
    if num_objects < 0 or num_roads < 0:
        raise ValueError(f"negative entity counts objects={num_objects}, roads={num_roads}")

    for _ in range(num_objects):
        offset = _entity_end(data, offset, expect_agent=True)
    roads_offset = offset
    for _ in range(num_roads):
        offset = _entity_end(data, offset, expect_agent=False)
    if offset != len(data):
        raise ValueError(f"{len(data) - offset} trailing bytes after the final road entity")

    header = HEADER.pack(scenario_id, -1, 0, 0, num_roads)
    return header + data[roads_offset:], num_objects, num_roads


def convert_map(source: Path, destination: Path, overwrite: bool) -> MapStats:
    if destination.exists() and not overwrite:
        # Resume only after reproducing the expected bytes from the source. This
        # catches a valid-looking output copied from the wrong map just as surely
        # as it catches truncation.
        source_data = source.read_bytes()
        expected, objects, roads = strip_map_bytes(source_data)
        actual = destination.read_bytes()
        if actual != expected:
            raise ValueError(f"existing output does not match its source: {destination}")
        return MapStats(len(source_data), len(actual), objects, roads)

    source_data = source.read_bytes()
    output_data, objects, roads = strip_map_bytes(source_data)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as file:
            file.write(output_data)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return MapStats(len(source_data), len(output_data), objects, roads)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-maps", type=int)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.input.resolve() == args.output.resolve():
        parser.error("--input and --output must be different directories")

    sources = sorted(args.input.glob("map_*.bin"))
    if args.max_maps is not None:
        sources = sources[: args.max_maps]
    if not sources:
        parser.error(f"no map_*.bin files found under {args.input}")

    args.output.mkdir(parents=True, exist_ok=True)
    # A killed process can leave only its hidden, PID-stamped temporary files;
    # completed destinations are installed atomically and never match this glob.
    for temporary in args.output.glob(".map_*.bin.tmp-*"):
        temporary.unlink()

    def job(source: Path) -> MapStats:
        return convert_map(source, args.output / source.name, args.overwrite)

    source_bytes = output_bytes = objects = roads = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for index, stats in enumerate(executor.map(job, sources), 1):
            source_bytes += stats.source_bytes
            output_bytes += stats.output_bytes
            objects += stats.objects_removed
            roads += stats.roads_kept
            if index % 1000 == 0 or index == len(sources):
                print(f"[{index}/{len(sources)}] maps", flush=True)

    ratio = output_bytes / source_bytes if source_bytes else 0.0
    print(f"removed {objects} logged object tracks; kept {roads} road entities")
    print(
        f"size {source_bytes / 2**30:.3f} GiB -> {output_bytes / 2**30:.3f} GiB "
        f"({ratio:.1%}, saved {(source_bytes - output_bytes) / 2**30:.3f} GiB)"
    )


if __name__ == "__main__":
    main()
