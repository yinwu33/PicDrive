"""Fit the agent state distribution used by the `giga` env from the WOMD map corpus.

Gigaflow samples vehicle size from independent uniforms -- length ~ U(0.8, 7),
width ~ U(0.8, 3) -- and has no notion of an agent *type* at all: a pedestrian is
just a small box. That is the wrong choice for this project, whose whole point is
that the rendered camera image should look like the Waymo sensor it will later be
aligned against. A vehicle 7 m long and 1.9 m wide renders as something that does
not exist on the road.

So instead we resample real (length, width, height) triples from the dataset. This
matters more than it might look: measured over 17,229 WOMD vehicles,
corr(length, width) = 0.822 and corr(length, height) = 0.811. Drawing the three
marginals independently -- the obvious implementation -- destroys that correlation
and produces bodies no real vehicle has. Resampling whole rows preserves the joint
distribution exactly, with no distributional assumption to get wrong.

Output format (little endian), consumed by pufferlib/giga/drive/agent_dist.h:

    magic       char[8]   "GIGADST1"
    num_types   int32
    per type:   int32 type_id, float32 probability, int32 num_rows
    per type:   float32 rows[num_rows][3]   # length, width, height

Usage:
    .venv/bin/python data_utils/womd/build_agent_dist.py \
        --map-dir resources/drive/binaries/training \
        --out resources/drive/agent_dist.bin
"""

import argparse
import glob
import struct
from collections import defaultdict

import numpy as np

VEHICLE, PEDESTRIAN, CYCLIST = 1, 2, 3
TYPE_NAMES = {VEHICLE: "VEHICLE", PEDESTRIAN: "PEDESTRIAN", CYCLIST: "CYCLIST"}
MAGIC = b"GIGADST1"


def read_objects(path):
    """Yield (type, length, width, height) for every object track in a .bin map."""
    with open(path, "rb") as f:
        b = f.read()
    o = 16  # scenario_id[16]
    (sdc,) = struct.unpack_from("i", b, o)
    o += 4
    (ntp,) = struct.unpack_from("i", b, o)
    o += 4 + 4 * ntp
    (nobj,) = struct.unpack_from("i", b, o)
    o += 4
    (nroad,) = struct.unpack_from("i", b, o)
    o += 4
    for i in range(nobj + nroad):
        _sc, typ, _id, size = struct.unpack_from("4i", b, o)
        o += 16
        o += 4 * size * 3  # x, y, z
        if typ in (VEHICLE, PEDESTRIAN, CYCLIST):
            o += 4 * size * 4  # vx, vy, vz, heading
            o += 4 * size  # valid (int32)
        w, l, h = struct.unpack_from("3f", b, o)
        o += 12
        o += 12 + 4  # goal xyz + mark_as_expert
        if typ in (VEHICLE, PEDESTRIAN, CYCLIST):
            yield typ, l, w, h


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--map-dir", default="resources/drive/binaries/training")
    ap.add_argument("--out", default="resources/drive/agent_dist.bin")
    ap.add_argument("--max-maps", type=int, default=2000, help="maps to scan; 2000 already gives >100k tracks")
    ap.add_argument("--rows-per-type", type=int, default=4096, help="triples kept per type")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    files = sorted(glob.glob(f"{args.map_dir}/map_*.bin"))[: args.max_maps]
    if not files:
        raise SystemExit(f"no maps found under {args.map_dir}")

    dims = defaultdict(list)
    for fp in files:
        for typ, l, w, h in read_objects(fp):
            # Guard against the handful of degenerate boxes in the corpus; a zero
            # extent would make the spawn clearance test meaningless.
            if l > 0.1 and w > 0.1 and h > 0.1:
                dims[typ].append((l, w, h))

    rng = np.random.default_rng(args.seed)
    total = sum(len(v) for v in dims.values())
    types = sorted(dims)

    print(f"scanned {len(files)} maps, {total} tracks")
    payload = []
    header = []
    for t in types:
        a = np.asarray(dims[t], dtype=np.float32)
        prob = len(a) / total
        if len(a) > args.rows_per_type:
            a = a[rng.choice(len(a), args.rows_per_type, replace=False)]
        header.append((t, prob, len(a)))
        payload.append(a)
        print(
            f"  {TYPE_NAMES.get(t, t):11s} p={prob:.4f}  rows={len(a):5d}  "
            f"len {a[:,0].mean():.3f}+-{a[:,0].std():.3f}  "
            f"wid {a[:,1].mean():.3f}+-{a[:,1].std():.3f}  "
            f"hgt {a[:,2].mean():.3f}+-{a[:,2].std():.3f}  "
            f"corr(l,w)={np.corrcoef(a[:,0],a[:,1])[0,1]:.3f} "
            f"corr(l,h)={np.corrcoef(a[:,0],a[:,2])[0,1]:.3f}"
        )

    with open(args.out, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<i", len(types)))
        for t, prob, n in header:
            f.write(struct.pack("<ifi", t, prob, n))
        for a in payload:
            f.write(np.ascontiguousarray(a, dtype="<f4").tobytes())
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
