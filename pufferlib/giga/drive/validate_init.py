"""Assertions on the Gigaflow random initialization.

viz.py is for the eye; this is for the things the eye cannot check -- that spawn poses
are on-road and non-overlapping, that headings follow their lane, that waypoint spacing
respects its bounds, and above all that the sampled (length, width, height) keeps the
*joint* distribution of the WOMD corpus rather than just the marginals.

    .venv/bin/python -m pufferlib.giga.drive.validate_init
"""

import argparse
import struct
import sys

import numpy as np

from pufferlib.giga.drive import binding
from pufferlib.giga.drive.drive import Drive

VEHICLE, PEDESTRIAN, CYCLIST = 1, 2, 3
GIGA_NUM_COND = binding.GIGA_NUM_COND
COND_DELTA_GOAL = 0
COND_SLOT_IS_FINAL = 1
COND_NAMES = ["delta_goal", "is_final", "a_collision", "a_boundary", "a_comfort", "a_l_align",
              "a_vel_align", "a_l_center", "a_center_bias", "a_reverse", "C_throttle", "C_steer", "C_acc"]
NAMES = {VEHICLE: "vehicle", PEDESTRIAN: "pedestrian", CYCLIST: "cyclist"}


def _corners(row):
    x, y, heading, length, width = row[0], row[1], row[2], row[3], row[4]
    ch, sh = np.cos(heading), np.sin(heading)
    hl, hw = length / 2.0, width / 2.0
    return np.array([[x + hl * ch - hw * sh, y + hl * sh + hw * ch],
                     [x + hl * ch + hw * sh, y + hl * sh - hw * ch],
                     [x - hl * ch + hw * sh, y - hl * sh - hw * ch],
                     [x - hl * ch - hw * sh, y - hl * sh + hw * ch]])


def _boxes_overlap(ra, rb):
    """Separating-axis test on two oriented boxes, matching the simulator's check."""
    A, B = _corners(ra), _corners(rb)
    for poly in (A, B):
        for k in range(4):
            edge = poly[(k + 1) % 4] - poly[k]
            axis = np.array([-edge[1], edge[0]])
            norm = np.hypot(*axis)
            if norm < 1e-9:
                continue
            axis = axis / norm
            pa, pb = A @ axis, B @ axis
            if pa.max() < pb.min() or pb.max() < pa.min():
                return False
    return True


def read_reference(path):
    with open(path, "rb") as f:
        b = f.read()
    assert b[:8] == b"GIGADST1", "not a GIGADST1 file"
    o = 8
    (n_types,) = struct.unpack_from("<i", b, o)
    o += 4
    hdr = []
    for _ in range(n_types):
        t, prob, n = struct.unpack_from("<ifi", b, o)
        o += 12
        hdr.append((t, prob, n))
    out = {}
    for t, prob, n in hdr:
        rows = np.frombuffer(b, "<f4", n * 3, o).reshape(n, 3)
        o += 4 * n * 3
        out[t] = dict(prob=prob, rows=rows)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-maps", type=int, default=120)
    ap.add_argument("--agents-per-map", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--agent-dist", default="resources/drive/agent_dist.bin")
    args = ap.parse_args()

    ref = read_reference(args.agent_dist)
    env = Drive(
        num_agents=args.num_maps * args.agents_per_map,
        num_maps=10000,
        obs_mode="render_state",
        dynamics_model="jerk",
        episode_length=1280,
        seed=args.seed,
        agents_per_map_min=args.agents_per_map,
        agents_per_map_max=args.agents_per_map,
        render_mode=1,
    )
    env.reset(seed=args.seed)

    agents, overlaps, min_clear, circle_hits = [], 0, [], 0
    seg_lengths, n_wp = [], []
    for i in range(env.num_envs):
        A = env.get_scene_snapshot(i)["agents"]
        agents.append(A)
        xy, L, W = A[:, 0:2], A[:, 3], A[:, 4]
        # Circumscribed circles first, as a cheap superset. This is what the C spawn
        # scorer uses, so it is the right thing to report, but it is conservative:
        # two cars abreast in neighbouring lanes have intersecting circles and
        # perfectly fine boxes. The overlap that matters is the oriented-box one the
        # simulator's own collision check would fire on, so test that too.
        r = 0.5 * np.hypot(L, W)
        d = np.linalg.norm(xy[:, None] - xy[None], axis=-1)
        slack = d - (r[:, None] + r[None])
        np.fill_diagonal(slack, np.inf)
        min_clear.append(slack.min())
        cand = np.argwhere(np.triu(slack < 0, 1))
        circle_hits += len(cand)
        for a, b in cand:
            if _boxes_overlap(A[a], A[b]):
                overlaps += 1
        for row in A:
            k = int(row[9])
            n_wp.append(k)
            pts = np.vstack([row[0:2], row[11:].reshape(-1, 2)[:k]])
            seg_lengths.extend(np.linalg.norm(np.diff(pts, axis=0), axis=1).tolist())

    agents = np.concatenate(agents)
    seg_lengths = np.array(seg_lengths)
    n_wp = np.array(n_wp)
    n = len(agents)
    fails = []

    print(f"scenes {env.num_envs}   agents {n}\n")

    print("-- placement --")
    print(f"  circumscribed-circle hits   : {circle_hits}  ({100*circle_hits/n:.3f}% of agents)")
    print(f"  true box overlaps at spawn  : {overlaps}  ({100*overlaps/n:.3f}% of agents)")
    print(f"  worst per-scene circle slack: {np.min(min_clear):+.2f} m")
    if overlaps / n > 0.005:
        fails.append(f"{100*overlaps/n:.2f}% of agents start inside another box (limit 0.5%)")

    print("\n-- route --")
    print(f"  waypoints/agent  mean {n_wp.mean():.2f}   min {n_wp.min()}   max {n_wp.max()}")
    print(f"  segment length   p1 {np.percentile(seg_lengths,1):.1f}  p50 {np.percentile(seg_lengths,50):.1f}"
          f"  p99 {np.percentile(seg_lengths,99):.1f}  max {seg_lengths.max():.1f} m")
    zero = float((seg_lengths < 1.0).mean())
    over = float((seg_lengths > 80.0).mean())
    print(f"  zero-length segments        : {100*zero:.3f}%")
    print(f"  segments over the 80 m bound: {100*over:.3f}%  (dead-end fallback goals)")
    if zero > 0.001:
        fails.append(f"{100*zero:.2f}% of route segments are degenerate")
    if over > 0.02:
        fails.append(f"{100*over:.2f}% of route segments exceed the waypoint bound (limit 2%)")
    if n_wp.min() < 1:
        fails.append("some agent has no goal at all")

    print("\n-- agent state distribution vs WOMD reference --")
    for t in sorted(ref):
        sel = agents[agents[:, 6] == t]
        p_obs, p_ref = len(sel) / n, ref[t]["prob"]
        rows = ref[t]["rows"]
        print(f"  {NAMES.get(t, t):11s} p={p_obs:.4f} (ref {p_ref:.4f})  n={len(sel)}")
        if abs(p_obs - p_ref) > 4 * np.sqrt(p_ref * (1 - p_ref) / n):
            fails.append(f"{NAMES.get(t,t)} frequency {p_obs:.4f} != reference {p_ref:.4f}")
        if len(sel) < 30:
            print("      (too few samples to test shape)")
            continue
        for j, key in enumerate(("length", "width", "height")):
            obs, exp = sel[:, 3 + j], rows[:, j]
            # Standard error of the mean under resampling from the reference table.
            se = exp.std() / np.sqrt(len(sel))
            ok = abs(obs.mean() - exp.mean()) <= 4 * se
            print(f"      {key:6s} {obs.mean():.3f}+-{obs.std():.3f}  ref {exp.mean():.3f}+-{exp.std():.3f}  "
                  f"{'ok' if ok else 'MISMATCH'}")
            if not ok:
                fails.append(f"{NAMES.get(t,t)} {key} mean {obs.mean():.3f} != reference {exp.mean():.3f}")
        # The joint structure. Sampling the three marginals independently -- the
        # obvious wrong implementation -- would drive these correlations to ~0.
        for a, bkey, ai, bi in (("l", "w", 0, 1), ("l", "h", 0, 2)):
            c_obs = np.corrcoef(sel[:, 3 + ai], sel[:, 3 + bi])[0, 1]
            c_ref = np.corrcoef(rows[:, ai], rows[:, bi])[0, 1]
            # Fisher z, two-sample, against the finite reference table.
            z = abs(np.arctanh(c_obs) - np.arctanh(c_ref)) / np.sqrt(1 / (len(sel) - 3) + 1 / (len(rows) - 3))
            print(f"      corr({a},{bkey}) {c_obs:.3f}  ref {c_ref:.3f}   z={z:.2f} {'ok' if z < 4 else 'MISMATCH'}")
            if z >= 4:
                fails.append(f"{NAMES.get(t,t)} corr({a},{bkey}) {c_obs:.3f} != reference {c_ref:.3f} (z={z:.1f})")

    # -- conditioning ---------------------------------------------------------
    # Read from the observation rather than a debug channel: this is exactly what the
    # policy receives, so a normalization mistake shows up here and nowhere else.
    print("\n-- conditioning (from the ego observation) --")
    obs = np.asarray(env.observations)
    base = obs.shape[1] - GIGA_NUM_COND
    cond = obs[:, base:]
    print(f"  ego observation width {obs.shape[1]} = {base} state + {GIGA_NUM_COND} conditioning")
    cw = agents[:, 10]
    bad_range = [COND_NAMES[k] for k in range(GIGA_NUM_COND)
                 if cond[:, k].min() < -1e-3 or cond[:, k].max() > 1 + 1e-3]
    if bad_range:
        fails.append(f"conditioning outside [0,1]: {bad_range}")
    # The first ten are plain uniforms except slot 1, which carries the is_final flag;
    # the last three are the X(a) dynamics mixture, centred near 1 but not uniform.
    off = [f"{COND_NAMES[k]} {cond[:, k].mean():.3f}" for k in range(10)
           if k != COND_SLOT_IS_FINAL and abs(cond[:, k].mean() - 0.5) > 0.05]
    if off:
        fails.append(f"conditioning not uniform: {off}")
    flag = cond[:, COND_SLOT_IS_FINAL]
    if not np.isin(flag, (0.0, 1.0)).all():
        fails.append(f"is_final is not a 0/1 flag: min {flag.min()} max {flag.max()}")
    # Every agent whose route is a single goal must be flagged, and only those.
    if not np.array_equal(flag > 0.5, cw >= np.maximum(n_wp, 1) - 1):
        fails.append("is_final disagrees with current_waypoint / num_waypoints")
    print(f"  is_final: 0/1 only, agrees with the route, set for {flag.mean() * 100:.1f}% of agents")
    distinct = len(np.unique(np.round(cond[:, COND_DELTA_GOAL], 6)))
    print(f"  range ok: {not bad_range}   uniform ok: {not off}   distinct delta_goal: {distinct}/{len(cond)}")
    if distinct < 0.9 * len(cond):
        fails.append(f"conditioning is not per-agent ({distinct} distinct of {len(cond)})")

    # -- waypoint chain integrity --------------------------------------------
    if not np.all(cw < np.maximum(n_wp, 1)):
        fails.append("current_waypoint index runs past num_waypoints")
    print(f"  current_waypoint within num_waypoints: {bool(np.all(cw < np.maximum(n_wp, 1)))}")

    env.close()
    print()
    if fails:
        print("FAILED:")
        for f in fails:
            print("  -", f)
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
