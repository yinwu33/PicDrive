"""Render the Gigaflow random initialization to PNG for inspection.

This exists to be *looked at*. Random spawn placement and route generation are easy
to get subtly wrong in ways that no scalar metric catches -- an agent facing down the
wrong lane, a goal on the far side of a wall, a footprint sized like nothing on the
road -- and all of them are obvious in a picture.

Each map produces one PNG showing:
  * the road network, by element type;
  * every agent drawn as its true length x width footprint at its spawn pose, with a
    heading arrow, coloured by class;
  * a dashed line from each agent to its goal, via its intermediate waypoints.

A summary PNG additionally compares the sampled size distribution against the WOMD
reference, which is the check that the joint (length, width, height) distribution
survived sampling.

Usage:
    .venv/bin/python -m pufferlib.teddy.drive.viz \\
        --num-maps 12 --agents-per-map 60 --seed 0 --out-dir /tmp/teddy_viz
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")  # no display on the training box
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon

from pufferlib.teddy.drive.drive import Drive

VEHICLE, PEDESTRIAN, CYCLIST = 1, 2, 3
ROAD_LANE, ROAD_LINE, ROAD_EDGE, STOP_SIGN, CROSSWALK, SPEED_BUMP, DRIVEWAY = 4, 5, 6, 7, 8, 9, 10

# Road styling. Lane centerlines are drawn faint and dashed because they are a map
# abstraction rather than paint on asphalt -- but they are the thing agents are
# sampled along, so leaving them out would hide what the check is about.
ROAD_STYLE = {
    ROAD_EDGE: dict(color="#111111", lw=1.6, ls="-", z=2, label="road edge"),
    ROAD_LINE: dict(color="#e8b923", lw=0.9, ls="-", z=2, label="road line"),
    ROAD_LANE: dict(color="#7f8c8d", lw=0.5, ls=(0, (4, 4)), z=1, label="lane centerline"),
    CROSSWALK: dict(color="#2980b9", lw=1.1, ls="-", z=2, label="crosswalk"),
    SPEED_BUMP: dict(color="#8e44ad", lw=1.1, ls="-", z=2, label="speed bump"),
    DRIVEWAY: dict(color="#95a5a6", lw=0.6, ls=":", z=1, label="driveway"),
    STOP_SIGN: dict(color="#c0392b", lw=1.4, ls="-", z=2, label="stop sign"),
}
AGENT_STYLE = {
    VEHICLE: dict(face="#c0392b", label="vehicle"),
    PEDESTRIAN: dict(face="#27ae60", label="pedestrian"),
    CYCLIST: dict(face="#f39c12", label="cyclist"),
}
# Measured over the WOMD training corpus by data_utils/womd/build_agent_dist.py.
WOMD_REFERENCE = {
    VEHICLE: dict(length=(4.702, 0.566), width=(2.088, 0.167), height=(1.721, 0.261), corr_lw=0.861),
    PEDESTRIAN: dict(length=(0.885, 0.100), width=(0.839, 0.073), height=(1.711, 0.126), corr_lw=0.748),
    CYCLIST: dict(length=(1.809, 0.141), width=(0.835, 0.059), height=(1.761, 0.111), corr_lw=0.671),
}


def footprint(x, y, heading, length, width):
    """The four corners of an oriented bounding box, same convention as the sim."""
    ch, sh = np.cos(heading), np.sin(heading)
    hl, hw = length / 2.0, width / 2.0
    return np.array(
        [
            [x + hl * ch - hw * sh, y + hl * sh + hw * ch],
            [x + hl * ch + hw * sh, y + hl * sh - hw * ch],
            [x - hl * ch + hw * sh, y - hl * sh - hw * ch],
            [x - hl * ch - hw * sh, y - hl * sh + hw * ch],
        ]
    )


def draw_scene(snap, out_path, seed):
    agents, road_xy, road_meta = snap["agents"], snap["road_xy"], snap["road_meta"]

    fig, ax = plt.subplots(figsize=(13, 13))
    ax.set_facecolor("#f7f7f5")

    seen_road = set()
    off = 0
    for road_type, n in road_meta:
        pts = road_xy[off : off + n]
        off += n
        st = ROAD_STYLE.get(int(road_type))
        if st is None or n < 2:
            continue
        ax.plot(pts[:, 0], pts[:, 1], color=st["color"], lw=st["lw"], ls=st["ls"], zorder=st["z"])
        seen_road.add(int(road_type))

    seen_agent = set()
    for row in agents:
        x, y, heading, length, width, _height, atype = row[0:7]
        atype = int(atype)
        st = AGENT_STYLE.get(atype, dict(face="#34495e", label=f"type {atype}"))
        n_wp = int(row[9])
        wps = row[11:].reshape(-1, 2)[:n_wp]

        # Spawn -> waypoints -> goal. Dashed so it reads as an intention, not a path
        # the agent has taken.
        route = np.vstack([[x, y], wps]) if n_wp else np.array([[x, y]])
        ax.plot(route[:, 0], route[:, 1], color=st["face"], lw=0.8, ls="--", alpha=0.55, zorder=3)
        if n_wp:
            # Intermediate waypoints small, the final goal ringed.
            if n_wp > 1:
                ax.plot(wps[:-1, 0], wps[:-1, 1], ".", color=st["face"], ms=3.5, alpha=0.8, zorder=4)
            ax.plot(wps[-1, 0], wps[-1, 1], "o", mfc="none", mec=st["face"], ms=7, mew=1.4, zorder=4)

        ax.add_patch(Polygon(footprint(x, y, heading, length, width), closed=True,
                             facecolor=st["face"], edgecolor="black", lw=0.4, alpha=0.85, zorder=5))
        # Heading arrow, scaled to the body so a pedestrian does not get a car-sized one.
        ax.arrow(x, y, np.cos(heading) * length * 0.8, np.sin(heading) * length * 0.8,
                 head_width=max(width * 0.45, 0.35), head_length=max(length * 0.3, 0.4),
                 fc="black", ec="black", lw=0.4, alpha=0.7, length_includes_head=True, zorder=6)
        seen_agent.add(atype)

    counts = {t: int((agents[:, 6] == t).sum()) for t in sorted(seen_agent)}
    n_wp_all = agents[:, 9]
    route_len = np.array(
        [np.linalg.norm(np.diff(np.vstack([r[0:2], r[11:].reshape(-1, 2)[: int(r[9])]]), axis=0), axis=1).sum()
         for r in agents]
    ) if len(agents) else np.zeros(0)

    handles = [Line2D([], [], color=ROAD_STYLE[t]["color"], lw=ROAD_STYLE[t]["lw"],
                      ls=ROAD_STYLE[t]["ls"], label=ROAD_STYLE[t]["label"]) for t in sorted(seen_road)]
    handles += [Line2D([], [], marker="s", ls="", mfc=AGENT_STYLE[t]["face"], mec="black",
                       label=f"{AGENT_STYLE[t]['label']} ({counts.get(t, 0)})") for t in sorted(seen_agent)
                if t in AGENT_STYLE]
    handles += [Line2D([], [], color="#555", lw=0.8, ls="--", label="spawn to goal"),
                Line2D([], [], marker="o", ls="", mfc="none", mec="#555", label="final goal")]
    ax.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.9)

    ax.set_aspect("equal")
    ax.set_title(
        f"map {snap['map_id']}  |  seed {seed}  |  {len(agents)} agents  |  "
        f"waypoints/agent {n_wp_all.mean():.2f}  |  route {route_len.mean():.0f} m mean",
        fontsize=11,
    )
    ax.set_xlabel("x (m, map-centered)")
    ax.set_ylabel("y (m, map-centered)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def draw_summary(all_agents, out_path, seed):
    """Sampled sizes against the WOMD reference, per type.

    The per-type scatter is the point of this figure: length and width are strongly
    correlated in the real data, and sampling the marginals independently -- the
    obvious wrong implementation -- shows up here as a round blob instead of an
    elongated one.
    """
    types = [t for t in (VEHICLE, PEDESTRIAN, CYCLIST) if (all_agents[:, 6] == t).any()]
    fig, axes = plt.subplots(2, len(types), figsize=(5 * len(types), 8), squeeze=False)

    for c, t in enumerate(types):
        sel = all_agents[all_agents[:, 6] == t]
        L, W, H = sel[:, 3], sel[:, 4], sel[:, 5]
        ref = WOMD_REFERENCE[t]
        name = AGENT_STYLE[t]["label"]

        ax = axes[0][c]
        ax.scatter(L, W, s=6, alpha=0.35, color=AGENT_STYLE[t]["face"])
        corr = np.corrcoef(L, W)[0, 1] if len(sel) > 2 else float("nan")
        ax.set_title(f"{name} n={len(sel)}\ncorr(l,w)={corr:.3f}  (WOMD {ref['corr_lw']:.3f})", fontsize=10)
        ax.set_xlabel("length (m)")
        ax.set_ylabel("width (m)")

        ax = axes[1][c]
        for arr, key, color in ((L, "length", "#c0392b"), (W, "width", "#2980b9"), (H, "height", "#27ae60")):
            ax.hist(arr, bins=40, alpha=0.5, color=color, label=f"{key} {arr.mean():.2f}±{arr.std():.2f}")
            mu, sd = ref[key]
            ax.axvline(mu, color=color, ls="--", lw=1.2)
        ax.legend(fontsize=8)
        ax.set_xlabel("m   (dashed = WOMD mean)")

    fig.suptitle(f"sampled agent state vs WOMD reference (seed {seed}, n={len(all_agents)})", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num-maps", type=int, default=12)
    ap.add_argument("--agents-per-map", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--map-dir", default="resources/drive/binaries/training")
    ap.add_argument("--num-map-pool", type=int, default=10000, help="maps to draw from")
    ap.add_argument("--out-dir", default="/tmp/teddy_viz")
    ap.add_argument("--wrong-way-frac", type=float, default=0.0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Pinning agents_per_map to a single value makes the packing deterministic:
    # exactly num_maps scenes of exactly agents_per_map agents.
    env = Drive(
        num_agents=args.num_maps * args.agents_per_map,
        num_maps=args.num_map_pool,
        map_dir=args.map_dir,
        obs_mode="render_state",
        dynamics_model="jerk",
        episode_length=1280,
        seed=args.seed,
        agents_per_map_min=args.agents_per_map,
        agents_per_map_max=args.agents_per_map,
        wrong_way_frac=args.wrong_way_frac,
        render_mode=1,
    )
    env.reset(seed=args.seed)
    print(f"packed {env.num_envs} scenes x {args.agents_per_map} agents")

    all_agents = []
    for i in range(env.num_envs):
        snap = env.get_scene_snapshot(i)
        out = os.path.join(args.out_dir, f"map_{snap['map_id']:04d}_seed{args.seed}.png")
        draw_scene(snap, out, args.seed)
        all_agents.append(snap["agents"])
        print(f"  wrote {out}  ({len(snap['agents'])} agents, {len(snap['road_meta'])} road polylines)")

    all_agents = np.concatenate(all_agents)
    summary = os.path.join(args.out_dir, f"summary_seed{args.seed}.png")
    draw_summary(all_agents, summary, args.seed)
    print(f"  wrote {summary}")
    env.close()


if __name__ == "__main__":
    main()
