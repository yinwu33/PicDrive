"""Roll one car around one map and print where every unit of reward came from.

The training env runs 1024 agents over thousands of maps and reports rates, which is
the wrong instrument for the question "why is this policy driving like that". Here a
single agent runs on a single map with the *same* C simulator and the *same* config
the trainer uses, and the per-step reward is read back split into the nine Gigaflow
terms it was summed from (see the GIGA_DBG_* block in drive.h).

The terms are not recomputed in Python -- they are the values drive.h added up -- so
the printed decomposition adds up to the reward the trainer would have seen, and the
consistency check at the bottom of the summary proves it.

Usage:
    .venv/bin/python -m pufferlib.giga.drive.trace_agent --steps 400
    .venv/bin/python -m pufferlib.giga.drive.trace_agent \\
        --policy experiments/puffer_giga.pt --map-id 3 --steps 1280 \\
        --csv /tmp/trace.csv --plot /tmp/trace.png
"""

import argparse
import os
import sys

import numpy as np

from pufferlib.giga.drive import binding
from pufferlib.giga.drive.drive import DEBUG_FEATURE_NAMES, Drive

# Columns that are a reward contribution rather than state.
TERMS = [n for n in DEBUG_FEATURE_NAMES if n.startswith("r_")]
COL = {name: i for i, name in enumerate(DEBUG_FEATURE_NAMES)}
COLLISION_STATE = {0: "-", 1: "veh", 2: "off"}


def build_env(args):
    """One agent, one map, otherwise the training configuration untouched.

    The env kwargs come from the same ini the trainer loads rather than from Drive's
    defaults, because a trace taken under different reward weights or a different
    dynamics model would be a trace of a different environment.
    """
    from pufferlib.pufferl import load_config

    # load_config builds its own argparse over the ini keys and would choke on this
    # script's flags.
    argv, sys.argv = sys.argv, sys.argv[:1]
    try:
        config = load_config(args.env_name)
    finally:
        sys.argv = argv

    env_kwargs = dict(config["env"])
    for key in ("num_agents", "num_maps", "resample_frequency", "render_mode"):
        env_kwargs.pop(key, None)
    env_kwargs.update(
        num_agents=args.num_agents,
        num_maps=1,
        # A scene is resampled every resample_frequency steps in training, which would
        # swap the map out from under a long trace.
        resample_frequency=0,
        agents_per_map_min=1,
        agents_per_map_max=1,
        force_map_id=args.map_id,
        seed=args.seed,
        ini_file=config.get("config_path"),
    )
    if args.episode_length is not None:
        env_kwargs["episode_length"] = args.episode_length
    if args.goal_behavior is not None:
        env_kwargs["goal_behavior"] = args.goal_behavior

    env = Drive(**env_kwargs)
    return env, config


def build_policy(args, env, config):
    """A checkpoint, or None for the random-action baseline."""
    if args.policy in (None, "random"):
        return None, None, None

    import torch

    from pufferlib.pufferl import load_policy

    config = dict(config)
    config["load_model_path"] = args.policy
    config["load_id"] = None
    # load_policy only ever touches vecenv.driver_env.
    shim = type("VecEnvShim", (), {"driver_env": env})()
    policy = load_policy(config, shim, args.env_name)
    policy.eval()
    device = config["train"]["device"]
    state = {}
    if config["train"]["use_rnn"]:
        state = dict(
            done=torch.zeros(env.num_agents, device=device),
            lstm_h=torch.zeros(env.num_agents, policy.hidden_size, device=device),
            lstm_c=torch.zeros(env.num_agents, policy.hidden_size, device=device),
        )
    return policy, state, device


def rollout(env, policy, state, device, steps, seed):
    import torch

    import pufferlib.pytorch

    trace = env.enable_debug_trace()
    obs, _ = env.reset(seed=seed)
    # Grab the scene now: the roads never move, and the route is the one the agent
    # starts with (a respawn would draw a new one).
    snap = env.get_scene_snapshot(0)
    row = snap["agents"][0]
    n_wp = int(row[9])
    snap["waypoints"] = row[11:11 + 2 * binding.MAX_WAYPOINTS].reshape(-1, 2)[:n_wp]
    rows = []
    n_actions = int(env.single_action_space.nvec[0])
    rng = np.random.default_rng(seed)

    for _ in range(steps):
        if policy is None:
            action = rng.integers(0, n_actions, size=(env.num_agents, 1)).astype(np.int32)
        else:
            with torch.no_grad():
                ob = torch.as_tensor(obs).to(device)
                logits, _ = policy.forward_eval(ob, state)
                action, _, _ = pufferlib.pytorch.sample_logits(logits)
                action = action.cpu().numpy().reshape(env.num_agents, -1).astype(np.int32)

        obs, reward, terminal, truncated, _ = env.step(action)
        # Copy: the C env overwrites the trace in place on the next step.
        rows.append(trace.copy())
        if state:
            # LSTMWrapper zeroes the recurrent state wherever this is set, so a
            # respawned or reset agent does not start with the last life's memory.
            state["done"] = torch.as_tensor(terminal | truncated).to(device)

    return np.stack(rows), snap  # [steps, agents, features]


def summarize(rec, agent, top_steps):
    a = rec[:, agent, :]
    steps = a.shape[0]
    total = a[:, COL["reward"]].sum()
    term_sums = {t: a[:, COL[t]].sum() for t in TERMS}
    gross = sum(abs(v) for v in term_sums.values()) or 1.0

    print(f"\n=== agent {agent}: {steps} steps, total reward {total:+.4f} "
          f"({total / steps:+.5f}/step) ===\n")
    print(f"{'term':<14}{'sum':>12}{'per step':>12}{'share':>9}{'steps hit':>11}")
    for t, v in sorted(term_sums.items(), key=lambda kv: -abs(kv[1])):
        hits = int((a[:, COL[t]] != 0).sum())
        print(f"{t:<14}{v:>12.4f}{v / steps:>12.5f}{abs(v) / gross * 100:>8.1f}%{hits:>11}")

    residual = np.abs(a[:, COL["reward"]] - sum(a[:, COL[t]] for t in TERMS)).max()
    print(f"\nmax |reward - sum(terms)| = {residual:.3e}  (0 means the split is complete)")

    off = a[:, COL["collision_state"]] == 2
    veh = a[:, COL["collision_state"]] == 1
    lane = a[:, COL["lane_valid"]] > 0
    print(f"offroad steps      {off.sum():>6} ({off.mean() * 100:5.1f}%)")
    print(f"collision steps    {veh.sum():>6} ({veh.mean() * 100:5.1f}%)")
    print(f"on lane graph      {lane.sum():>6} ({lane.mean() * 100:5.1f}%)")
    print(f"goals reached      {a[-1, COL['goals_reached']]:>6.0f}   respawns {a[-1, COL['respawn_count']]:.0f}")
    print(f"speed  mean {a[:, COL['speed']].mean():5.2f}  max {a[:, COL['speed']].max():5.2f} m/s")
    print(f"dist to goal  first {a[0, COL['dist_to_goal']]:6.1f}  last {a[-1, COL['dist_to_goal']]:6.1f} m")

    conds = [n for n in DEBUG_FEATURE_NAMES if n.startswith("cond_")]
    print("\nconditioning (redrawn on every respawn; shown at the last step)")
    print("  " + "  ".join(f"{c[5:]}={a[-1, COL[c]]:.4g}" for c in conds))

    if top_steps:
        print(f"\nfirst {top_steps} steps")
        cols = ["reward", "speed", "dist_to_goal", "is_final", "lane_heading_err", "action"]
        print(f"{'t':>5}{'coll':>6}" + "".join(f"{c:>20}" for c in cols))
        for k in range(min(top_steps, steps)):
            state = COLLISION_STATE.get(int(a[k, COL["collision_state"]]), "?")
            print(f"{a[k, COL['timestep']]:>5.0f}{state:>6}"
                  + "".join(f"{a[k, COL[c]]:>20.4f}" for c in cols))


def write_csv(rec, agent, path):
    a = rec[:, agent, :]
    np.savetxt(path, a, delimiter=",", header=",".join(DEBUG_FEATURE_NAMES), comments="", fmt="%.6g")
    print(f"\nwrote {path}  [{a.shape[0]} steps x {a.shape[1]} columns]")


def plot(rec, agent, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    a = rec[:, agent, :]
    t = a[:, COL["timestep"]]
    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)

    ax = axes[0]
    for term in TERMS:
        v = a[:, COL[term]]
        if np.any(v != 0):
            ax.plot(t, np.cumsum(v), lw=1.2, label=term)
    ax.plot(t, np.cumsum(a[:, COL["reward"]]), lw=2.0, color="k", label="total")
    ax.set_ylabel("cumulative reward")
    ax.legend(fontsize=7, ncol=3)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(t, a[:, COL["speed"]], lw=1.0, label="speed")
    ax.plot(t, a[:, COL["signed_v"]], lw=0.8, ls="--", label="signed v")
    ax.set_ylabel("m/s")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(t, a[:, COL["dist_to_goal"]], lw=1.0, color="#2980b9")
    for k in np.flatnonzero(a[:, COL["r_goal"]] > 0):
        ax.axvline(t[k], color="#27ae60", lw=0.8, alpha=0.7)
    ax.set_ylabel("dist to goal (m)\ngreen = goal paid")
    ax.grid(alpha=0.3)

    ax = axes[3]
    ax.plot(t, a[:, COL["lane_lateral_offset"]], lw=1.0, label="lateral offset (m)")
    ax.plot(t, a[:, COL["lane_heading_err"]], lw=1.0, label="heading err (rad)")
    off = a[:, COL["collision_state"]] == 2
    veh = a[:, COL["collision_state"]] == 1
    ax.plot(t[off], np.zeros(off.sum()), "|", color="#c0392b", ms=12, label="offroad")
    ax.plot(t[veh], np.zeros(veh.sum()), "|", color="#8e44ad", ms=12, label="collision")
    ax.set_ylabel("lane frame")
    ax.set_xlabel("timestep")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


def draw_roads(ax, snap, label=False):
    """The map's polylines, styled by element type (same palette as viz.py)."""
    from pufferlib.giga.drive.viz import ROAD_STYLE

    pt = 0
    seen = set()
    for rtype, count in snap["road_meta"]:
        pts = snap["road_xy"][pt:pt + count]
        pt += count
        st = ROAD_STYLE.get(int(rtype))
        if st is None:
            continue
        name = st["label"] if (label and st["label"] not in seen) else None
        ax.plot(pts[:, 0], pts[:, 1], color=st["color"], lw=st["lw"], ls=st["ls"], zorder=st["z"], label=name)
        seen.add(st["label"])


def video(rec, agent, snap, path, fps, title):
    """An mp4 of the rollout: the car on the map, with the reward split live.

    The static plot answers "which term dominated the episode"; this answers "what was
    the car doing at the moment that term fired", which is the part no aggregate and no
    time series makes obvious -- a reversing car and a forward one trace the same line.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    from matplotlib.collections import LineCollection
    from matplotlib.colors import TwoSlopeNorm

    from pufferlib.giga.drive.viz import footprint

    a = rec[:, agent, :]
    x, y, r = a[:, COL["x"]], a[:, COL["y"]], a[:, COL["reward"]]
    t, heading = a[:, COL["timestep"]], a[:, COL["heading"]]
    steps = len(a)
    wps = snap.get("waypoints")
    active = [term for term in TERMS if np.any(a[:, COL[term]] != 0)]

    fig = plt.figure(figsize=(14, 7.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1], height_ratios=[3, 1], hspace=0.3, wspace=0.16)
    ax = fig.add_subplot(gs[:, 0])
    ax_bar = fig.add_subplot(gs[0, 1])
    ax_t = fig.add_subplot(gs[1, 1])

    draw_roads(ax, snap)
    if wps is not None and len(wps):
        ax.plot(wps[:, 0], wps[:, 1], "o--", color="#16a085", ms=8, lw=1.2, zorder=6)
    half = max(max(np.ptp(x), np.ptp(y)) / 2 * 1.35, 18.0)
    cx, cy = (x.min() + x.max()) / 2, (y.min() + y.max()) / 2
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal")
    ax.grid(alpha=0.2)

    lim = max(np.percentile(np.abs(r), 98), 1e-6)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim)
    trail = LineCollection([], cmap="RdYlGn", norm=norm, linewidths=3.5, zorder=5)
    ax.add_collection(trail)
    fig.colorbar(trail, ax=ax, orientation="horizontal", fraction=0.04, pad=0.05,
                 label=f"reward on that step (clipped at +/-{lim:.3f})")
    car = plt.Polygon(footprint(x[0], y[0], heading[0], 4.7, 2.1), closed=True, fill=True,
                      fc="#2980b9", ec="#1b4f72", alpha=0.75, zorder=9)
    ax.add_patch(car)
    nose, = ax.plot([], [], color="#1b4f72", lw=2.2, zorder=10)
    hit = ax.scatter([], [], s=90, marker="x", color="#c0392b", zorder=11)
    head = ax.set_title("", fontsize=10, loc="left", family="monospace")

    # Current-step contributions. Fixed symmetric log-ish scale so bars do not rescale
    # every frame, which would make them impossible to compare across time.
    bar_lim = max(np.abs(a[:, [COL[c] for c in active]]).max(), 1e-6)
    ypos = np.arange(len(active))
    bars = ax_bar.barh(ypos, np.zeros(len(active)), color="#c0392b")
    ax_bar.set_yticks(ypos)
    ax_bar.set_yticklabels(active, fontsize=9)
    ax_bar.invert_yaxis()
    ax_bar.set_xscale("symlog", linthresh=1e-5)
    ax_bar.set_xlim(-bar_lim * 1.1, bar_lim * 1.1)
    # The default symlog locator crowds a dozen labels around the linear threshold.
    decades = [10.0 ** e for e in range(-5, 1) if 10.0 ** e <= bar_lim * 1.1]
    ax_bar.set_xticks([-d for d in reversed(decades)] + [0] + decades)
    ax_bar.tick_params(axis="x", labelsize=7)
    ax_bar.axvline(0, color="k", lw=0.8)
    ax_bar.set_title("this step's reward, by term (symlog)", fontsize=9)
    ax_bar.grid(alpha=0.3, axis="x")

    ax_t.plot(t, np.cumsum(r), color="k", lw=1.2)
    ax_t.set_xlim(t[0], t[-1])
    ax_t.set_ylabel("cumulative", fontsize=9)
    ax_t.set_xlabel("timestep", fontsize=9)
    ax_t.grid(alpha=0.3)
    cursor = ax_t.axvline(t[0], color="#c0392b", lw=1.2)

    writer = FFMpegWriter(fps=fps, bitrate=2400, metadata=dict(title=title))
    with writer.saving(fig, path, dpi=110):
        for k in range(steps):
            if k:
                seg = np.stack([np.column_stack([x[:k], y[:k]]),
                                np.column_stack([x[1:k + 1], y[1:k + 1]])], axis=1)
                trail.set_segments(seg)
                trail.set_array(r[:k])
            car.set_xy(footprint(x[k], y[k], heading[k], 4.7, 2.1))
            nose.set_data([x[k], x[k] + 5 * np.cos(heading[k])], [y[k], y[k] + 5 * np.sin(heading[k])])
            state = int(a[k, COL["collision_state"]])
            hit.set_offsets(np.array([[x[k], y[k]]]) if state else np.empty((0, 2)))
            for bar, term in zip(bars, active):
                bar.set_width(a[k, COL[term]])
                bar.set_color("#c0392b" if a[k, COL[term]] < 0 else "#27ae60")
            cursor.set_xdata([t[k], t[k]])
            head.set_text(
                f"{title}\nt={int(t[k]):4d}  reward {r[k]:+.4f}  cum {r[:k + 1].sum():+8.3f}\n"
                f"v={a[k, COL['signed_v']]:+5.2f} m/s  a_long={a[k, COL['a_long']]:+5.2f}  "
                f"goal {a[k, COL['dist_to_goal']]:5.1f} m  {COLLISION_STATE.get(state, '?')}")
            writer.grab_frame()
    plt.close(fig)
    print(f"wrote {path}  [{steps} frames @ {fps} fps]")


def scene_plot(rec, agent, snap, path, annotate_every, title, frame="traj"):
    """The trajectory on the map, coloured by reward, beside the per-step term values.

    Three views of the same numbers, because no single one answers the question:
    *where* the reward was paid (map), *how much* each term was worth on each step
    (symlog lines -- the terms span four orders of magnitude, so a shared linear axis
    shows only the largest), and *when* each term was active (heatmap normalized per
    row, which is the only way they share one colour scale).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import TwoSlopeNorm

    from pufferlib.giga.drive.viz import ROAD_STYLE, footprint

    a = rec[:, agent, :]
    x, y, r = a[:, COL["x"]], a[:, COL["y"]], a[:, COL["reward"]]
    t = a[:, COL["timestep"]]
    steps = len(a)
    wps = snap.get("waypoints")
    active = [term for term in TERMS if np.any(a[:, COL[term]] != 0)]

    fig = plt.figure(figsize=(14.5, 14))
    gs = fig.add_gridspec(3, 2, height_ratios=[2.5, 1.1, 1.1], width_ratios=[1.75, 1], hspace=0.3, wspace=0.02)
    ax = fig.add_subplot(gs[0, 0])

    draw_roads(ax, snap, label=True)

    # --- trajectory, coloured by the reward paid on each step ----------------
    seg = np.stack([np.column_stack([x[:-1], y[:-1]]), np.column_stack([x[1:], y[1:]])], axis=1)
    lim = max(np.percentile(np.abs(r), 98), 1e-6)  # one offroad spike would flatten everything else
    lc = LineCollection(seg, cmap="RdYlGn", norm=TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim),
                        linewidths=3.5, zorder=5)
    lc.set_array(r[:-1])
    ax.add_collection(lc)
    # Under the map, not beside it: a vertical colorbar would run into the totals.
    fig.colorbar(lc, ax=ax, orientation="horizontal", fraction=0.045, pad=0.06,
                 label=f"reward on that step (clipped at +/-{lim:.3f})")

    off = a[:, COL["collision_state"]] == 2
    veh = a[:, COL["collision_state"]] == 1
    goal = a[:, COL["r_goal"]] > 0
    ax.scatter(x[off], y[off], s=46, marker="x", color="#c0392b", zorder=7, label="offroad")
    ax.scatter(x[veh], y[veh], s=46, marker="P", color="#8e44ad", zorder=7, label="collision")
    ax.scatter(x[goal], y[goal], s=170, marker="*", color="#16a085", zorder=8, label="goal paid")

    for k, edge in ((0, "#2c3e50"), (steps - 1, "#7f8c8d")):
        ax.add_patch(plt.Polygon(footprint(x[k], y[k], a[k, COL["heading"]], 4.7, 2.1),
                                 closed=True, fill=False, lw=1.6, edgecolor=edge, zorder=9))
    # Heading at the start: under reverse driving the path alone does not show which
    # way the car was pointing, which is exactly the question here.
    ax.arrow(x[0], y[0], 7 * np.cos(a[0, COL["heading"]]), 7 * np.sin(a[0, COL["heading"]]),
             head_width=1.5, color="#2c3e50", zorder=10, label="start heading")
    ax.annotate("start", (x[0], y[0]), textcoords="offset points", xytext=(8, 8), fontsize=9, zorder=11)
    ax.annotate("end", (x[-1], y[-1]), textcoords="offset points", xytext=(8, -14), fontsize=9, zorder=11)

    if wps is not None and len(wps):
        ax.plot(wps[:, 0], wps[:, 1], "o--", color="#16a085", ms=9, lw=1.2, zorder=6, label="route / goal")
        for i, (wx, wy) in enumerate(wps):
            ax.annotate(f"wp{i}", (wx, wy), textcoords="offset points", xytext=(7, 7),
                        fontsize=9, color="#16a085", zorder=11)

    # A WOMD crop is ~300 m across; framing the whole map renders the trajectory as a
    # dot. "traj" frames what was driven, "route" also keeps every waypoint in view.
    fx, fy = (x, y) if frame == "traj" or wps is None or not len(wps) else (
        np.concatenate([x, wps[:, 0]]), np.concatenate([y, wps[:, 1]]))
    half = max(np.ptp(fx), np.ptp(fy)) / 2
    half = max(half * 1.25, 15.0)
    cx, cy = (fx.min() + fx.max()) / 2, (fy.min() + fy.max()) / 2
    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)

    if annotate_every > 0:
        # Step index and the reward paid there, spaced by distance rather than by step
        # count: a car that is barely moving would stack every label on one point.
        last, min_gap, flip = None, half / 6.0, 1
        for k in range(0, steps, annotate_every):
            if last is not None and np.hypot(x[k] - last[0], y[k] - last[1]) < min_gap:
                continue
            ax.annotate(f"t{int(t[k])}  {r[k]:+.3f}", (x[k], y[k]), textcoords="offset points",
                        xytext=(6, 6 * flip), fontsize=7.5, color="#2c3e50", zorder=12,
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))
            last, flip = (x[k], y[k]), -flip

    ax.set_aspect("equal")
    ax.set_title(title, fontsize=11, loc="left")
    ax.legend(fontsize=7, loc="upper left", ncol=2, framealpha=0.85)
    ax.grid(alpha=0.2)

    # --- totals, next to the map ---------------------------------------------
    info = fig.add_subplot(gs[0, 1])
    info.axis("off")
    lines = [f"total reward   {r.sum():+.4f}   ({r.sum() / steps:+.5f}/step)", "",
             f"{'term':<12}{'sum':>10}{'steps':>7}"]
    for term in sorted(active, key=lambda n: -abs(a[:, COL[n]].sum())):
        v = a[:, COL[term]]
        lines.append(f"{term:<12}{v.sum():>10.4f}{int((v != 0).sum()):>7}")
    lines += ["", f"{'offroad steps':<16}{int(off.sum()):>5}  ({off.mean() * 100:.0f}%)",
              f"{'collision steps':<16}{int(veh.sum()):>5}  ({veh.mean() * 100:.0f}%)",
              f"{'reversing':<16}{int((a[:, COL['signed_v']] < 0).sum()):>5}  "
              f"({(a[:, COL['signed_v']] < 0).mean() * 100:.0f}%)",
              f"{'goals reached':<16}{a[-1, COL['goals_reached']]:>5.0f}",
              f"{'mean speed':<16}{a[:, COL['speed']].mean():>5.2f} m/s",
              f"{'dist to goal':<16}{a[0, COL['dist_to_goal']]:>5.1f} -> {a[-1, COL['dist_to_goal']]:.1f} m", ""]
    conds = [n for n in DEBUG_FEATURE_NAMES if n.startswith("cond_")]
    lines.append("conditioning (last step)")
    lines += [f"  {c[5:]:<18}{a[-1, COL[c]]:>9.4g}" for c in conds]
    info.text(0, 1, "\n".join(lines), family="monospace", fontsize=8, va="top", ha="left",
              transform=info.transAxes)

    # --- per-step value of every term ----------------------------------------
    ax2 = fig.add_subplot(gs[1, :])
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(active), 1)))
    for term, c in zip(active, colors):
        ax2.plot(t, a[:, COL[term]], lw=1.1, color=c, label=term)
    ax2.plot(t, r, lw=0.9, color="k", ls="--", label="total")
    # The terms span 1e-6 to 1e0; a linear axis would show only r_offroad.
    ax2.set_yscale("symlog", linthresh=1e-5)
    ax2.set_ylabel("reward on each step\n(symlog)")
    ax2.legend(fontsize=7, ncol=9, loc="lower left", bbox_to_anchor=(0, 1.01, 1, 0.15), mode="expand",
               borderaxespad=0, frameon=False)
    ax2.grid(alpha=0.3)

    # --- when each term is active --------------------------------------------
    ax3 = fig.add_subplot(gs[2, :])
    rows, labels = [], []
    for term in active:
        v = a[:, COL[term]]
        scale = np.abs(v).max()
        rows.append(v / scale)
        labels.append(f"{term}  (peak {scale:.4f})")
    ax3.imshow(np.array(rows), aspect="auto", cmap="RdYlGn", vmin=-1, vmax=1,
               extent=[t[0], t[-1], len(rows) - 0.5, -0.5], interpolation="nearest")
    ax3.set_yticks(range(len(labels)))
    ax3.set_yticklabels(labels, fontsize=8)
    ax3.set_xlabel("timestep")
    ax3.set_title("each term normalized by its own peak -- red = paying, green = earning", fontsize=9)

    fig.savefig(path, dpi=115, bbox_inches="tight")
    print(f"wrote {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env-name", default="puffer_giga", help="config to take the env settings from")
    p.add_argument("--policy", default=None, help="checkpoint path, or omit for random actions")
    p.add_argument("--map-id", type=int, default=0, help="map_%%03d.bin to run on")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--num-agents", type=int, default=1, help="cars in the scene; 1 is the debug case")
    p.add_argument("--agent", type=int, default=0, help="which agent to report on")
    p.add_argument("--episode-length", type=int, default=None, help="override the config episode length")
    p.add_argument("--goal-behavior", type=int, default=None, help="0 respawn, 1 new goals, 2 stop")
    p.add_argument("--print-steps", type=int, default=20, help="rows of the per-step table (0 to skip)")
    p.add_argument("--csv", default=None)
    p.add_argument("--plot", default=None, help="time-series PNG")
    p.add_argument("--scene-plot", default=None, help="map view: the trajectory coloured by reward")
    p.add_argument("--annotate-every", type=int, default=25,
                   help="print the step index and its reward on the map every N steps (0 to skip)")
    p.add_argument("--video", default=None, help="mp4 of the rollout with the reward split live")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--frame", choices=["traj", "route"], default="traj",
                   help="frame the map on what was driven, or wide enough to keep every waypoint")
    args = p.parse_args()

    env, config = build_env(args)
    print(f"map {os.path.basename(env.map_name if hasattr(env, 'map_name') else '')} "
          f"map_id={args.map_id} agents={env.num_agents} envs={env.num_envs} "
          f"episode_length={env.episode_length} goal_behavior={env.goal_behavior} "
          f"policy={'random' if args.policy in (None, 'random') else args.policy}")

    policy, state, device = build_policy(args, env, config)
    rec, snap = rollout(env, policy, state, device, args.steps, args.seed)
    summarize(rec, args.agent, args.print_steps)
    if args.csv:
        write_csv(rec, args.agent, args.csv)
    if args.plot:
        plot(rec, args.agent, args.plot)
    title = (f"map_{args.map_id:03d}  seed {args.seed}  agent {args.agent}  {args.steps} steps  "
             f"policy={'random' if args.policy in (None, 'random') else os.path.basename(args.policy)}")
    if args.scene_plot:
        scene_plot(rec, args.agent, snap, args.scene_plot, args.annotate_every, title, args.frame)
    if args.video:
        video(rec, args.agent, snap, args.video, args.fps, title)
    env.close()


if __name__ == "__main__":
    main()
