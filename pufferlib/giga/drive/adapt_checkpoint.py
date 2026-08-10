"""Widen an ocean DriveCam checkpoint so it loads into the giga policy.

The two policies are the same network (`ocean/torch.py` and `giga/torch.py` are
identical apart from the class names), but giga appends the per-agent
conditioning vector to the ego observation, so `ego_encoder.0` is wider on the
giga side:

    ocean, jerk:  [rel_goal_x, rel_goal_y, signed_speed, w, l, collided,
                   steer, a_long, a_lat, respawning, type]              -> 11
    giga,  jerk:  the same eleven, then GIGA_NUM_COND = 13 conditioning
                  values normalized to [0, 1] (conditioning.h)          -> 24

Because the conditioning block is appended rather than interleaved (see the
`cond_base` loop in `compute_observations`), the transferred columns keep their
meaning and only the tail is new. The new columns are zeroed, not randomized: at
step zero the widened policy computes exactly what the source checkpoint did,
whatever conditioning is sampled, and it learns to use the block from there.
Random init would instead inject noise scaled by the conditioning into a network
that is already trained.

Usage:
    python -m pufferlib.giga.drive.adapt_checkpoint SRC.pt DST.pt [--env puffer_giga_3cam]
"""

import argparse
import sys

import torch

from pufferlib.giga.drive import binding
from pufferlib.pufferl import load_config

EGO_WIDTHS = {"jerk": binding.EGO_FEATURES_JERK, "classic": binding.EGO_FEATURES_CLASSIC}
EGO_LAYER = "ego_encoder.0.weight"


def target_ego_width(env_name):
    """Ego width the giga env actually emits, read from the compiled binding."""
    # load_config parses sys.argv for its own dynamic --section.key flags, so it
    # would reject this script's positional arguments. Hide them for the call
    # rather than reimplementing the config lookup here.
    argv, sys.argv = sys.argv, sys.argv[:1]
    try:
        dynamics = load_config(env_name)["env"]["dynamics_model"]
    finally:
        sys.argv = argv
    if dynamics not in EGO_WIDTHS:
        raise SystemExit(f"unknown dynamics_model {dynamics!r}; expected one of {sorted(EGO_WIDTHS)}")
    return EGO_WIDTHS[dynamics], dynamics


def adapt(src, dst, env_name):
    state = torch.load(src, map_location="cpu", weights_only=False)
    state = {k.replace("module.", ""): v for k, v in state.items()}

    width, dynamics = target_ego_width(env_name)
    if EGO_LAYER not in state:
        raise SystemExit(f"{src} has no {EGO_LAYER}; is it a DriveCam checkpoint?")

    weight = state[EGO_LAYER]
    have = weight.shape[1]
    if have == width:
        print(f"{EGO_LAYER} is already {have} wide -- nothing to do")
    elif have > width:
        raise SystemExit(
            f"{EGO_LAYER} is {have} wide but {env_name} ({dynamics}) emits {width}. "
            "Truncating would drop trained features, so this is left to you."
        )
    else:
        widened = weight.new_zeros((weight.shape[0], width))
        widened[:, :have] = weight
        state[EGO_LAYER] = widened
        print(f"{EGO_LAYER}: {tuple(weight.shape)} -> {tuple(widened.shape)}, "
              f"columns {have}..{width - 1} zeroed (conditioning block)")

    torch.save(state, dst)
    print(f"wrote {dst}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--env", default="puffer_giga_3cam", help="target giga env name")
    a = ap.parse_args()
    adapt(a.src, a.dst, a.env)
