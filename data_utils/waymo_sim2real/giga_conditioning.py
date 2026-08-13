"""Deterministic offline conditioning for a frozen giga planning head."""

from __future__ import annotations

import hashlib

import numpy as np


GIGA_CONDITIONING_DIM = 13
GIGA_EGO_OBS_DIM = 11 + GIGA_CONDITIONING_DIM

# Keep these in the exact COND_* order of pufferlib/giga/drive/conditioning.h.
# They are public because an external vehicle controller needs the raw dynamics
# coefficients represented by the final three normalized policy inputs.
GIGA_CONDITIONING_LOW = np.asarray(
    [
        2.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.00025,
        0.0,
        0.00025,
        -0.5,
        0.00025,
        0.8,
        0.8,
        2.0 / 3.0,
    ],
    dtype=np.float32,
)
GIGA_CONDITIONING_HIGH = np.asarray(
    [12.0, 20.0, 3.0, 3.0, 0.05, 0.025, 1.0, 0.075, 0.5, 0.0075, 1.25, 1.25, 1.5],
    dtype=np.float32,
)


def _xmix_normalized(rng: np.random.Generator, scale: float) -> float:
    """Sample Gigaflow X(a), then normalize through conditioning.h bounds."""

    low = 1.0 / scale
    raw = rng.uniform(low, 1.0) if rng.integers(2) else rng.uniform(1.0, scale)
    return float((raw - low) / (scale - low))


def segment_conditioning(segment_id: str, seed: int = 42) -> np.ndarray:
    """One reproducible normalized conditioning draw for a recorded episode.

    Giga samples conditioning once per agent life. CARLA/Waymo logs contain no
    such counterfactual control vector, so the offline analogue assigns one
    independent draw to the whole segment. The first ten normalized dimensions
    are uniform; the final dynamics dimensions preserve Gigaflow's X(a)
    mixture rather than replacing it with a uniform approximation.
    """

    digest = hashlib.sha256(f"giga-conditioning:{seed}:{segment_id}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    conditioning = np.empty(GIGA_CONDITIONING_DIM, dtype=np.float32)
    conditioning[:10] = rng.random(10, dtype=np.float32)
    conditioning[10] = _xmix_normalized(rng, 1.25)
    conditioning[11] = _xmix_normalized(rng, 1.25)
    conditioning[12] = _xmix_normalized(rng, 1.5)
    return conditioning


def append_giga_conditioning(ego: np.ndarray, segment_id: str, seed: int = 42) -> np.ndarray:
    if ego.shape[-1] != 11:
        raise ValueError(f"base ego observation must be 11-D, got {ego.shape[-1]}")
    conditioning = segment_conditioning(segment_id, seed)
    return np.concatenate((ego.astype(np.float32, copy=False), conditioning)).astype(
        np.float32, copy=False
    )


def conditioning_to_raw(conditioning: np.ndarray) -> np.ndarray:
    """Invert the simulator's per-dimension normalization."""

    conditioning = np.asarray(conditioning, dtype=np.float32)
    if conditioning.shape[-1] != GIGA_CONDITIONING_DIM:
        raise ValueError(
            f"conditioning must end in {GIGA_CONDITIONING_DIM}, got {conditioning.shape}"
        )
    if not np.isfinite(conditioning).all() or (conditioning < 0).any() or (conditioning > 1).any():
        raise ValueError("normalized conditioning must be finite and inside [0, 1]")
    return GIGA_CONDITIONING_LOW + conditioning * (
        GIGA_CONDITIONING_HIGH - GIGA_CONDITIONING_LOW
    )


def nominal_conditioning() -> np.ndarray:
    """A stable evaluation driver with nominal (1.0) actuation coefficients.

    Reward preferences sit at the midpoint of their training ranges.  The final
    three values are chosen so denormalization gives exactly C_throttle=1,
    C_steer=1 and C_acc=1, rather than the midpoint of asymmetric bounds.
    """

    raw = (GIGA_CONDITIONING_LOW + GIGA_CONDITIONING_HIGH) * 0.5
    raw[10:] = 1.0
    return ((raw - GIGA_CONDITIONING_LOW) / (GIGA_CONDITIONING_HIGH - GIGA_CONDITIONING_LOW)).astype(
        np.float32
    )
