"""Deterministic offline conditioning for a frozen giga planning head."""

from __future__ import annotations

import hashlib

import numpy as np


GIGA_CONDITIONING_DIM = 13
GIGA_EGO_OBS_DIM = 11 + GIGA_CONDITIONING_DIM


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
