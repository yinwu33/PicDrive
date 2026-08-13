"""CARLA-to-simulation paired-data pipeline.

This package is the producer half only.  It writes the *same* artifact tree as
``data_utils.waymo_sim2real``, so teacher-feature extraction, visualization,
verification and distillation training all run against a CARLA dataset without
a single change.
"""

from __future__ import annotations

__all__ = ["EPISODE_FRAMES", "FRAME_INTERVAL_MICROS", "REAL_HEIGHT", "REAL_WIDTH"]

# One CARLA episode is exactly one simulator episode: ``episode_length = 91``
# steps at ``dt = 0.1`` in config/ocean/drive_3cam.ini.
EPISODE_FRAMES = 91
FRAME_INTERVAL_MICROS = 100_000

# Waymo's 1920x1280 cameras are resized to this in preprocess.py; matching it
# keeps `RealPerception` and the audit PNG geometry identical across sources.
REAL_HEIGHT = 256
REAL_WIDTH = 384
