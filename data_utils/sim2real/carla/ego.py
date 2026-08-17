"""The 11-D ego observation for a CARLA episode.

The vector is not rebuilt here.  ``sim2real.waymo.ego_state.ego_observations``
already encodes every normalizer and every clip that keeps the reconstruction
inside the JERK integrator's reachable set, and it is the exact code the Waymo
branch runs.  Sharing it is the point: a bias in the finite differences is then
common to both sources instead of being one more thing the student has to
absorb when it moves between them.

Two arguments differ from the Waymo call:

* ``axle_to_center=0.0`` -- CARLA's actor transform plus ``bounding_box.location``
  is already the box centre, so there is no rear-axle offset to undo.
* ``goal`` -- the episode's own final box centre, held fixed.  See the note in
  ``ego_observations``: with ``goal_behavior = 0`` the frozen head trained on a
  fixed endpoint that decays toward the vehicle, not a receding lookahead.
"""

from __future__ import annotations

import numpy as np

from data_utils.sim2real.waymo.ego_state import ego_observations


def pose_matrices(centers: np.ndarray, yaws: np.ndarray) -> np.ndarray:
    """Pack right-handed box centres and headings into ``[N, 4, 4]`` transforms.

    ``ego_observations`` reads only the 2x2 rotation block and the xy
    translation, but it type-checks the full 4x4 that Waymo hands it.
    """
    centers = np.asarray(centers, dtype=np.float64)
    yaws = np.asarray(yaws, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError(f"centers must be [N,2], got {centers.shape}")
    if yaws.shape != (len(centers),):
        raise ValueError(f"{len(centers)} centers for {yaws.shape} yaws")
    poses = np.tile(np.eye(4, dtype=np.float64), (len(centers), 1, 1))
    cos_yaw, sin_yaw = np.cos(yaws), np.sin(yaws)
    poses[:, 0, 0], poses[:, 0, 1] = cos_yaw, -sin_yaw
    poses[:, 1, 0], poses[:, 1, 1] = sin_yaw, cos_yaw
    poses[:, :2, 3] = centers
    return poses


def episode_ego_obs(
    centers: np.ndarray,
    yaws: np.ndarray,
    timestamps_micros: np.ndarray,
    *,
    length: float,
    width: float,
) -> np.ndarray:
    """Return the ``[N, 11]`` ego table for one recorded episode."""
    poses = pose_matrices(centers, yaws)
    return ego_observations(
        poses,
        timestamps_micros,
        length=length,
        width=width,
        axle_to_center=0.0,
        goal=poses[-1, :2, 3],
    )
