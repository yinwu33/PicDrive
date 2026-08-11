"""Derive PufferDrive's ego observation vector from a Waymo pose sequence.

The frozen planning head distilled in :mod:`train_distillation` consumes the
same 11-D ego vector the simulator writes in ``compute_observations``
(``EGO_FEATURES_JERK`` in ``drive.h``).  Waymo Perception logs do not label the
ego vehicle at all, so the vector has to be reconstructed from the per-frame
vehicle pose.  This module is the single definition of that reconstruction; it
is pure NumPy so it can be unit-tested without the simulator.

Layout, mirroring ``compute_observations``:

===  ===========================================================
  0  relative goal x in the ego frame, scaled by 0.005
  1  relative goal y in the ego frame, scaled by 0.005
  2  signed speed / MAX_SPEED
  3  box width / MAX_VEH_WIDTH
  4  box length / MAX_VEH_LEN
  5  collision flag (always 0 on a logged human drive)
  6  steering angle / pi
  7  longitudinal acceleration, asymmetrically normalized
  8  lateral acceleration / JERK_LAT[2]
  9  respawn flag (always 0)
 10  entity type / 3 (VEHICLE)
===  ===========================================================

Derived quantities are clipped to the ranges the JERK integrator can actually
reach before being normalized.  A logged human drive occasionally leaves them
-- Waymo brakes harder than -5 m/s^2 at times -- and feeding the head values it
never saw in self-play would move it off the manifold the distillation is
supposed to measure.
"""

from __future__ import annotations

import numpy as np

from .processed import (
    EGO_OBS_DIM,
    GOAL_OBS_SCALE,
    GOAL_TARGET_DISTANCE,
    JERK_LAT_MAX,
    JERK_LONG_MAX,
    JERK_LONG_MIN,
    MAX_SPEED,
    MAX_VEH_LEN,
    MAX_VEH_WIDTH,
    WAYMO_REAR_AXLE_TO_BOX_CENTER,
    WAYMO_SDC_LENGTH,
    WAYMO_SDC_WIDTH,
)

# Reachable state ranges of the JERK integrator in drive.h, applied so the
# reconstructed vector stays inside the planning head's training distribution.
SPEED_RANGE = (-2.0, 20.0)
ACCEL_LONG_RANGE = (-5.0, 2.5)
ACCEL_LAT_RANGE = (-JERK_LAT_MAX, JERK_LAT_MAX)
STEERING_LIMIT = 0.55
# Curvature is omega / v, so it explodes as the vehicle stops. Below this speed
# the sim's own steering integrator is equally meaningless; report zero instead.
CURVATURE_SPEED_FLOOR = 0.5

VEHICLE_TYPE = 1


def _yaw_from_poses(poses: np.ndarray) -> np.ndarray:
    return np.arctan2(poses[:, 1, 0], poses[:, 0, 0])


def _box_centers(poses: np.ndarray) -> np.ndarray:
    """Waymo poses sit at the rear axle; the simulator's entity is box-centred."""
    yaw = _yaw_from_poses(poses)
    offset = WAYMO_REAR_AXLE_TO_BOX_CENTER * np.stack([np.cos(yaw), np.sin(yaw)], axis=1)
    return poses[:, :2, 3] + offset


def _goal_points(centers: np.ndarray, yaw: np.ndarray, goal_distance: float) -> np.ndarray:
    """The ego's own logged pose ``goal_distance`` metres further along its path.

    The simulator picks a lane-centre point that far ahead; on a real log the
    driven path is the closest available stand-in.  Segments that never travel
    that far -- a fifth of Waymo's training set is a car waiting at a light --
    are extended along the final heading so the goal stays roughly a fixed
    distance ahead instead of collapsing onto the vehicle.
    """
    steps = np.linalg.norm(np.diff(centers, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(steps)])
    targets = arc + goal_distance
    # ``arc`` is non-decreasing, so the first frame at or beyond each target is a
    # search rather than a scan.
    index = np.searchsorted(arc, targets, side="left")
    reached = index < len(arc)
    goals = np.empty_like(centers)
    goals[reached] = centers[np.clip(index[reached], 0, len(arc) - 1)]
    if (~reached).any():
        remaining = (targets[~reached] - arc[-1])[:, None]
        forward = np.stack([np.cos(yaw[-1]), np.sin(yaw[-1])])
        goals[~reached] = centers[-1] + remaining * forward
    return goals


def ego_observations(
    poses: np.ndarray,
    timestamps_micros: np.ndarray,
    *,
    length: float = WAYMO_SDC_LENGTH,
    width: float = WAYMO_SDC_WIDTH,
    goal_distance: float = GOAL_TARGET_DISTANCE,
) -> np.ndarray:
    """Return the ``[N, 11]`` ego observation table for one segment.

    ``poses`` are the ``[N, 4, 4]`` vehicle-to-global transforms of consecutive
    frames and ``timestamps_micros`` their capture times.  Frames must be in
    chronological order; that is how they appear in a TFRecord.
    """
    poses = np.asarray(poses, dtype=np.float64)
    times = np.asarray(timestamps_micros, dtype=np.float64) / 1e6
    if poses.ndim != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"poses must be [N,4,4], got {poses.shape}")
    if len(times) != len(poses):
        raise ValueError(f"{len(poses)} poses for {len(times)} timestamps")
    if len(poses) == 0:
        return np.zeros((0, EGO_OBS_DIM), dtype=np.float32)
    if np.any(np.diff(times) <= 0):
        raise ValueError("frame timestamps must be strictly increasing")

    yaw = _yaw_from_poses(poses)
    centers = _box_centers(poses)
    heading = np.stack([np.cos(yaw), np.sin(yaw)], axis=1)

    if len(poses) == 1:
        # A single frame carries no derivatives; only the static fields are real.
        velocity = np.zeros((1, 2))
        yaw_rate = np.zeros(1)
    else:
        velocity = np.stack([np.gradient(centers[:, axis], times) for axis in (0, 1)], axis=1)
        yaw_rate = np.gradient(np.unwrap(yaw), times)

    speed = np.copysign(np.linalg.norm(velocity, axis=1), np.sum(velocity * heading, axis=1))
    speed = np.clip(speed, *SPEED_RANGE)
    accel_long = np.gradient(speed, times) if len(poses) > 1 else np.zeros(1)
    accel_long = np.clip(accel_long, *ACCEL_LONG_RANGE)
    accel_lat = np.clip(speed * yaw_rate, *ACCEL_LAT_RANGE)

    moving = np.abs(speed) >= CURVATURE_SPEED_FLOOR
    curvature = np.where(moving, yaw_rate / np.where(moving, speed, 1.0), 0.0)
    steering = np.clip(np.arctan(curvature * 0.6 * length), -STEERING_LIMIT, STEERING_LIMIT)

    goals = _goal_points(centers, yaw, goal_distance)
    delta = goals - centers
    relative_goal = np.stack(
        [
            delta[:, 0] * heading[:, 0] + delta[:, 1] * heading[:, 1],
            -delta[:, 0] * heading[:, 1] + delta[:, 1] * heading[:, 0],
        ],
        axis=1,
    )

    obs = np.zeros((len(poses), EGO_OBS_DIM), dtype=np.float64)
    obs[:, 0] = relative_goal[:, 0] * GOAL_OBS_SCALE
    obs[:, 1] = relative_goal[:, 1] * GOAL_OBS_SCALE
    obs[:, 2] = speed / MAX_SPEED
    obs[:, 3] = width / MAX_VEH_WIDTH
    obs[:, 4] = length / MAX_VEH_LEN
    obs[:, 5] = 0.0  # the logged drive never collides
    obs[:, 6] = steering / np.pi
    obs[:, 7] = np.where(accel_long < 0, accel_long / (-JERK_LONG_MIN), accel_long / JERK_LONG_MAX)
    obs[:, 8] = accel_lat / JERK_LAT_MAX
    obs[:, 9] = 0.0  # nothing respawns on a real log
    obs[:, 10] = VEHICLE_TYPE / 3.0
    if not np.isfinite(obs).all():
        raise ValueError("ego observation contains non-finite values")
    return obs.astype(np.float32)
