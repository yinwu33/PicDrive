"""Simulator-free control, ego observation and route tracking for CARLA.

Split out of :mod:`demo.closed_loop` so a rollout host that is not this repository's
own evaluator -- the Bench2Drive leaderboard agent, in particular -- can drive
the same policy through the same adapter without importing ``pufferlib`` or the
teacher renderer.  Everything here is numpy and the ``carla`` actor API only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from data_utils.sim2real.waymo.giga_conditioning import GIGA_EGO_OBS_DIM, conditioning_to_raw
from data_utils.sim2real.waymo.processed import (
    GOAL_OBS_SCALE,
    JERK_LAT_MAX,
    JERK_LONG_MAX,
    JERK_LONG_MIN,
    MAX_SPEED,
    MAX_VEH_LEN,
    MAX_VEH_WIDTH,
)

from .roads import world_to_ego


def box_center(transform, offset) -> tuple[float, float]:
    """World box centre in the right-handed frame, y negated out of CARLA's."""
    yaw = math.radians(transform.rotation.yaw)
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    location = transform.location
    # Only the yaw term matters: pitch and roll of a grounded vehicle move the
    # box centre by millimetres, and the renderer is planar anyway.
    x = location.x + offset[0] * cos_yaw - offset[1] * sin_yaw
    y = location.y + offset[0] * sin_yaw + offset[1] * cos_yaw
    return x, -y


def spectator_transform(ego_transform, carla, *, distance: float = 8.0, height: float = 4.0):
    """A stable chase pose behind the ego for CARLA's spectator camera.

    Only useful when something is actually rendering a window; the evaluation
    server normally runs ``-RenderOffScreen`` and nothing reads this.
    """

    forward = ego_transform.get_forward_vector()
    location = carla.Location(
        x=ego_transform.location.x - distance * forward.x,
        y=ego_transform.location.y - distance * forward.y,
        z=ego_transform.location.z + height,
    )
    rotation = carla.Rotation(pitch=-15.0, yaw=ego_transform.rotation.yaw, roll=0.0)
    return carla.Transform(location, rotation)


def carla_mount(
    pos: tuple[float, float, float],
    pitch_deg: float,
    yaw_deg: float,
    roll_deg: float,
    box_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Convert a rig camera pose into a CARLA attachment. See :func:`rig.mount`.

    Kept here rather than in :mod:`rig` so a host without ``pufferlib`` can
    place the same sensors from a serialized rig.
    """

    x, y, z = pos
    return (
        (box_offset[0] + x, box_offset[1] - y, z),
        (-pitch_deg, -yaw_deg, roll_deg),
    )


JERK_LONG = np.asarray([-15.0, -4.0, 0.0, 4.0], dtype=np.float32)
JERK_LAT = np.asarray([-4.0, 0.0, 4.0], dtype=np.float32)
STEERING_LIMIT = 0.55
STEERING_RATE = 0.6
SPEED_RANGE = (-2.0, 20.0)
ACCEL_LONG_RANGE = (-5.0, 2.5)
ACCEL_LAT_RANGE = (-4.0, 4.0)


def decode_jerk_action(action: int) -> tuple[float, float]:
    """Decode the checkpoint's single 12-way action into simulator jerk bins."""

    action = int(action)
    if not 0 <= action < len(JERK_LONG) * len(JERK_LAT):
        raise ValueError(f"jerk action must be in [0, 11], got {action}")
    return float(JERK_LONG[action // len(JERK_LAT)]), float(JERK_LAT[action % len(JERK_LAT)])


@dataclass(frozen=True)
class ControlCommand:
    throttle: float
    brake: float
    steer: float
    reverse: bool
    target_speed: float
    target_accel_long: float
    target_accel_lat: float
    target_steering_angle: float
    jerk_long: float
    jerk_lat: float


class JerkController:
    """Translate giga's jerk integrator state into CARLA low-level controls.

    The target acceleration, speed and steering states follow ``drive.h``.  A
    compact acceleration/speed feedback layer then realizes those targets with
    CARLA throttle/brake/steer.  This adapter is intentionally separate from the
    learned policy: it has no scene information and can be tuned from rollout
    telemetry without retraining perception.
    """

    def __init__(
        self,
        dt: float,
        conditioning: np.ndarray,
        *,
        accel_feedback: float = 0.35,
        speed_feedback: float = 0.8,
    ):
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")
        self.dt = float(dt)
        self.raw_conditioning = conditioning_to_raw(conditioning)
        self.accel_feedback = float(accel_feedback)
        self.speed_feedback = float(speed_feedback)
        self.reset()

    def reset(self, signed_speed: float = 0.0) -> None:
        self.accel_long = 0.0
        self.accel_lat = 0.0
        self.steering_angle = 0.0
        self.target_speed = float(np.clip(signed_speed, *SPEED_RANGE))
        self.reverse = self.target_speed < -0.1

    def step(
        self,
        action: int,
        *,
        signed_speed: float,
        measured_accel_long: float,
        wheelbase: float,
        max_wheel_steer: float,
    ) -> ControlCommand:
        jerk_long, jerk_lat = decode_jerk_action(action)
        jerk_long *= float(self.raw_conditioning[10])
        jerk_lat *= float(self.raw_conditioning[11])

        next_long = self.accel_long + jerk_long * self.dt
        if self.accel_long * next_long < 0:
            next_long = 0.0
        else:
            next_long = float(
                np.clip(next_long, ACCEL_LONG_RANGE[0], 2.5 * self.raw_conditioning[12])
            )

        next_lat = self.accel_lat + jerk_lat * self.dt
        if self.accel_lat * next_lat < 0:
            next_lat = 0.0
        else:
            next_lat = float(np.clip(next_lat, *ACCEL_LAT_RANGE))

        # The simulator integrates from the realized current speed, not from a
        # free-running command state. Doing the same prevents controller error
        # from accumulating into an unreachable target after a collision or a
        # long brake hold.
        speed_target = signed_speed + 0.5 * (self.accel_long + next_long) * self.dt
        if signed_speed * speed_target < 0:
            speed_target = 0.0
        speed_target = float(np.clip(speed_target, *SPEED_RANGE))

        curvature = next_lat / max(speed_target * speed_target, 1e-5)
        if abs(curvature) < 1e-5:
            curvature = math.copysign(1e-5, curvature if curvature else 1.0)
        requested_steer = math.atan(curvature * max(wheelbase, 0.1))
        steer_delta = float(
            np.clip(
                requested_steer - self.steering_angle,
                -STEERING_RATE * self.dt,
                STEERING_RATE * self.dt,
            )
        )
        steering_angle = float(
            np.clip(self.steering_angle + steer_delta, -STEERING_LIMIT, STEERING_LIMIT)
        )
        # drive.h recomputes lateral acceleration after the steering limiter.
        next_lat = speed_target * speed_target * math.tan(steering_angle) / max(wheelbase, 0.1)

        # Reverse is sticky around zero to avoid flipping CARLA's gearbox each
        # tick while a braking trajectory crosses rest.
        if self.reverse:
            self.reverse = speed_target < 0.15
        else:
            self.reverse = speed_target < -0.15 and signed_speed < 0.25

        if self.reverse:
            target_magnitude = max(0.0, -speed_target)
            measured_magnitude = max(0.0, -signed_speed)
            desired_accel = -next_long
            measured_accel = -measured_accel_long
            effort = (
                desired_accel
                + self.accel_feedback * (desired_accel - measured_accel)
                + self.speed_feedback * (target_magnitude - measured_magnitude)
            )
        else:
            effort = (
                next_long
                + self.accel_feedback * (next_long - measured_accel_long)
                + self.speed_feedback * (speed_target - signed_speed)
            )

        throttle = float(np.clip(effort / 2.5, 0.0, 1.0))
        brake = float(np.clip(-effort / 5.0, 0.0, 1.0))
        if abs(speed_target) < 0.08 and abs(signed_speed) < 0.25 and next_long <= 0:
            throttle = 0.0
            brake = max(brake, 0.35)
        # Policy-positive is left; CARLA control-positive is right.
        steer = float(np.clip(-steering_angle / max(max_wheel_steer, 0.1), -1.0, 1.0))

        self.accel_long = next_long
        self.accel_lat = next_lat
        self.steering_angle = steering_angle
        self.target_speed = speed_target
        return ControlCommand(
            throttle=throttle,
            brake=brake,
            steer=steer,
            reverse=self.reverse,
            target_speed=speed_target,
            target_accel_long=next_long,
            target_accel_lat=next_lat,
            target_steering_angle=steering_angle,
            jerk_long=jerk_long,
            jerk_lat=jerk_lat,
        )


def base_ego_observation(
    relative_goal: tuple[float, float] | np.ndarray,
    *,
    signed_speed: float,
    width: float,
    length: float,
    collision: bool,
    steering_angle: float,
    accel_long: float,
    accel_lat: float,
    respawn: bool,
) -> np.ndarray:
    """Build giga's normalized 11-D JERK observation from online telemetry."""

    relative_goal = np.asarray(relative_goal, dtype=np.float64)
    if relative_goal.shape != (2,):
        raise ValueError(f"relative_goal must be [2], got {relative_goal.shape}")
    signed_speed = float(np.clip(signed_speed, *SPEED_RANGE))
    accel_long = float(np.clip(accel_long, *ACCEL_LONG_RANGE))
    accel_lat = float(np.clip(accel_lat, *ACCEL_LAT_RANGE))
    steering_angle = float(np.clip(steering_angle, -STEERING_LIMIT, STEERING_LIMIT))
    obs = np.zeros(11, dtype=np.float32)
    obs[:2] = relative_goal * GOAL_OBS_SCALE
    obs[2] = signed_speed / MAX_SPEED
    obs[3] = width / MAX_VEH_WIDTH
    obs[4] = length / MAX_VEH_LEN
    obs[5] = float(collision)
    obs[6] = steering_angle / math.pi
    obs[7] = accel_long / (-JERK_LONG_MIN if accel_long < 0 else JERK_LONG_MAX)
    obs[8] = accel_lat / JERK_LAT_MAX
    obs[9] = float(respawn)
    obs[10] = 1.0 / 3.0
    if not np.isfinite(obs).all():
        raise ValueError("online ego observation contains non-finite values")
    return obs


@dataclass(frozen=True)
class EgoTelemetry:
    observation: np.ndarray
    center_right_handed: np.ndarray
    signed_speed: float
    accel_long: float
    accel_lat: float
    steering_angle: float
    wheelbase: float
    max_wheel_steer: float


def _vehicle_geometry(vehicle) -> tuple[float, float]:
    """Return wheelbase and maximum wheel angle in metres/radians."""

    length = 2.0 * vehicle.bounding_box.extent.x
    fallback_wheelbase = max(1.5, 0.6 * length)
    try:
        wheels = list(vehicle.get_physics_control().wheels)
        xs = [float(wheel.position.x) for wheel in wheels]
        # WheelPhysicsControl positions are centimetres in CARLA's API.
        wheelbase = (max(xs) - min(xs)) / 100.0 if len(xs) >= 2 else fallback_wheelbase
        steerable = [math.radians(float(wheel.max_steer_angle)) for wheel in wheels if wheel.max_steer_angle > 0]
        max_steer = max(steerable) if steerable else STEERING_LIMIT
        if not 1.0 <= wheelbase <= 6.0:
            wheelbase = fallback_wheelbase
        return wheelbase, max(max_steer, 0.1)
    except (AttributeError, RuntimeError, ValueError):
        return fallback_wheelbase, STEERING_LIMIT


def _physical_steering_angle(vehicle, carla, max_wheel_steer: float) -> float:
    try:
        locations = (carla.VehicleWheelLocation.FL_Wheel, carla.VehicleWheelLocation.FR_Wheel)
        angles = [math.radians(float(vehicle.get_wheel_steer_angle(location))) for location in locations]
        # CARLA-positive is right; policy-positive is left.
        return float(np.clip(-np.mean(angles), -STEERING_LIMIT, STEERING_LIMIT))
    except (AttributeError, RuntimeError):
        return float(
            np.clip(-vehicle.get_control().steer * max_wheel_steer, -STEERING_LIMIT, STEERING_LIMIT)
        )


def read_ego_telemetry(
    vehicle,
    goal_carla_xy: np.ndarray,
    conditioning: np.ndarray,
    *,
    collision: bool,
    respawn: bool,
    carla,
    geometry: tuple[float, float],
) -> EgoTelemetry:
    transform = vehicle.get_transform()
    velocity = vehicle.get_velocity()
    acceleration = vehicle.get_acceleration()
    forward = transform.get_forward_vector()
    right = transform.get_right_vector()
    signed_speed = velocity.x * forward.x + velocity.y * forward.y + velocity.z * forward.z
    accel_long = acceleration.x * forward.x + acceleration.y * forward.y + acceleration.z * forward.z
    # CARLA's right vector is policy-negative lateral.
    accel_lat = -(acceleration.x * right.x + acceleration.y * right.y + acceleration.z * right.z)
    center = np.asarray(
        box_center(
            transform,
            (
                vehicle.bounding_box.location.x,
                vehicle.bounding_box.location.y,
                vehicle.bounding_box.location.z,
            ),
        ),
        dtype=np.float64,
    )
    yaw = -math.radians(transform.rotation.yaw)
    goal_right_handed = np.asarray([goal_carla_xy[0], -goal_carla_xy[1]], dtype=np.float64)
    relative_goal = world_to_ego(goal_right_handed, center, yaw)
    wheelbase, max_wheel_steer = geometry
    steering = _physical_steering_angle(vehicle, carla, max_wheel_steer)
    base = base_ego_observation(
        relative_goal,
        signed_speed=signed_speed,
        width=2.0 * vehicle.bounding_box.extent.y,
        length=2.0 * vehicle.bounding_box.extent.x,
        collision=collision,
        steering_angle=steering,
        accel_long=accel_long,
        accel_lat=accel_lat,
        respawn=respawn,
    )
    observation = np.concatenate((base, conditioning)).astype(np.float32, copy=False)
    if observation.shape != (GIGA_EGO_OBS_DIM,):
        raise AssertionError(f"expected {GIGA_EGO_OBS_DIM}-D ego vector, got {observation.shape}")
    return EgoTelemetry(
        observation=observation,
        center_right_handed=center,
        signed_speed=float(np.clip(signed_speed, *SPEED_RANGE)),
        accel_long=float(np.clip(accel_long, *ACCEL_LONG_RANGE)),
        accel_lat=float(np.clip(accel_lat, *ACCEL_LAT_RANGE)),
        steering_angle=steering,
        wheelbase=wheelbase,
        max_wheel_steer=max_wheel_steer,
    )


@dataclass(frozen=True)
class RoutePlan:
    points: np.ndarray
    arc_length: np.ndarray
    goal_indices: tuple[int, ...]

    @classmethod
    def from_trace(cls, trace: list[tuple[Any, Any]], max_goals: int) -> RoutePlan:
        if max_goals < 1:
            raise ValueError("max_goals must be positive")
        points = np.asarray(
            [
                [item[0].transform.location.x, item[0].transform.location.y, item[0].transform.location.z]
                for item in trace
            ],
            dtype=np.float64,
        )
        if len(points) < 2:
            raise ValueError("global route contains fewer than two waypoints")
        step = np.linalg.norm(np.diff(points, axis=0), axis=1)
        arc = np.concatenate(([0.0], np.cumsum(step)))
        if arc[-1] <= 0:
            raise ValueError("global route has zero length")
        num_goals = min(max_goals, max(1, math.ceil(arc[-1] / 50.0)))
        # Pin the final entry to the final route sample. Repeated zero-length
        # samples at a destination make searchsorted choose an earlier equal-arc
        # index, which otherwise appends an accidental (max_goals + 1)-th goal.
        desired = np.linspace(arc[-1] / num_goals, arc[-1], num_goals)
        indices = tuple(
            int(min(len(points) - 1, np.searchsorted(arc, distance)))
            for distance in desired[:-1]
        ) + (len(points) - 1,)
        # Searchsorted can collapse adjacent goals on a sparse route.
        indices = tuple(dict.fromkeys(indices))
        if indices[-1] != len(points) - 1:
            indices = (*indices, len(points) - 1)
        return cls(points=points, arc_length=arc, goal_indices=indices)

    @property
    def length(self) -> float:
        return float(self.arc_length[-1])


class RouteTracker:
    """Monotonic route progress plus a small sequence of policy-visible goals."""

    def __init__(self, plan: RoutePlan):
        self.plan = plan
        self.route_index = 0
        self.goal_cursor = 0
        self.progress = 0.0
        self.deviation = math.inf

    @property
    def current_goal(self) -> np.ndarray:
        return self.plan.points[self.plan.goal_indices[self.goal_cursor]]

    @property
    def final_goal(self) -> np.ndarray:
        return self.plan.points[self.plan.goal_indices[-1]]

    @property
    def at_final_goal(self) -> bool:
        return self.goal_cursor == len(self.plan.goal_indices) - 1

    @property
    def completion(self) -> float:
        return float(np.clip(self.progress / self.plan.length, 0.0, 1.0))

    def update(self, carla_xyz: np.ndarray, intermediate_radius: float) -> int:
        location = np.asarray(carla_xyz, dtype=np.float64)
        start = max(0, self.route_index - 5)
        # 250 samples is 500 m at the default route resolution and avoids
        # jumping to a later copy of the same road on looping routes.
        stop = min(len(self.plan.points), self.route_index + 250)
        distances = np.linalg.norm(self.plan.points[start:stop, :2] - location[:2], axis=1)
        nearest = start + int(np.argmin(distances))
        self.route_index = max(self.route_index, nearest)
        self.progress = max(self.progress, float(self.plan.arc_length[self.route_index]))
        self.deviation = float(distances[nearest - start])

        advanced = 0
        while not self.at_final_goal:
            target_index = self.plan.goal_indices[self.goal_cursor]
            target_distance = float(np.linalg.norm(self.plan.points[target_index, :2] - location[:2]))
            if self.progress < self.plan.arc_length[target_index] - intermediate_radius and target_distance > intermediate_radius:
                break
            self.goal_cursor += 1
            advanced += 1
        return advanced

    def distance_to_goal(self, carla_xyz: np.ndarray) -> float:
        return float(np.linalg.norm(self.current_goal[:2] - np.asarray(carla_xyz)[:2]))
