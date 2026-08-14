"""Closed-loop CARLA evaluation for distilled camera perception.

The policy split is the same one used during distillation::

    CARLA RGB -> student 256-D scene latent ----\
                                                   frozen ego encoder/trunk/LSTM/actor
    CARLA ground truth -> teacher scene latent --/

Only one branch controls the ego.  The other can run as a shadow on the exact
same trajectory, with its own persistent recurrent state, to measure feature
and action disagreement.  Traffic Manager controls background vehicles and
CARLA AI controls pedestrians.

The CARLA server must already be running.  With the repository's local 0.9.16
wheel, expose only the PythonAPI directory (not an older ``dist`` egg)::

    PYTHONPATH=/home/tjhu78u/CARLA_0_9_16/PythonAPI/carla \
    .venv/bin/python -m data_utils.carla_sim2real.closed_loop \
        --student artifacts/waymo_sim2real/runs/dino_waymo_2hz_b32_e30/deployment.pt \
        --checkpoint experiments/skynet/model_puffer_giga_3cam_001400.pt \
        --output artifacts/carla_sim2real/eval/student_town01 \
        --town Town01 --episodes 10 --control student
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import random
import sys
import time
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from data_utils.waymo_sim2real.giga_conditioning import (
    GIGA_EGO_OBS_DIM,
    conditioning_to_raw,
    good_conditioning,
    nominal_conditioning,
    segment_conditioning,
)
from data_utils.waymo_sim2real.processed import (
    GOAL_OBS_SCALE,
    JERK_LAT_MAX,
    JERK_LONG_MAX,
    JERK_LONG_MIN,
    MAX_SPEED,
    MAX_VEH_LEN,
    MAX_VEH_WIDTH,
    SIM_HEIGHT,
    SIM_WIDTH,
)
from data_utils.waymo_sim2real.real_perception import load_deployment_bundle
from data_utils.waymo_sim2real.render_roads import prepare_runtime_roads
from data_utils.waymo_sim2real.teacher import (
    RecurrentPlanningRuntime,
    load_frozen_planning_head,
    load_teacher,
    scene_features,
    sha256_file,
)

from . import REAL_HEIGHT, REAL_WIDTH
from .collect import DEFAULT_DT, DEFAULT_RADIUS, Collector, Weather, _box_center, _decode
from .rig import rig_array, rig_cameras, sensor_intrinsics
from .roads import world_to_ego


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
        _box_center(
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


class EpisodeEvents:
    def __init__(self):
        self.collisions = 0
        self.collision_impulse = 0.0
        self.lane_invasions = 0
        self.last_collision_frame = -1

    def on_collision(self, event) -> None:
        self.collisions += 1
        impulse = event.normal_impulse
        self.collision_impulse += math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.last_collision_frame = int(event.frame)

    def on_lane_invasion(self, event) -> None:
        self.lane_invasions += max(1, len(event.crossed_lane_markings))


class TeacherRenderer:
    def __init__(self, device: torch.device):
        self.device = device
        self.rig = torch.from_numpy(rig_array()).to(device)

    @torch.inference_mode()
    def render(self, agents: np.ndarray, roads: np.ndarray) -> torch.Tensor:
        agents_t = torch.from_numpy(np.ascontiguousarray(agents)).to(self.device)
        roads_t = torch.from_numpy(prepare_runtime_roads(roads)).to(self.device)
        egos = torch.tensor([[0.0, 0.0, 1.0, 0.0, -1.0]], device=self.device)
        if self.device.type == "cuda":
            from pufferlib.ocean.drive import raster_cuda

            return raster_cuda.render(
                agents=agents_t,
                roads=roads_t,
                egos=egos,
                cameras=rig_cameras(),
                rig=self.rig,
            )
        from pufferlib.ocean.drive import raster_ref

        return raster_ref.render(agents_t, roads_t, egos, cameras=rig_cameras())


def puffer_preview_images(rendered: torch.Tensor) -> np.ndarray:
    """Move one Puffer raster batch into display-order-agnostic RGB arrays."""

    expected = (1, 3, 3, SIM_HEIGHT, SIM_WIDTH)
    if tuple(rendered.shape) != expected or rendered.dtype != torch.uint8:
        raise ValueError(
            f"Puffer preview expects uint8 raster shaped {expected}, got "
            f"{rendered.dtype} {tuple(rendered.shape)}"
        )
    return rendered[0].permute(0, 2, 3, 1).contiguous().cpu().numpy()


class PolicyBranches:
    """Student/teacher perceptions feeding independent copies of LSTM state."""

    def __init__(self, args):
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        self.device = torch.device(args.device)
        self.control = args.control
        self.need_student = args.control == "student" or args.shadow
        self.need_teacher = args.control == "teacher" or args.shadow
        if self.need_student and args.student is None:
            raise ValueError("student control/shadow requires --student DEPLOYMENT_PT")

        head = load_frozen_planning_head(args.checkpoint, self.device, require_recurrent=True)
        if head.action_dims != [12]:
            raise ValueError(f"closed-loop jerk control expects one 12-way action, got {head.action_dims}")
        if head.ego_features != GIGA_EGO_OBS_DIM:
            raise ValueError(
                f"checkpoint planner expects {head.ego_features} ego values, expected {GIGA_EGO_OBS_DIM}"
            )
        self.student_runtime = RecurrentPlanningRuntime(head) if self.need_student else None
        self.teacher_runtime = RecurrentPlanningRuntime(head) if self.need_teacher else None
        self.student = (
            load_deployment_bundle(args.student, self.device) if self.need_student else None
        )
        self.teacher = load_teacher(args.checkpoint, self.device) if self.need_teacher else None
        # Camera preview always includes the exact Puffer raster row, even when
        # the teacher branch itself is disabled. When the teacher is live the
        # same tensor is reused for perception instead of rendering twice.
        self.renderer = (
            TeacherRenderer(self.device)
            if self.need_teacher or getattr(args, "camera_preview", False)
            else None
        )
        self.amp_dtype = {
            "bf16": torch.bfloat16,
            "fp16": torch.float16,
            "off": None,
        }[args.amp]

    def reset(self) -> None:
        if self.student_runtime is not None:
            self.student_runtime.reset()
        if self.teacher_runtime is not None:
            self.teacher_runtime.reset()

    @torch.inference_mode()
    def step(
        self,
        real_images: np.ndarray,
        ego: np.ndarray,
        *,
        agents: np.ndarray | None,
        roads: np.ndarray | None,
        rendered_teacher_images: torch.Tensor | None = None,
    ) -> tuple[int, dict[str, float | int]]:
        started = time.perf_counter()
        latents: dict[str, torch.Tensor] = {}
        logits: dict[str, list[torch.Tensor]] = {}
        autocast_enabled = self.amp_dtype is not None and self.device.type == "cuda"

        if self.student is not None:
            images = torch.from_numpy(real_images).permute(0, 3, 1, 2).unsqueeze(0).to(self.device)
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=autocast_enabled,
            ):
                latents["student"] = self.student(images).float()

        if self.teacher is not None:
            sim_images = rendered_teacher_images
            if sim_images is None:
                if agents is None or roads is None:
                    raise ValueError("teacher perception requires live agents and roads")
                if self.renderer is None:
                    raise AssertionError("teacher perception has no Puffer renderer")
                sim_images = self.renderer.render(agents, roads)
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=autocast_enabled,
            ):
                latents["teacher"] = scene_features(self.teacher, sim_images).float()

        ego_t = torch.from_numpy(ego).unsqueeze(0).to(self.device)
        if self.student_runtime is not None:
            logits["student"] = self.student_runtime.step(latents["student"], ego_t)
        if self.teacher_runtime is not None:
            logits["teacher"] = self.teacher_runtime.step(latents["teacher"], ego_t)
        actions = {name: int(parts[0].argmax(dim=-1).item()) for name, parts in logits.items()}
        metrics: dict[str, float | int] = {
            f"{name}_action": action for name, action in actions.items()
        }
        if "student" in latents and "teacher" in latents:
            metrics["feature_mse"] = float(F.mse_loss(latents["student"], latents["teacher"]).item())
            metrics["feature_cosine"] = float(
                F.cosine_similarity(latents["student"], latents["teacher"], dim=-1).item()
            )
            plan_kl = 0.0
            for student_part, teacher_part in zip(logits["student"], logits["teacher"]):
                plan_kl += float(
                    F.kl_div(
                        F.log_softmax(student_part, dim=-1),
                        F.log_softmax(teacher_part, dim=-1),
                        log_target=True,
                        reduction="batchmean",
                    ).item()
                )
            metrics["plan_kl"] = plan_kl
            metrics["action_agreement"] = float(actions["student"] == actions["teacher"])
        metrics["inference_ms"] = 1000.0 * (time.perf_counter() - started)
        return actions[self.control], metrics


class MetricAccumulator:
    def __init__(self):
        self.values: dict[str, list[float]] = {}

    def add(self, **values: float) -> None:
        for key, value in values.items():
            if math.isfinite(float(value)):
                self.values.setdefault(key, []).append(float(value))

    def means(self) -> dict[str, float]:
        return {f"mean_{key}": float(np.mean(values)) for key, values in self.values.items() if values}


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


class JsonlWriter:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.handle = path.open("a", encoding="utf-8")

    def write(self, item: dict[str, Any]) -> None:
        self.handle.write(json.dumps(_json_value(item), sort_keys=True) + "\n")
        self.handle.flush()

    def close(self) -> None:
        self.handle.close()


class CameraPreviewClosed(Exception):
    """Raised when the operator closes the optional ego-camera preview."""


@dataclass(frozen=True)
class CameraProjection:
    """One world point projected into a CARLA RGB camera."""

    u: float
    v: float
    in_front: bool


def project_world_point_to_camera(
    point: np.ndarray,
    world_to_camera: np.ndarray,
    intrinsics: tuple[float, float, float, float],
) -> CameraProjection:
    """Project CARLA world xyz into pixel coordinates without touching an image.

    CARLA camera coordinates are x-forward/y-right/z-up; image coordinates are
    u-right/v-down.  A point behind the camera gets a finite directional proxy
    so the preview can still place an arrow on the appropriate image edge.
    """

    point = np.asarray(point, dtype=np.float64)
    matrix = np.asarray(world_to_camera, dtype=np.float64)
    if point.shape != (3,) or matrix.shape != (4, 4):
        raise ValueError(
            f"expected point [3] and world_to_camera [4,4], got {point.shape} and {matrix.shape}"
        )
    fx, fy, cx, cy = map(float, intrinsics)
    camera = matrix @ np.append(point, 1.0)
    depth, right, up = map(float, camera[:3])
    if depth > 1e-4:
        return CameraProjection(
            u=cx + fx * right / depth,
            v=cy - fy * up / depth,
            in_front=True,
        )

    # A pinhole projection is undefined behind the image plane.  Preserve the
    # horizontal/vertical direction instead so the UI can draw an edge arrow.
    horizontal = 1.0 if right >= 0.0 else -1.0
    denominator = max(abs(depth), abs(right), 1e-4)
    return CameraProjection(
        u=cx + horizontal * 2.0 * REAL_WIDTH,
        v=cy - fy * float(np.clip(up / denominator, -2.0, 2.0)),
        in_front=False,
    )


class CameraPreview:
    """Display CARLA RGB above the corresponding Puffer teacher rasters."""

    DISPLAY_ORDER = (1, 0, 2)
    DISPLAY_NAMES = ("front_left", "front", "front_right")
    HEADER_HEIGHT = 30
    ROW_STRIDE = HEADER_HEIGHT + REAL_HEIGHT

    def __init__(self) -> None:
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        try:
            import pygame
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "--camera-preview requires pygame; install it with "
                "`.venv/bin/python -m pip install 'pygame>=2.6,<3'`"
            ) from error

        self.pygame = pygame
        try:
            pygame.display.init()
            pygame.font.init()
            self.screen = pygame.display.set_mode(
                (3 * REAL_WIDTH, 2 * self.ROW_STRIDE)
            )
        except pygame.error as error:
            pygame.quit()
            raise RuntimeError(
                "--camera-preview could not open a display; make sure DISPLAY is available "
                "or run without --camera-preview"
            ) from error
        pygame.display.set_caption(
            "PufferDrive cameras: CARLA RGB (top) / Puffer raster (bottom)"
        )
        self.font = pygame.font.Font(None, 24)
        self.closed = False

    def _draw_goal_indicator(
        self,
        projection: CameraProjection,
        panel_x: int,
        panel_y: int,
        color: tuple[int, int, int],
    ) -> None:
        pygame = self.pygame
        inside = (
            projection.in_front
            and 0.0 <= projection.u < REAL_WIDTH
            and 0.0 <= projection.v < REAL_HEIGHT
        )
        if inside:
            x = panel_x + round(projection.u)
            y = panel_y + round(projection.v)
            pygame.draw.circle(self.screen, color, (x, y), 11, width=3)
            pygame.draw.line(self.screen, color, (x - 16, y), (x + 16, y), width=2)
            pygame.draw.line(self.screen, color, (x, y - 16), (x, y + 16), width=2)
            return

        center = np.asarray([REAL_WIDTH / 2.0, REAL_HEIGHT / 2.0], dtype=np.float64)
        direction = np.asarray([projection.u, projection.v], dtype=np.float64) - center
        if np.linalg.norm(direction) < 1e-6:
            direction[0] = 1.0
        margin = 18.0
        half_extent = np.asarray(
            [REAL_WIDTH / 2.0 - margin, REAL_HEIGHT / 2.0 - margin], dtype=np.float64
        )
        scale = np.min(half_extent / np.maximum(np.abs(direction), 1e-6))
        tip_local = center + direction * scale
        unit = direction / np.linalg.norm(direction)
        perpendicular = np.asarray([-unit[1], unit[0]])
        tip = np.asarray(
            [panel_x + tip_local[0], panel_y + tip_local[1]], dtype=np.float64
        )
        base = tip - 16.0 * unit
        triangle = [tip, base + 7.0 * perpendicular, base - 7.0 * perpendicular]
        pygame.draw.polygon(
            self.screen,
            color,
            [(round(point[0]), round(point[1])) for point in triangle],
        )

    def update(
        self,
        carla_images: np.ndarray,
        puffer_images: np.ndarray,
        goal_projections: tuple[CameraProjection, CameraProjection, CameraProjection] | None = None,
        *,
        goal_distance_m: float | None = None,
        final_goal: bool = False,
    ) -> bool:
        carla_expected = (3, REAL_HEIGHT, REAL_WIDTH, 3)
        if carla_images.shape != carla_expected or carla_images.dtype != np.uint8:
            raise ValueError(
                f"camera preview expects uint8 CARLA images shaped {carla_expected}, got "
                f"{carla_images.dtype} {carla_images.shape}"
            )
        puffer_expected = (3, SIM_HEIGHT, SIM_WIDTH, 3)
        if puffer_images.shape != puffer_expected or puffer_images.dtype != np.uint8:
            raise ValueError(
                f"camera preview expects uint8 Puffer images shaped {puffer_expected}, got "
                f"{puffer_images.dtype} {puffer_images.shape}"
            )
        if goal_projections is not None and len(goal_projections) != 3:
            raise ValueError(f"expected three goal projections, got {len(goal_projections)}")
        if self.closed:
            return False

        pygame = self.pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (
                event.type == pygame.KEYDOWN and event.key in (pygame.K_ESCAPE, pygame.K_q)
            ):
                self.close()
                return False

        self.screen.fill((18, 18, 18))
        rows = (("CARLA", carla_images), ("PUFFER", puffer_images))
        for row, (source, source_images) in enumerate(rows):
            header_y = row * self.ROW_STRIDE
            image_y = header_y + self.HEADER_HEIGHT
            for column, (camera_index, name) in enumerate(
                zip(self.DISPLAY_ORDER, self.DISPLAY_NAMES)
            ):
                frame = source_images[camera_index]
                surface = pygame.surfarray.make_surface(np.swapaxes(frame, 0, 1))
                if surface.get_size() != (REAL_WIDTH, REAL_HEIGHT):
                    # Nearest-neighbour scaling keeps the semantic raster crisp.
                    surface = pygame.transform.scale(surface, (REAL_WIDTH, REAL_HEIGHT))
                x = column * REAL_WIDTH
                self.screen.blit(surface, (x, image_y))
                label_text = f"{source} / {name}"
                if column == 1 and goal_distance_m is not None:
                    kind = "FINAL" if final_goal else "NEXT"
                    label_text = f"{label_text}  |  {kind} GOAL {goal_distance_m:.1f} m"
                label = self.font.render(label_text, True, (240, 240, 240))
                self.screen.blit(
                    label,
                    (
                        x + (REAL_WIDTH - label.get_width()) // 2,
                        header_y + (self.HEADER_HEIGHT - label.get_height()) // 2,
                    ),
                )
                if goal_projections is not None:
                    color = (255, 170, 0) if final_goal else (0, 255, 80)
                    self._draw_goal_indicator(
                        goal_projections[camera_index], x, image_y, color
                    )
        pygame.display.flip()
        return True

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        self.pygame.display.quit()
        self.pygame.font.quit()


class ClosedLoopEvaluator:
    def __init__(self, args):
        self.args = args
        self.camera_preview = CameraPreview() if args.camera_preview else None
        try:
            self.branches = PolicyBranches(args)
            self.collector = Collector(args)
        except BaseException:
            if self.camera_preview is not None:
                self.camera_preview.close()
            raise
        self.route_planner = None
        self.output = args.output
        self.output.mkdir(parents=True, exist_ok=True)
        self.step_writer = JsonlWriter(self.output / "steps.jsonl")
        self.episode_writer = JsonlWriter(self.output / "episodes.jsonl")

    def close(self) -> None:
        self.step_writer.close()
        self.episode_writer.close()
        if self.camera_preview is not None:
            self.camera_preview.close()
        self.collector.restore()

    def _load_town(self, town: str) -> None:
        previous = self.collector.town
        self.collector.load_town(town)
        if town != previous:
            try:
                from agents.navigation.global_route_planner import GlobalRoutePlanner
            except ImportError as error:
                raise RuntimeError(
                    "CARLA navigation agents are unavailable. Set PYTHONPATH to the 0.9.16 "
                    "PythonAPI/carla directory shown in this module's example."
                ) from error
            self.route_planner = GlobalRoutePlanner(
                self.collector.world.get_map(), self.args.route_resolution
            )

    def _choose_route(self, rng: random.Random) -> tuple[Any, RoutePlan, Any]:
        points = list(self.collector.world.get_map().get_spawn_points())
        if len(points) < 2:
            raise RuntimeError(f"{self.collector.town} exposes fewer than two spawn points")
        for _ in range(self.args.route_attempts):
            start, destination = rng.sample(points, 2)
            try:
                trace = self.route_planner.trace_route(start.location, destination.location)
                plan = RoutePlan.from_trace(trace, self.args.max_route_goals)
            except (KeyError, RuntimeError, ValueError, IndexError):
                continue
            if self.args.min_route_distance <= plan.length <= self.args.max_route_distance:
                return start, plan, destination
        raise RuntimeError(
            f"could not find a route in [{self.args.min_route_distance}, "
            f"{self.args.max_route_distance}] m after {self.args.route_attempts} attempts"
        )

    def _spawn_ego_and_route(self, rng: random.Random):
        blueprint = self.collector.world.get_blueprint_library().find(self.args.ego_blueprint)
        blueprint.set_attribute("role_name", "hero")
        for _ in range(self.args.spawn_attempts):
            start, plan, destination = self._choose_route(rng)
            ego = self.collector.world.try_spawn_actor(blueprint, start)
            if ego is not None:
                return ego, start, plan, destination
        raise RuntimeError("could not spawn the ego at any sampled route start")

    def _attach_event_sensors(self, ego, events: EpisodeEvents) -> list[Any]:
        carla = self.collector.carla
        world = self.collector.world
        sensors = []
        collision = world.spawn_actor(
            world.get_blueprint_library().find("sensor.other.collision"),
            carla.Transform(),
            attach_to=ego,
        )
        collision.listen(events.on_collision)
        sensors.append(collision)
        lane = world.spawn_actor(
            world.get_blueprint_library().find("sensor.other.lane_invasion"),
            carla.Transform(),
            attach_to=ego,
        )
        lane.listen(events.on_lane_invasion)
        sensors.append(lane)
        return sensors

    def _episode_conditioning(self, episode_id: str) -> np.ndarray:
        if self.args.conditioning == "nominal":
            return nominal_conditioning()
        elif self.args.conditioning == "good":
            return good_conditioning()
        return segment_conditioning(episode_id, self.args.conditioning_seed)

    def _weather(self, rng: random.Random) -> dict[str, Any]:
        if self.args.weather == "random":
            weather = Weather.sample(rng)
            self.collector.world.set_weather(weather.to_carla(self.collector.carla))
            return asdict(weather)
        self.collector.world.set_weather(self.collector.carla.WeatherParameters.ClearNoon)
        return {"preset": "ClearNoon"}

    def _seed_simulation(self, rng: random.Random) -> None:
        """Seed TM and walkers after a world load, tolerating its startup race.

        CARLA's Traffic Manager uses a separate RPC endpoint with a hard-coded
        two-second timeout. Immediately after ``load_world`` that endpoint can
        accept ``set_synchronous_mode`` and still time out on the first seed
        call while map registration finishes. Re-acquiring the same port and
        retrying is safe because no episode actors exist yet.
        """

        tm_seed = rng.randrange(1 << 30)
        pedestrian_seed = rng.randrange(1 << 30)
        for attempt in range(3):
            try:
                self.collector.traffic_manager.set_random_device_seed(tm_seed)
                self.collector.world.set_pedestrians_seed(pedestrian_seed)
                return
            except RuntimeError:
                if attempt == 2:
                    raise
                time.sleep(0.5 * (attempt + 1))
                self.collector.traffic_manager = self.collector.client.get_trafficmanager(
                    self.args.tm_port
                )
                self.collector.traffic_manager.set_synchronous_mode(True)
                self.collector.traffic_manager.set_hybrid_physics_mode(False)

    def _update_spectator(self, ego_transform) -> None:
        """Put CARLA's spectator in a stable chase view behind the ego."""

        if not self.args.spectator:
            return
        carla = self.collector.carla
        forward = ego_transform.get_forward_vector()
        location = carla.Location(
            x=ego_transform.location.x - 8.0 * forward.x,
            y=ego_transform.location.y - 8.0 * forward.y,
            z=ego_transform.location.z + 4.0,
        )
        rotation = carla.Rotation(
            pitch=-15.0,
            yaw=ego_transform.rotation.yaw,
            roll=0.0,
        )
        self.collector.world.get_spectator().set_transform(carla.Transform(location, rotation))

    def _draw_active_goal(self, goal: np.ndarray, *, final: bool) -> None:
        """Show a spectator HUD label without adding geometry to RGB sensor views."""

        if not self.args.spectator:
            return
        carla = self.collector.carla
        color = carla.Color(255, 170, 0) if final else carla.Color(0, 255, 80)
        top = carla.Location(x=float(goal[0]), y=float(goal[1]), z=float(goal[2]) + 3.5)
        life_time = max(0.05, 1.25 * self.args.dt)
        self.collector.world.debug.draw_string(
            top,
            "FINAL GOAL" if final else "NEXT GOAL",
            draw_shadow=True,
            color=color,
            life_time=life_time,
            persistent_lines=False,
        )

    def run_episode(self, town: str, episode_index: int) -> dict[str, Any]:
        episode_id = f"{town}_{episode_index:04d}"
        seed = zlib.crc32(f"closed-loop:{self.args.seed}:{episode_id}".encode()) & 0xFFFFFFFF
        rng = random.Random(seed)
        self._load_town(town)
        world = self.collector.world
        carla = self.collector.carla
        spawned: list[int] = []
        controllers: list[int] = []
        camera_sensors: list[Any] = []
        camera_queues: list[queue.Queue] = []
        event_sensors: list[Any] = []
        weather = self._weather(rng)
        self._seed_simulation(rng)

        try:
            ego, start_transform, plan, destination = self._spawn_ego_and_route(rng)
            spawned.append(ego.id)
            spawned.extend(self.collector._spawn_vehicles(rng, start_transform))
            walkers, controllers = self.collector._spawn_walkers(rng)
            spawned.extend(walkers)
            events = EpisodeEvents()
            camera_sensors, camera_queues = self.collector._attach_cameras(ego)
            event_sensors = self._attach_event_sensors(ego, events)
            self.collector._listen(camera_sensors, camera_queues)

            # Let traffic spread and Unreal build temporal history while keeping
            # the manually controlled ego fixed at its route start.
            ego.apply_control(carla.VehicleControl(hand_brake=True))
            self._update_spectator(ego.get_transform())
            for _ in range(self.args.warmup):
                frame_id = world.tick()
                for sensor_queue in camera_queues:
                    self.collector._await_frame(sensor_queue, frame_id)
            ego.apply_control(carla.VehicleControl(brake=1.0))

            actors = self.collector._static_geometry(ego)
            conditioning = self._episode_conditioning(episode_id)
            raw_conditioning = conditioning_to_raw(conditioning)
            controller = JerkController(
                self.args.dt,
                conditioning,
                accel_feedback=self.args.accel_feedback,
                speed_feedback=self.args.speed_feedback,
            )
            geometry = _vehicle_geometry(ego)
            tracker = RouteTracker(plan)
            self.branches.reset()
            accumulator = MetricAccumulator()
            best_progress = 0.0
            last_progress_step = 0
            reached_intermediate = 0
            reason = "timeout"
            success = False
            episode_started = time.perf_counter()
            steps_run = 0

            for step in range(self.args.max_steps):
                steps_run = step + 1
                frame_id = world.tick()
                raw_images = [
                    self.collector._await_frame(sensor_queue, frame_id)
                    for sensor_queue in camera_queues
                ]
                images = np.stack([_decode(image) for image in raw_images])
                snapshot = world.get_snapshot()
                transform = ego.get_transform()
                self._update_spectator(transform)
                carla_xyz = np.asarray(
                    [transform.location.x, transform.location.y, transform.location.z], dtype=np.float64
                )
                reached_intermediate += tracker.update(carla_xyz, self.args.intermediate_goal_radius)
                goal = tracker.current_goal
                distance_to_goal = tracker.distance_to_goal(carla_xyz)
                agents = roads = None
                if self.branches.need_teacher or self.camera_preview is not None:
                    ego_transform = snapshot.find(ego.id).get_transform()
                    center = _box_center(
                        ego_transform,
                        (
                            ego.bounding_box.location.x,
                            ego.bounding_box.location.y,
                            ego.bounding_box.location.z,
                        ),
                    )
                    yaw = -math.radians(ego_transform.rotation.yaw)
                    agents = self.collector._agents(snapshot, actors, center, yaw)
                    roads = self.collector.road_index.crop(
                        np.asarray(center), yaw, self.args.radius
                    )

                rendered_teacher_images = None
                if self.camera_preview is not None:
                    if agents is None or roads is None or self.branches.renderer is None:
                        raise AssertionError("camera preview has no live Puffer scene")
                    rendered_teacher_images = self.branches.renderer.render(agents, roads)
                    puffer_images = puffer_preview_images(rendered_teacher_images)
                    projection_point = np.asarray(goal, dtype=np.float64).copy()
                    projection_point[2] += 0.5
                    projections = tuple(
                        project_world_point_to_camera(
                            projection_point,
                            np.asarray(sensor.get_transform().get_inverse_matrix()),
                            sensor_intrinsics(camera),
                        )
                        for sensor, camera in zip(camera_sensors, rig_cameras())
                    )
                    if not self.camera_preview.update(
                        images,
                        puffer_images,
                        projections,
                        goal_distance_m=distance_to_goal,
                        final_goal=tracker.at_final_goal,
                    ):
                        ego.apply_control(carla.VehicleControl(brake=1.0, hand_brake=True))
                        raise CameraPreviewClosed
                self._draw_active_goal(goal, final=tracker.at_final_goal)
                telemetry = read_ego_telemetry(
                    ego,
                    goal[:2],
                    conditioning,
                    collision=events.collisions > 0,
                    respawn=step == 0,
                    carla=carla,
                    geometry=geometry,
                )

                action, policy_metrics = self.branches.step(
                    images,
                    telemetry.observation,
                    agents=agents,
                    roads=roads,
                    rendered_teacher_images=rendered_teacher_images,
                )
                command = controller.step(
                    action,
                    signed_speed=telemetry.signed_speed,
                    measured_accel_long=telemetry.accel_long,
                    wheelbase=telemetry.wheelbase,
                    max_wheel_steer=telemetry.max_wheel_steer,
                )
                final_spatial = tracker.at_final_goal and distance_to_goal <= float(raw_conditioning[0])
                final_speed = abs(telemetry.signed_speed) <= float(raw_conditioning[1])

                if tracker.progress > best_progress + self.args.progress_epsilon:
                    best_progress = tracker.progress
                    last_progress_step = step

                if events.collisions and self.args.terminate_on_collision:
                    reason = "collision"
                elif final_spatial and final_speed:
                    reason = "goal_reached"
                    success = True
                elif tracker.deviation > self.args.max_route_deviation:
                    reason = "route_deviation"
                elif (
                    step >= self.args.stuck_grace_steps
                    and step - last_progress_step >= self.args.stuck_steps
                    and abs(telemetry.signed_speed) < self.args.stuck_speed
                ):
                    reason = "stuck"
                else:
                    reason = "running"

                step_record: dict[str, Any] = {
                    "episode_id": episode_id,
                    "town": town,
                    "step": step,
                    "carla_frame": frame_id,
                    "control": self.args.control,
                    "action": action,
                    "speed_mps": telemetry.signed_speed,
                    "accel_long_mps2": telemetry.accel_long,
                    "accel_lat_mps2": telemetry.accel_lat,
                    "physical_steer_rad": telemetry.steering_angle,
                    "throttle": command.throttle,
                    "brake": command.brake,
                    "steer": command.steer,
                    "reverse": command.reverse,
                    "target_speed_mps": command.target_speed,
                    "target_accel_long_mps2": command.target_accel_long,
                    "target_accel_lat_mps2": command.target_accel_lat,
                    "jerk_long_mps3": command.jerk_long,
                    "jerk_lat_mps3": command.jerk_lat,
                    "route_progress_m": tracker.progress,
                    "route_completion": tracker.completion,
                    "route_deviation_m": tracker.deviation,
                    "goal_index": tracker.goal_cursor,
                    "num_goals": len(plan.goal_indices),
                    "distance_to_goal_m": distance_to_goal,
                    "collision_count": events.collisions,
                    "lane_invasions": events.lane_invasions,
                    "termination": reason,
                    **policy_metrics,
                }
                self.step_writer.write(step_record)
                accumulator.add(
                    speed_mps=abs(telemetry.signed_speed),
                    inference_ms=float(policy_metrics["inference_ms"]),
                    **{
                        key: float(policy_metrics[key])
                        for key in ("feature_mse", "feature_cosine", "plan_kl", "action_agreement")
                        if key in policy_metrics
                    },
                )

                if reason != "running":
                    ego.apply_control(carla.VehicleControl(brake=1.0, hand_brake=True))
                    break
                ego.apply_control(
                    carla.VehicleControl(
                        throttle=command.throttle,
                        brake=command.brake,
                        steer=command.steer,
                        reverse=command.reverse,
                        manual_gear_shift=False,
                    )
                )

            if reason == "running":
                reason = "timeout"
            elapsed = time.perf_counter() - episode_started
            summary = {
                "episode_id": episode_id,
                "town": town,
                "seed": seed,
                "control": self.args.control,
                "shadow": self.args.shadow,
                "success": success,
                "termination": reason,
                "steps": steps_run,
                "simulated_seconds": steps_run * self.args.dt,
                "wall_seconds": elapsed,
                "route_length_m": plan.length,
                "route_progress_m": tracker.progress,
                "route_completion": tracker.completion,
                "route_deviation_m": tracker.deviation,
                "num_goals": len(plan.goal_indices),
                "intermediate_goals_reached": reached_intermediate,
                "collision_count": events.collisions,
                "collision_impulse": events.collision_impulse,
                "lane_invasions": events.lane_invasions,
                "start": [
                    start_transform.location.x,
                    start_transform.location.y,
                    start_transform.location.z,
                ],
                "destination": [
                    destination.location.x,
                    destination.location.y,
                    destination.location.z,
                ],
                "conditioning": conditioning,
                "conditioning_raw": raw_conditioning,
                "weather": weather,
                **accumulator.means(),
            }
            self.episode_writer.write(summary)
            print(
                f"[{episode_id}] {reason}: completion={tracker.completion:.1%}, "
                f"collisions={events.collisions}, lane_invasions={events.lane_invasions}, "
                f"steps={steps_run}",
                flush=True,
            )
            return summary
        finally:
            self.collector._mute(camera_sensors, camera_queues)
            for sensor in event_sensors:
                try:
                    sensor.stop()
                except RuntimeError:
                    pass
            for sensor in (*camera_sensors, *event_sensors):
                try:
                    sensor.destroy()
                except RuntimeError:
                    pass
            if controllers:
                for controller_actor in world.get_actors(controllers):
                    try:
                        controller_actor.stop()
                    except RuntimeError:
                        pass
                self.collector.client.apply_batch(
                    [carla.command.DestroyActor(actor_id) for actor_id in controllers]
                )
            if spawned:
                self.collector.client.apply_batch(
                    [carla.command.DestroyActor(actor_id) for actor_id in spawned]
                )
            try:
                world.tick()
            except RuntimeError:
                pass


def _aggregate(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if not summaries:
        return {"episodes": 0}
    numeric = (
        "success",
        "route_completion",
        "collision_count",
        "lane_invasions",
        "mean_speed_mps",
        "mean_feature_mse",
        "mean_feature_cosine",
        "mean_plan_kl",
        "mean_action_agreement",
        "mean_inference_ms",
    )
    result: dict[str, Any] = {"episodes": len(summaries)}
    for key in numeric:
        values = [float(item[key]) for item in summaries if key in item]
        if values:
            output_key = "success_rate" if key == "success" else key
            if key != "success" and not output_key.startswith("mean_"):
                output_key = f"mean_{output_key}"
            result[output_key] = float(np.mean(values))
    result["terminations"] = {
        reason: sum(item["termination"] == reason for item in summaries)
        for reason in sorted({item["termination"] for item in summaries})
    }
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--student", type=Path, help="self-contained deployment.pt from distillation")
    parser.add_argument("--checkpoint", type=Path, required=True, help="frozen recurrent giga checkpoint")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control", choices=("student", "teacher"), default="student")
    parser.add_argument(
        "--shadow",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="run the non-controlling perception/LSTM branch and log disagreement",
    )
    parser.add_argument("--town", action="append", dest="towns", default=None)
    parser.add_argument("--episodes", type=int, default=10, help="episodes per town")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--conditioning", choices=("nominal", "sampled", "good"), default="good")
    parser.add_argument("--conditioning-seed", type=int, default=42)
    parser.add_argument("--weather", choices=("clear", "random"), default="clear")
    parser.add_argument(
        "--spectator",
        action="store_true",
        help="keep CARLA's third-person spectator 8 m behind and 4 m above the ego",
    )
    parser.add_argument(
        "--camera-preview",
        action="store_true",
        help=(
            "show a 2x3 view: CARLA RGB on top and the matching Puffer teacher raster below, "
            "with display-only projected goal markers"
        ),
    )
    parser.add_argument("--vehicles", type=int, default=40)
    parser.add_argument("--walkers", type=int, default=20)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=8000)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--dt", type=float, default=DEFAULT_DT)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--radius", type=float, default=DEFAULT_RADIUS)
    parser.add_argument("--road-resolution", type=float, default=1.0)
    parser.add_argument("--route-resolution", type=float, default=2.0)
    parser.add_argument("--min-route-distance", type=float, default=80.0)
    parser.add_argument("--max-route-distance", type=float, default=200.0)
    parser.add_argument("--max-route-goals", type=int, default=4)
    parser.add_argument("--route-attempts", type=int, default=200)
    parser.add_argument("--spawn-attempts", type=int, default=20)
    parser.add_argument("--intermediate-goal-radius", type=float, default=6.0)
    parser.add_argument("--max-route-deviation", type=float, default=15.0)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--stuck-seconds", type=float, default=15.0)
    parser.add_argument("--stuck-grace-seconds", type=float, default=10.0)
    parser.add_argument("--stuck-speed", type=float, default=0.3)
    parser.add_argument("--progress-epsilon", type=float, default=1.0)
    parser.add_argument(
        "--terminate-on-collision", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--accel-feedback", type=float, default=0.35)
    parser.add_argument("--speed-feedback", type=float, default=0.8)
    parser.add_argument("--ego-blueprint", default="vehicle.lincoln.mkz_2020")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--amp", choices=("bf16", "fp16", "off"), default="bf16")
    parser.add_argument("--resume", action="store_true")
    return parser


def _validate_args(parser: argparse.ArgumentParser, args) -> None:
    if args.episodes < 1 or args.max_steps < 1 or args.max_route_goals < 1:
        parser.error("--episodes, --max-steps and --max-route-goals must be positive")
    if args.dt <= 0 or args.route_resolution <= 0 or args.road_resolution <= 0:
        parser.error("time and map resolutions must be positive")
    if args.min_route_distance <= 0 or args.max_route_distance < args.min_route_distance:
        parser.error("route distance bounds are invalid")
    if args.checkpoint is None or not args.checkpoint.is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint}")
    if (args.control == "student" or args.shadow) and (
        args.student is None or not args.student.is_file()
    ):
        parser.error("student control/shadow requires an existing --student deployment.pt")
    if abs(args.dt - DEFAULT_DT) > 1e-9:
        print(
            f"warning: --dt {args.dt} differs from the checkpoint's 0.1 s dynamics",
            file=sys.stderr,
        )
    args.stuck_steps = max(1, round(args.stuck_seconds / args.dt))
    args.stuck_grace_steps = max(0, round(args.stuck_grace_seconds / args.dt))


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    args.towns = tuple(args.towns or ("Town01",))
    completed: set[str] = set()
    episodes_path = args.output / "episodes.jsonl"
    if args.output.exists() and any(args.output.iterdir()):
        if not args.resume:
            parser.error(f"{args.output} is not empty; pass --resume to append missing episodes")
        if episodes_path.is_file():
            for line in episodes_path.read_text().splitlines():
                if line.strip():
                    completed.add(str(json.loads(line)["episode_id"]))

    args.output.mkdir(parents=True, exist_ok=True)
    config = {
        **{key: _json_value(value) for key, value in vars(args).items()},
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "student_sha256": sha256_file(args.student) if args.student is not None else None,
        "camera_order": [camera.name for camera in rig_cameras()],
        "camera_shape": [3, REAL_HEIGHT, REAL_WIDTH, 3],
        "puffer_camera_shape": [3, SIM_HEIGHT, SIM_WIDTH, 3],
        "camera_preview_layout": ["carla", "puffer"] if args.camera_preview else None,
    }
    config_path = args.output / "config.json"
    if not config_path.exists():
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

    evaluator = ClosedLoopEvaluator(args)
    summaries: list[dict[str, Any]] = []
    preview_closed = False
    try:
        for town in args.towns:
            for episode_index in range(args.episodes):
                episode_id = f"{town}_{episode_index:04d}"
                if episode_id in completed:
                    print(f"[{episode_id}] already complete; skipping", flush=True)
                    continue
                try:
                    summaries.append(evaluator.run_episode(town, episode_index))
                except CameraPreviewClosed:
                    print("camera preview closed; stopping evaluation", flush=True)
                    preview_closed = True
                    break
            if preview_closed:
                break
    finally:
        evaluator.close()

    all_summaries = []
    if episodes_path.is_file():
        all_summaries = [json.loads(line) for line in episodes_path.read_text().splitlines() if line.strip()]
    aggregate = _aggregate(all_summaries)
    (args.output / "summary.json").write_text(json.dumps(aggregate, indent=2, sort_keys=True) + "\n")
    print(json.dumps(aggregate, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
