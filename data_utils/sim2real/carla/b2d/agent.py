"""The giga policy as a Bench2Drive (CARLA leaderboard 2.0) agent.

This is ``demo/closed_loop.py``'s rollout, restated in the shape the leaderboard
demands.  The differences are all in who owns the loop, and each one is a place
the policy would silently misbehave if it were ignored:

* **The leaderboard ticks at 20 Hz; the policy is a 10 Hz policy.**
  ``leaderboard_evaluator.frame_rate = 20.0``, while the checkpoint was trained
  at ``dt = 0.1`` and :class:`JerkController` integrates jerk over that period.
  Running the policy every tick would halve every commanded acceleration ramp
  and double the effective steering rate, so the policy runs on every second
  tick and the control is held in between -- which is also what a real 10 Hz
  planner feeding a faster actuator does.
* **The LSTM state has to survive between calls.**  ``run_step`` returns one
  control per tick and owns no rollout, so the recurrent state lives here and
  is zeroed once per route, exactly like ``RecurrentPlanningRuntime.reset``.
* **Perception runs from TorchScript, not from this repository.**  This no
  longer has to be true -- the leaderboard now runs in the repository's own
  Python 3.10.16 venv, where ``pufferlib`` imports fine -- but it stays true on
  purpose: the scores name a frozen policy.  ``b2d/export.py`` writes the two
  traced graphs and the rig/conditioning manifest this agent loads, and nothing
  here imports the simulator-side stack.

Configuration comes through ``TEAM_CONFIG`` (the bundle directory written by
``b2d/export.py``) plus two optional environment variables:

    SAVE_PATH           leaderboard-standard dump root; also enables
                        ``rgb_front/`` and ``meta/`` frame dumps
    PICDRIVE_VIZ        "1" to write per-policy-tick frames for video rendering
    PICDRIVE_SPECTATOR  "chase" to drive CARLA's spectator from here; anything
                        else leaves the leaderboard's top-down camera in charge
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import carla
import numpy as np
import torch
from leaderboard.autoagents.autonomous_agent import AutonomousAgent, Track

from data_utils.sim2real.carla.control import (
    JerkController,
    RoutePlan,
    RouteTracker,
    _vehicle_geometry,
    carla_mount,
    read_ego_telemetry,
    spectator_transform,
)


# Goals every ~50 m, the spacing RoutePlan.from_trace targets. Bench2Drive
# routes vary in length, so the count is left free and only the spacing is
# pinned; capping the count instead would stretch the goal spacing on the long
# routes and hand the policy a conditioning it never saw.
MAX_ROUTE_GOALS = 64
INTERMEDIATE_GOAL_RADIUS = 6.0


def get_entry_point() -> str:
    return "PicDriveAgent"


class PicDriveAgent(AutonomousAgent):
    """Bench2Drive SENSORS-track agent driving the distilled camera policy."""

    # set_global_plan runs before setup(), so the route lands here first and
    # setup() must not clear it.
    tracker: RouteTracker | None = None

    def setup(self, path_to_conf_file: str) -> None:
        self.track = Track.SENSORS

        # The evaluator appends '+<route name>' to TEAM_CONFIG before every
        # setup() and never resets it, so after three routes the string is
        # "bundle+route1+route2+route3". The bundle is the first field and the
        # route this call is for is the last.
        fields = str(path_to_conf_file).split("+")
        self.route_name = fields[-1] if len(fields) > 1 else "route"
        bundle_dir = Path(fields[0]).expanduser()
        if bundle_dir.is_file():
            bundle_dir = bundle_dir.parent
        self.bundle = json.loads((bundle_dir / "bundle.json").read_text())
        if int(self.bundle["schema_version"]) != 1:
            raise ValueError(f"{bundle_dir}/bundle.json is not a supported policy bundle")

        if not torch.cuda.is_available():
            raise RuntimeError("the leaderboard agent needs CUDA; a CPU score is not comparable")
        self.device = torch.device("cuda")
        self.encoder = torch.jit.load(str(bundle_dir / "encoder.ts"), map_location=self.device).eval()
        self.planner = torch.jit.load(str(bundle_dir / "planner.ts"), map_location=self.device).eval()

        self.dt = float(self.bundle["dt"])
        # Read the simulator's own step rather than assuming the leaderboard's
        # 20 Hz: this is the number the policy is actually being decimated
        # against, and a fork that changes frame_rate must not silently halve
        # every acceleration ramp.
        tick_period = self.hero_actor.get_world().get_settings().fixed_delta_seconds
        if not tick_period or tick_period <= 0.0:
            raise ValueError("the world is not in fixed-delta mode; the policy has no tick period")
        tick_period = float(tick_period)
        stride = self.dt / tick_period
        if abs(stride - round(stride)) > 1e-6 or round(stride) < 1:
            raise ValueError(
                f"policy dt {self.dt}s is not a whole multiple of the simulator's {tick_period}s tick"
            )
        self.decimation = round(stride)
        self.conditioning = np.asarray(self.bundle["conditioning_vector"], dtype=np.float32)
        self.hidden_size = int(self.bundle["hidden_size"])
        self.image_shape = (
            len(self.bundle["rig"]),
            self.bundle["image_height"],
            self.bundle["image_width"],
        )

        self.controller = JerkController(self.dt, self.conditioning)
        self.hidden = torch.zeros((1, self.hidden_size), device=self.device)
        self.cell = torch.zeros((1, self.hidden_size), device=self.device)
        self.tick = 0
        self.policy_steps = 0
        self.geometry: tuple[float, float] | None = None
        self.last_control = carla.VehicleControl(throttle=0.0, brake=1.0, steer=0.0)
        self.collisions = 0
        self._attach_collision_sensor()
        self._setup_viz()
        # "chase" is the only mode this agent drives. Under "bev" (and when
        # unset) the leaderboard's own top-down camera in scenario_manager runs
        # instead -- they must not both be on, or the window alternates between
        # the two every tick. Only meaningful against a server with a window;
        # the default -RenderOffScreen one has nobody watching.
        self.spectator = (
            self.hero_actor.get_world().get_spectator()
            if os.environ.get("PICDRIVE_SPECTATOR", "") == "chase"
            else None
        )

    def _attach_collision_sensor(self) -> None:
        """Feed ``obs[5]`` the same collision flag the offline rollout had.

        The leaderboard's own collision criterion is not visible to the agent
        and no collision sensor is in ``ALLOWED_SENSORS``, so this one is
        attached directly to the hero and torn down in :meth:`destroy`.  It is
        read-only telemetry: it feeds the ego vector and nothing else.
        """

        world = self.hero_actor.get_world()
        blueprint = world.get_blueprint_library().find("sensor.other.collision")
        self.collision_sensor = world.spawn_actor(
            blueprint, carla.Transform(), attach_to=self.hero_actor
        )
        self.collision_sensor.listen(self._on_collision)

    def _on_collision(self, event) -> None:
        self.collisions += 1

    def _setup_viz(self) -> None:
        self.viz_dir: Path | None = None
        save_path = os.environ.get("SAVE_PATH")
        if not save_path or os.environ.get("PICDRIVE_VIZ", "") not in ("1", "true", "True"):
            return
        # save_name already carries route id, town, scenario and a timestamp,
        # so it is unique across repetitions without a counter of our own.
        self.viz_dir = Path(save_path) / self.route_name
        (self.viz_dir / "rgb_front").mkdir(parents=True, exist_ok=True)
        (self.viz_dir / "rgb_strip").mkdir(parents=True, exist_ok=True)
        (self.viz_dir / "meta").mkdir(parents=True, exist_ok=True)

    def sensors(self) -> list[dict]:
        """The exported rig, mounted the way ``collect._attach_cameras`` mounts it."""

        box = self.hero_actor.bounding_box.location
        box_offset = (box.x, box.y, box.z)
        sensors = []
        for camera in self.bundle["rig"]:
            (x, y, z), (pitch, yaw, roll) = carla_mount(
                camera["pos"],
                camera["pitch_deg"],
                camera["yaw_deg"],
                camera["roll_deg"],
                box_offset,
            )
            sensors.append(
                {
                    "type": "sensor.camera.rgb",
                    "id": camera["id"],
                    "x": x,
                    "y": y,
                    "z": z,
                    "pitch": pitch,
                    "yaw": yaw,
                    "roll": roll,
                    "width": camera["width"],
                    "height": camera["height"],
                    "fov": camera["fov_deg"],
                }
            )
        return sensors

    def set_global_plan(self, global_plan_gps, global_plan_world_coord) -> None:
        """Keep the dense route as well as the base class's 50 m downsample.

        :class:`RoutePlan` measures arc length off the route samples and places
        its goals along it, so it wants every sample; the downsampled plan the
        base class keeps would put the first goal wherever the downsample landed.
        """

        super().set_global_plan(global_plan_gps, global_plan_world_coord)
        trace = [
            (SimpleNamespace(transform=transform), option)
            for transform, option in global_plan_world_coord
        ]
        self.tracker = RouteTracker(RoutePlan.from_trace(trace, MAX_ROUTE_GOALS))

    def _images(self, input_data: dict) -> np.ndarray:
        """Stack the rig into the ``[3, H, W, 3]`` RGB uint8 the student wants."""

        frames = []
        for camera in self.bundle["rig"]:
            _, array = input_data[camera["id"]]
            # CARLA hands the leaderboard a BGRA buffer; collect._decode takes
            # the same 2::-1 slice, and the student was distilled on that order.
            frames.append(np.ascontiguousarray(array[:, :, 2::-1]))
        images = np.stack(frames)
        if images.shape != (*self.image_shape, 3):
            raise ValueError(f"expected {(*self.image_shape, 3)} camera stack, got {images.shape}")
        return images

    @torch.inference_mode()
    def _policy_action(self, images: np.ndarray, ego: np.ndarray) -> int:
        tensor = torch.from_numpy(images).permute(0, 3, 1, 2).unsqueeze(0).to(self.device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            scene = self.encoder(tensor)
        ego_tensor = torch.from_numpy(ego).unsqueeze(0).to(self.device)
        logits, self.hidden, self.cell = self.planner(
            scene.float(), ego_tensor, self.hidden, self.cell
        )
        return int(logits.argmax(dim=-1).item())

    def run_step(self, input_data: dict, timestamp: float) -> carla.VehicleControl:
        tick = self.tick
        self.tick += 1
        if self.spectator is not None:
            # Every tick, not every policy tick: at 10 Hz the chase camera in
            # the window visibly stutters, and this is only an RPC.
            self.spectator.set_transform(
                spectator_transform(self.hero_actor.get_transform(), carla)
            )
        if tick % self.decimation:
            # Between policy ticks the actuator holds the last command, which is
            # what a 10 Hz planner on a 20 Hz vehicle actually does.
            return self.last_control

        hero = self.hero_actor
        if self.geometry is None:
            self.geometry = _vehicle_geometry(hero)
        transform = hero.get_transform()
        carla_xyz = np.asarray(
            [transform.location.x, transform.location.y, transform.location.z], dtype=np.float64
        )
        self.tracker.update(carla_xyz, INTERMEDIATE_GOAL_RADIUS)
        goal = self.tracker.current_goal

        telemetry = read_ego_telemetry(
            hero,
            goal[:2],
            self.conditioning,
            collision=self.collisions > 0,
            respawn=self.policy_steps == 0,
            carla=carla,
            geometry=self.geometry,
        )
        action = self._policy_action(self._images(input_data), telemetry.observation)
        command = self.controller.step(
            action,
            signed_speed=telemetry.signed_speed,
            measured_accel_long=telemetry.accel_long,
            wheelbase=telemetry.wheelbase,
            max_wheel_steer=telemetry.max_wheel_steer,
        )
        self.last_control = carla.VehicleControl(
            throttle=command.throttle,
            steer=command.steer,
            brake=command.brake,
            reverse=command.reverse,
        )
        if self.viz_dir is not None:
            self._dump(input_data, telemetry, command, action)
        self.policy_steps += 1
        return self.last_control

    def _dump(self, input_data, telemetry, command, action: int) -> None:
        import cv2

        index = self.policy_steps
        # front_left | front | front_right reads left-to-right like the road.
        strip = np.concatenate(
            [input_data[name][1][:, :, :3] for name in ("front_left", "front", "front_right")],
            axis=1,
        )
        cv2.imwrite(str(self.viz_dir / "rgb_front" / f"{index:04d}.jpg"), input_data["front"][1][:, :, :3])
        cv2.imwrite(str(self.viz_dir / "rgb_strip" / f"{index:04d}.jpg"), strip)
        (self.viz_dir / "meta" / f"{index:04d}.json").write_text(
            json.dumps(
                {
                    "steer": command.steer,
                    "throttle": command.throttle,
                    "brake": command.brake,
                    "reverse": bool(command.reverse),
                    "speed": telemetry.signed_speed,
                    "action": action,
                    "jerk_long": command.jerk_long,
                    "jerk_lat": command.jerk_lat,
                    "target_speed": command.target_speed,
                    "route_completion": self.tracker.completion,
                    "route_deviation": self.tracker.deviation,
                    "goal_index": self.tracker.goal_cursor,
                    "num_goals": len(self.tracker.plan.goal_indices),
                    "collisions": self.collisions,
                    "tick": self.tick,
                }
            )
        )

    def destroy(self) -> None:
        sensor = getattr(self, "collision_sensor", None)
        # The leaderboard tears the world down before calling destroy() when a
        # route ends badly, so the sensor is often already gone; touching a dead
        # actor logs a CARLA-level error that looks like a leak and is not one.
        if sensor is not None:
            try:
                if sensor.is_alive:
                    sensor.stop()
                    sensor.destroy()
            except RuntimeError:
                pass
            self.collision_sensor = None
        self.hidden = None
        self.cell = None
        self.encoder = None
        self.planner = None
        torch.cuda.empty_cache()
