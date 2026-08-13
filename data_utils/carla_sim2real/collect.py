"""Record paired CARLA-image / simulator-scene samples from a running server.

This is the only module that talks to CARLA.  It writes the *same* artifact tree
as ``data_utils.waymo_sim2real.preprocess`` plus ``extract_ego_state``, so the
whole downstream half -- teacher features, audit PNGs, the verifier, the
distillation trainer -- runs against a CARLA dataset unchanged.

One episode is one segment: 91 frames at 10 Hz, matching ``episode_length = 91``
and ``dt = 0.1`` in config/ocean/drive_3cam.ini.  Traffic Manager drives the ego
and all traffic; this process only observes, which makes it the direct analogue
of Waymo log replay.

    python -m data_utils.carla_sim2real.collect \
        --output artifacts/carla_sim2real/training \
        --town Town01 --town Town02 --town Town10HD \
        --episodes 100 --resume

The server must already be up:

    /home/tjhu78u/CARLA_0_9_16/CarlaUE4.sh -RenderOffScreen -carla-rpc-port=2000
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import random
import shutil
import sys
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from data_utils.waymo_sim2real.processed import (
    CAMERA_NAMES,
    EGO_SCHEMA_VERSION,
    MAX_SPEED,
    PROCESSED_SCHEMA_VERSION,
    atomic_savez,
)

from . import EPISODE_FRAMES, FRAME_INTERVAL_MICROS, REAL_HEIGHT, REAL_WIDTH
from .ego import episode_ego_obs
from .rig import mount, rig_array, rig_cameras, sensor_fov_deg, source_calibration
from .roads import ROAD_LANE, RoadIndex, town_roads, world_to_ego

# RenderState type ids (drive.h:150). CARLA has no cyclist actor category: a
# bicycle is a vehicle blueprint with two wheels.
VEHICLE, PEDESTRIAN, CYCLIST = 1, 2, 3

DEFAULT_TOWNS = ("Town01", "Town02", "Town10HD")
# Town03/04/05 have real elevation, and the rasterizer is a flat-ground model:
# roads at z=0 and agent boxes based at z=0. They need a height filter first.
FLAT_TOWNS = frozenset(DEFAULT_TOWNS)

# Matches preprocess.py's default scene cull.
DEFAULT_RADIUS = 220.0
DEFAULT_DT = 0.1


@dataclass(frozen=True)
class Weather:
    """One episode's sky, sampled per episode and recorded in the manifest."""

    cloudiness: float
    precipitation: float
    precipitation_deposits: float
    wind_intensity: float
    sun_azimuth_angle: float
    sun_altitude_angle: float
    fog_density: float
    fog_distance: float
    wetness: float

    @classmethod
    def sample(cls, rng: random.Random) -> "Weather":
        """Draw a sky.

        Skewed toward daylight on purpose: Waymo Perception is overwhelmingly
        daytime, and the point of this data is to widen appearance around that
        distribution rather than to replace it with one the real branch never
        sees.  Dusk gets a small share; true night is excluded, because at
        ``sun_altitude_angle < -10`` the three cameras carry almost no signal
        while the abstract render is unchanged -- an unlearnable pair.
        """
        precipitation = 0.0 if rng.random() < 0.6 else rng.uniform(0.0, 80.0)
        fog = 0.0 if rng.random() < 0.8 else rng.uniform(0.0, 40.0)
        roll = rng.random()
        if roll < 0.80:
            altitude = rng.uniform(20.0, 85.0)
        elif roll < 0.95:
            altitude = rng.uniform(0.0, 20.0)
        else:
            altitude = rng.uniform(-10.0, 0.0)
        return cls(
            cloudiness=rng.uniform(0.0, 90.0),
            precipitation=precipitation,
            precipitation_deposits=rng.uniform(0.0, max(precipitation, 1e-6)),
            wind_intensity=rng.uniform(0.0, 50.0),
            sun_azimuth_angle=rng.uniform(0.0, 360.0),
            sun_altitude_angle=altitude,
            fog_density=fog,
            fog_distance=rng.uniform(10.0, 90.0),
            wetness=rng.uniform(0.0, 60.0),
        )

    def to_carla(self, carla):
        return carla.WeatherParameters(**asdict(self))


def _agent_type(actor) -> int:
    """Map a CARLA actor onto a RenderState type, or 0 for things we do not draw."""
    type_id = actor.type_id
    if type_id.startswith("walker.pedestrian"):
        return PEDESTRIAN
    if not type_id.startswith("vehicle."):
        return 0
    wheels = actor.attributes.get("number_of_wheels")
    return CYCLIST if wheels is not None and int(wheels) == 2 else VEHICLE


def _decode(image) -> np.ndarray:
    """CARLA's BGRA buffer as an RGB ``[H, W, 3]`` uint8 array."""
    raw = np.frombuffer(image.raw_data, dtype=np.uint8)
    return np.ascontiguousarray(raw.reshape(image.height, image.width, 4)[:, :, 2::-1])


class _Actor:
    """An actor's static geometry, read once so the per-frame loop stays cheap."""

    __slots__ = ("id", "type", "length", "width", "height", "offset")

    def __init__(self, actor, render_type: int):
        box = actor.bounding_box
        self.id = actor.id
        self.type = render_type
        self.length = 2.0 * box.extent.x
        self.width = 2.0 * box.extent.y
        self.height = 2.0 * box.extent.z
        self.offset = (box.location.x, box.location.y, box.location.z)


class DegenerateTraffic(RuntimeError):
    """The traffic fleet is not driving, so the episode is not a driving scene.

    Kept as a gate even though the Traffic-Manager handling in ``load_town`` is
    what actually fixes this: the failure it guards against produces 91 frames
    that are correctly paired and structurally perfect, differing from good data
    only in that the cars are parked on pavements at odd angles. Nothing
    downstream can detect that, so it is checked here or not at all.
    """


class OffRoadEgo(RuntimeError):
    """The ego is not on a drivable lane, so the episode is not a driving scene.

    Spawning tens of vehicles at once shoves cars around, and an ego that ends up
    parked on grass still records 91 perfectly well-paired frames -- the abstract
    render correctly shows black ground with the road in the distance. Nothing is
    *wrong* with the sample; it simply is not the distribution the policy needs,
    so it is rejected rather than collected.
    """


def kept_indices(span: int, stride: int) -> list[int]:
    """Frame offsets written out of a segment of ``span`` frames.

    The full span is always simulated: the ego track has to stay at the
    simulator's 10 Hz or the finite differences behind obs[6..8] change meaning.
    Only the written frames are thinned, which is what buys scenes at a fixed
    storage budget.
    """
    if span < 2 or stride < 1:
        raise ValueError(f"span must be >= 2 and stride >= 1, got {span} and {stride}")
    return list(range(0, span, stride))


def _nearest_lane_distance(roads: np.ndarray) -> float:
    """Distance from the ego to the nearest lane centreline sample, in metres.

    Read off the already-cropped ego-frame road array, so this is exactly the
    geometry the renderer sees and costs no extra call into CARLA.
    """
    lane = roads[roads[:, 5] == ROAD_LANE]
    if not len(lane):
        return math.inf
    return float(np.linalg.norm(lane[:, 0:2], axis=1).min())


def _stray_fraction(frames: list[dict], threshold: float, stride: int = 10) -> float:
    """Fraction of recorded vehicle sightings further than ``threshold`` from a lane.

    Point-to-segment against the frame's own lane centrelines, so it measures the
    exact geometry the renderer drew.  Pedestrians are excluded: they belong on
    pavements, and counting them makes every healthy episode look broken.
    """
    stray = total = 0
    for frame in frames[::stride]:
        roads, agents = frame["roads"], frame["agents"]
        lane = roads[roads[:, 5] == ROAD_LANE]
        vehicles = agents[agents[:, 7] == VEHICLE]
        if not len(lane) or not len(vehicles):
            continue
        start, end = lane[:, 0:2], lane[:, 2:4]
        along = end - start
        length_sq = np.maximum((along * along).sum(1), 1e-9)
        offset = vehicles[:, None, 0:2] - start[None]
        t = np.clip((offset * along[None]).sum(2) / length_sq[None], 0.0, 1.0)
        distance = np.linalg.norm(start[None] + t[..., None] * along[None] - vehicles[:, None, 0:2], axis=2)
        stray += int((distance.min(1) > threshold).sum())
        total += len(vehicles)
    return stray / total if total else 0.0


def _box_center(transform, offset) -> tuple[float, float]:
    """World box centre in the right-handed frame, y negated out of CARLA's."""
    yaw = math.radians(transform.rotation.yaw)
    cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
    location = transform.location
    # Only the yaw term matters: pitch and roll of a grounded vehicle move the
    # box centre by millimetres, and the renderer is planar anyway.
    x = location.x + offset[0] * cos_yaw - offset[1] * sin_yaw
    y = location.y + offset[0] * sin_yaw + offset[1] * cos_yaw
    return x, -y


class Collector:
    """Owns the CARLA session for one run and yields finished episodes."""

    def __init__(self, args):
        import carla

        self.carla = carla
        self.args = args
        self.client = carla.Client(args.host, args.port)
        self.client.set_timeout(args.timeout)
        # Acquired per town in load_town, never here: see the note there.
        self.traffic_manager = None
        self.rig = rig_array()
        self.intrinsics, self.extrinsics, self.image_sizes = source_calibration()
        self.world = None
        self.town = None
        self.road_index = None
        # Tracked so muting an already-muted rig does not make the server warn.
        self._listening = False

    # -- session -----------------------------------------------------------

    def load_town(self, town: str) -> None:
        if self.town == town:
            return
        # Tear the Traffic Manager down before the world goes away. It lives in
        # the *server*, keyed by port, so ``get_trafficmanager`` hands back the
        # same instance no matter how often the handle is re-fetched; without
        # this it survives the world load still holding the previous town's
        # actor registry. That is what made the third town collected behave far
        # worse than the same town collected first.
        if self.traffic_manager is not None:
            try:
                self.traffic_manager.shut_down()
            except RuntimeError:
                pass
            self.traffic_manager = None

        self.world = self.client.load_world(town)
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self.args.dt
        self.world.apply_settings(settings)

        # The Traffic Manager must be re-acquired *after* load_world, every time.
        # Carrying one across a world load leaves it holding a registry of the
        # previous world's actors, and its path following then collapses: traffic
        # spawns correctly, drives off within five seconds, and ends up parked on
        # the pavement. Measured on Town01, 50 vehicles, 60 ticks -- 31 of 50 on
        # a Sidewalk lane with the fleet's median speed at 0.02 m/s, against
        # 0 of 50 at 7.02 m/s once the handle is re-acquired here.
        self.traffic_manager = self.client.get_trafficmanager(self.args.tm_port)
        self.traffic_manager.set_synchronous_mode(True)
        # Hybrid mode freezes physics for vehicles far from a hero actor, which
        # with our single hero leaves most of the fleet stationary.
        self.traffic_manager.set_hybrid_physics_mode(False)
        self.traffic_manager.set_global_distance_to_leading_vehicle(2.5)
        self.road_index = RoadIndex(town_roads(self.world.get_map(), self.args.road_resolution))
        self.town = town
        print(
            f"[{town}] {len(self.road_index.roads)} road segments "
            f"at {self.args.road_resolution} m resolution",
            flush=True,
        )

    def restore(self) -> None:
        """Hand the server back asynchronous, or the next client hangs on tick."""
        if self.world is None:
            return
        try:
            self.traffic_manager.set_synchronous_mode(False)
            settings = self.world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self.world.apply_settings(settings)
        except RuntimeError:
            pass

    # -- one episode -------------------------------------------------------

    def record_episode(self, base_index: int, rng: random.Random) -> list[dict]:
        carla = self.carla
        world = self.world
        spawned: list[int] = []
        sensors: list = []
        queues: list = []
        controllers: list[int] = []
        try:
            weather = Weather.sample(rng)
            world.set_weather(weather.to_carla(carla))
            self.traffic_manager.set_random_device_seed(rng.randrange(1 << 30))
            world.set_pedestrians_seed(rng.randrange(1 << 30))

            ego, ego_spawn = self._spawn_ego(rng)
            spawned.append(ego.id)
            spawned.extend(self._spawn_vehicles(rng, ego_spawn))
            walkers, controllers = self._spawn_walkers(rng)
            spawned.extend(walkers)

            sensors, queues = self._attach_cameras(ego)

            # Settle: vehicles drop onto their suspension and Traffic Manager
            # spreads them out. The cameras stay muted for all but the last two
            # ticks, which prime them and are discarded -- the first capture
            # after a spawn can come back before the scene is fully streamed in.
            prime = min(2, self.args.warmup)
            for step in range(self.args.warmup):
                if step == self.args.warmup - prime:
                    self._listen(sensors, queues)
                world.tick()
                if step >= self.args.warmup - prime:
                    for sensor_queue in queues:
                        sensor_queue.get(timeout=self.args.timeout)

            ego_transform = world.get_snapshot().find(ego.id).get_transform()
            ego_offset = (ego.bounding_box.location.x, ego.bounding_box.location.y)
            settled = self.road_index.crop(
                np.asarray(_box_center(ego_transform, ego_offset)),
                -math.radians(ego_transform.rotation.yaw),
                self.args.radius,
            )
            offroad = _nearest_lane_distance(settled)
            if offroad > self.args.max_offroad:
                raise OffRoadEgo(f"ego settled {offroad:.1f} m from the nearest lane")

            stray, fleet_speed = self._traffic_health(ego)
            if stray > self.args.max_stray_fraction:
                raise DegenerateTraffic(
                    f"{stray:.0%} of traffic is off-lane (median fleet speed {fleet_speed:.2f} m/s)"
                )

            actors = self._static_geometry(ego)
            recorded = self._run(ego, actors, sensors, queues, base_index, weather)

            # The warm-up gates above only prove the drive *started* on-road.
            # Traffic Manager also steers the ego into buildings mid-episode, and
            # those frames are correctly paired -- three cameras pressed against
            # a wall, with the abstract render faithfully showing black ground.
            # Only the recorded track reveals it, and it is judged per segment so
            # one bad window does not discard the good ones beside it.
            accepted = []
            for segment in recorded:
                strayed = max(_nearest_lane_distance(f["roads"]) for f in segment["frames"])
                if strayed > self.args.max_offroad:
                    print(
                        f"[{self.town}] {segment['segment_id']}: dropped, ego strayed "
                        f"{strayed:.1f} m from a lane",
                        flush=True,
                    )
                    continue
                stray = _stray_fraction(segment["frames"], self.args.max_offroad, stride=1)
                if stray > self.args.max_stray_fraction:
                    print(
                        f"[{self.town}] {segment['segment_id']}: dropped, {stray:.0%} of traffic "
                        "left the lanes",
                        flush=True,
                    )
                    continue
                accepted.append(segment)
            if not accepted:
                raise OffRoadEgo("no segment from this spawn stayed on the road")
            return self._deduplicate(accepted)
        finally:
            self._mute(sensors, queues)
            for sensor in sensors:
                sensor.destroy()
            if controllers:
                for controller in world.get_actors(controllers):
                    controller.stop()
                self.client.apply_batch([carla.command.DestroyActor(i) for i in controllers])
            if spawned:
                self.client.apply_batch([carla.command.DestroyActor(i) for i in spawned])
            world.tick()

    def _spawn_ego(self, rng: random.Random):
        """Spawn the ego and return it together with the transform it occupies.

        The transform is returned rather than read back later because in
        synchronous mode ``try_spawn_actor`` hands back an actor whose transform
        is still the origin until the next tick.  Trusting ``get_transform()``
        here silently placed traffic on top of the ego -- the spawn-point filter
        excluded a 1 m ball around (0, 0, 0) instead of around the car -- and a
        pinned ego records 91 frames of a stationary vehicle.
        """
        blueprint = self.world.get_blueprint_library().find(self.args.ego_blueprint)
        blueprint.set_attribute("role_name", "hero")
        points = self.world.get_map().get_spawn_points()
        rng.shuffle(points)
        for transform in points:
            actor = self.world.try_spawn_actor(blueprint, transform)
            if actor is not None:
                actor.set_autopilot(True, self.args.tm_port)
                return actor, transform
        raise RuntimeError(f"could not spawn the ego on {self.town}: every spawn point was occupied")

    def _spawn_vehicles(self, rng: random.Random, ego_spawn) -> list[int]:
        carla = self.carla
        library = [
            blueprint
            for blueprint in self.world.get_blueprint_library().filter("vehicle.*")
            if not blueprint.has_attribute("base_type")
            or blueprint.get_attribute("base_type").as_str() in ("car", "truck", "van", "bicycle")
        ]
        points = [
            point
            for point in self.world.get_map().get_spawn_points()
            if point.location.distance(ego_spawn.location) > 1.0
        ]
        rng.shuffle(points)
        batch = []
        for transform in points[: self.args.vehicles]:
            blueprint = rng.choice(library)
            if blueprint.has_attribute("color"):
                blueprint.set_attribute(
                    "color", rng.choice(blueprint.get_attribute("color").recommended_values)
                )
            blueprint.set_attribute("role_name", "autopilot")
            batch.append(
                carla.command.SpawnActor(blueprint, transform).then(
                    carla.command.SetAutopilot(
                        carla.command.FutureActor, True, self.traffic_manager.get_port()
                    )
                )
            )
        return [r.actor_id for r in self.client.apply_batch_sync(batch, True) if not r.error]

    def _spawn_walkers(self, rng: random.Random) -> tuple[list[int], list[int]]:
        carla = self.carla
        library = self.world.get_blueprint_library().filter("walker.pedestrian.*")
        batch, speeds = [], []
        for _ in range(self.args.walkers):
            location = self.world.get_random_location_from_navigation()
            if location is None:
                continue
            blueprint = rng.choice(library)
            if blueprint.has_attribute("is_invincible"):
                blueprint.set_attribute("is_invincible", "false")
            speed = 1.4
            if blueprint.has_attribute("speed"):
                speed = float(blueprint.get_attribute("speed").recommended_values[1])
            speeds.append(speed)
            batch.append(carla.command.SpawnActor(blueprint, carla.Transform(location)))
        walkers, kept_speeds = [], []
        for response, speed in zip(self.client.apply_batch_sync(batch, True), speeds):
            if not response.error:
                walkers.append(response.actor_id)
                kept_speeds.append(speed)
        if not walkers:
            return [], []

        controller_bp = self.world.get_blueprint_library().find("controller.ai.walker")
        batch = [
            carla.command.SpawnActor(controller_bp, carla.Transform(), walker) for walker in walkers
        ]
        controllers = [r.actor_id for r in self.client.apply_batch_sync(batch, True) if not r.error]
        self.world.tick()
        for controller, speed in zip(self.world.get_actors(controllers), kept_speeds):
            controller.start()
            controller.go_to_location(self.world.get_random_location_from_navigation())
            controller.set_max_speed(speed)
        return walkers, controllers

    def _attach_cameras(self, ego):
        carla = self.carla
        box_offset = (ego.bounding_box.location.x, ego.bounding_box.location.y)
        sensors, queues = [], []
        for camera in rig_cameras():
            blueprint = self.world.get_blueprint_library().find("sensor.camera.rgb")
            blueprint.set_attribute("image_size_x", str(REAL_WIDTH))
            blueprint.set_attribute("image_size_y", str(REAL_HEIGHT))
            blueprint.set_attribute("fov", f"{sensor_fov_deg(camera):.6f}")
            blueprint.set_attribute("sensor_tick", "0.0")
            (x, y, z), (pitch, yaw, roll) = mount(camera, box_offset)
            transform = carla.Transform(
                carla.Location(x=x, y=y, z=z),
                carla.Rotation(pitch=pitch, yaw=yaw, roll=roll),
            )
            sensor = self.world.spawn_actor(blueprint, transform, attach_to=ego)
            sensors.append(sensor)
            queues.append(queue.Queue())
        return sensors, queues

    def _listen(self, sensors, queues) -> None:
        for sensor, sensor_queue in zip(sensors, queues):
            sensor.listen(sensor_queue.put)
        self._listening = True

    def _mute(self, sensors, queues) -> None:
        """Unsubscribe, and drop anything already in flight.

        A stopped camera is not rendered at all -- measured 63.1 ms per tick
        while listening against 28.0 ms while stopped, the whole 35 ms being the
        three captures. At ``--frame-stride 10`` nine ticks in ten need no image,
        so muting between kept frames is most of the run's wall clock.
        """
        if not self._listening:
            return
        for sensor in sensors:
            sensor.stop()
        self._listening = False
        for sensor_queue in queues:
            while not sensor_queue.empty():
                sensor_queue.get_nowait()

    def _deduplicate(self, segments: list[dict]) -> list[dict]:
        """Drop segments the ego never drove far enough to make new.

        Consecutive segments share a spawn, so when the ego waits out a red light
        the next window opens exactly where the last one did. Measured on
        Town10HD, two such first frames differ by 4.0/255 -- nine times *less*
        than two frames one second apart inside a moving segment, so counting
        them as two scenes overstates the dataset.

        Displacement accumulates from the last *kept* segment rather than the
        previous one, so a stop is not deleted -- the first window at the light
        is kept, and only the copies behind it are dropped. That matters: a car
        waiting at a light is about a fifth of Waymo's training set.
        """
        threshold = self.args.min_segment_displacement
        if threshold <= 0 or not segments:
            return segments
        kept = [segments[0]]
        for segment in segments[1:]:
            moved = math.dist(segment["start_xy"], kept[-1]["start_xy"])
            if moved < threshold:
                print(
                    f"[{self.town}] {segment['segment_id']}: dropped, ego only moved "
                    f"{moved:.1f} m since {kept[-1]['segment_id']}",
                    flush=True,
                )
                continue
            kept.append(segment)
        return kept

    def _traffic_health(self, ego) -> tuple[float, float]:
        """Return (fraction of traffic off a driving lane, fleet median speed)."""
        carla = self.carla
        offroad, speeds = [], []
        lane_map = self.world.get_map()
        for actor in self.world.get_actors().filter("vehicle.*"):
            if actor.id == ego.id:
                continue
            transform = actor.get_transform()
            velocity = actor.get_velocity()
            speeds.append(math.hypot(velocity.x, velocity.y))
            waypoint = lane_map.get_waypoint(
                transform.location, project_to_road=True, lane_type=carla.LaneType.Driving
            )
            distance = (
                transform.location.distance(waypoint.transform.location)
                if waypoint is not None
                else math.inf
            )
            offroad.append(distance > self.args.max_offroad)
        if not offroad:
            return 0.0, 0.0
        return float(np.mean(offroad)), float(np.median(speeds))

    def _static_geometry(self, ego) -> list[_Actor]:
        actors = []
        for actor in self.world.get_actors():
            if actor.id == ego.id:
                continue
            render_type = _agent_type(actor)
            if render_type:
                actors.append(_Actor(actor, render_type))
        return actors

    def _run(self, ego, actors, sensors, queues, base_index: int, weather: Weather) -> list[dict]:
        """Record consecutive segments from one spawn.

        A segment is the goal window, and it is why the ego keeps driving rather
        than the segments getting longer. The frozen head trained on a goal that
        decays to zero over exactly 91 steps; stretch that window and ``obs[0]``
        starts an order of magnitude too large, off the manifold the head ever
        saw. Recording several 91-step windows back to back gives a longer
        continuous drive with the goal distribution intact, and amortises the
        warm-up over all of them.
        """
        world = self.world
        span = self.args.segment_frames
        stride = self.args.frame_stride
        ego_box = ego.bounding_box
        ego_offset = (ego_box.location.x, ego_box.location.y)
        ego_length, ego_width = 2.0 * ego_box.extent.x, 2.0 * ego_box.extent.y

        # With no thinning every tick is a capture, so toggling would only add
        # two RPC round trips per frame for nothing.
        always_on = stride == 1
        if always_on:
            self._listen(sensors, queues)

        segments = []
        for offset in range(self.args.segments_per_spawn):
            index = base_index + offset
            frames, centers, yaws, stamps = [], [], [], []
            for step in range(span):
                keep = step % stride == 0
                # Subscribing before the tick is what makes the capture land on
                # this exact frame; _await_frame then asserts the frame id, so a
                # mistimed toggle fails loudly rather than pairing the abstract
                # scene with a neighbouring image.
                # Render from `camera_prime` ticks before each kept frame:
                # Unreal's temporal anti-aliasing accumulates across frames, so a
                # capture taken with no history is measurably more aliased than
                # the same frame in a continuously rendered run. Expressed as
                # distance to the *next* kept frame so step 0 is covered -- a
                # condition keyed on the previous one leaves it unsubscribed and
                # _await_frame then blocks on an empty queue.
                if always_on or (-step) % stride <= self.args.camera_prime:
                    if not self._listening:
                        self._listen(sensors, queues)
                frame_id = world.tick()
                if keep:
                    raw = [self._await_frame(sensor_queue, frame_id) for sensor_queue in queues]
                    if not always_on:
                        self._mute(sensors, queues)
                else:
                    raw = None
                    if self._listening:
                        # A priming frame is rendered, then thrown away.
                        for sensor_queue in queues:
                            sensor_queue.get(timeout=self.args.timeout)
                snapshot = world.get_snapshot()

                ego_transform = snapshot.find(ego.id).get_transform()
                center = _box_center(ego_transform, ego_offset)
                yaw = -math.radians(ego_transform.rotation.yaw)
                stamp = (index * span + step) * FRAME_INTERVAL_MICROS
                centers.append(center)
                yaws.append(yaw)
                stamps.append(stamp)

                # Every tick is simulated -- the ego track has to stay at the
                # simulator's 10 Hz for the finite differences behind obs[6..8]
                # to mean anything -- but only every stride-th frame is kept.
                if not keep:
                    continue
                frames.append(
                    {
                        "frame_index": step,
                        "timestamp_micros": stamp,
                        "real_images": np.stack([_decode(image) for image in raw]),
                        "agents": self._agents(snapshot, actors, center, yaw),
                        "roads": self.road_index.crop(np.asarray(center), yaw, self.args.radius),
                    }
                )

            # Built on the full-rate track, then subsampled onto the kept frames,
            # so striding never changes a single ego value.
            ego_obs = episode_ego_obs(
                np.asarray(centers),
                np.asarray(yaws),
                np.asarray(stamps, dtype=np.int64),
                length=ego_length,
                width=ego_width,
            )
            segments.append(
                {
                    "segment_id": f"{self.town}_{index:05d}",
                    "start_xy": tuple(centers[0]),
                    "frames": frames,
                    "ego_obs": ego_obs[kept_indices(span, stride)],
                    "weather": asdict(weather),
                    "ego_blueprint": ego.type_id,
                    "ego_length": ego_length,
                    "ego_width": ego_width,
                }
            )
        return segments

    def _await_frame(self, sensor_queue: queue.Queue, frame_id: int):
        """Drain until the image belonging to this tick arrives, undecoded.

        A sensor can be a frame behind after a spawn, so matching on the frame id
        rather than trusting queue order is what keeps the three views and the
        actor snapshot describing one instant.
        """
        while True:
            image = sensor_queue.get(timeout=self.args.timeout)
            if image.frame == frame_id:
                return image
            if image.frame > frame_id:
                raise RuntimeError(f"sensor overran: got frame {image.frame}, wanted {frame_id}")

    def _agents(self, snapshot, actors: list[_Actor], center, yaw: float) -> np.ndarray:
        rows = []
        radius_sq = self.args.radius**2
        for actor in actors:
            actor_snapshot = snapshot.find(actor.id)
            if actor_snapshot is None:
                continue
            transform = actor_snapshot.get_transform()
            world_xy = _box_center(transform, actor.offset)
            dx, dy = world_xy[0] - center[0], world_xy[1] - center[1]
            if dx * dx + dy * dy > radius_sq:
                continue
            x, y = world_to_ego(np.asarray(world_xy), np.asarray(center), yaw)
            heading = -math.radians(transform.rotation.yaw) - yaw
            rows.append(
                [
                    x,
                    y,
                    math.cos(heading),
                    math.sin(heading),
                    actor.length,
                    actor.width,
                    actor.height,
                    float(actor.type),
                ]
            )
        if not rows:
            return np.zeros((0, 8), dtype=np.float32)
        return np.ascontiguousarray(np.asarray(rows, dtype=np.float32))


def write_episode(episode: dict, output: Path, rig, intrinsics, extrinsics, sizes) -> list[dict]:
    """Write one episode's processed samples and its ego table."""
    processed_dir = output / "processed"
    ego_dir = output / "ego_state"
    processed_dir.mkdir(parents=True, exist_ok=True)
    ego_dir.mkdir(parents=True, exist_ok=True)
    segment_id = episode["segment_id"]
    camera_names = np.asarray(CAMERA_NAMES)

    entries = []
    for frame in episode["frames"]:
        name = f"{segment_id}__{frame['timestamp_micros']}.npz"
        atomic_savez(
            processed_dir / name,
            schema_version=np.asarray(PROCESSED_SCHEMA_VERSION),
            segment_id=np.asarray(segment_id),
            timestamp_micros=np.asarray(frame["timestamp_micros"], dtype=np.int64),
            frame_index=np.asarray(frame["frame_index"], dtype=np.int32),
            camera_names=camera_names,
            real_images=frame["real_images"],
            agents=frame["agents"],
            roads=frame["roads"],
            ego=np.asarray([0.0, 0.0, 1.0, 0.0, -1.0], dtype=np.float32),
            rig=rig,
            source_intrinsics=intrinsics,
            source_extrinsics=extrinsics,
            source_image_sizes=sizes,
        )
        entries.append(
            {
                "file": name,
                "segment_id": segment_id,
                "timestamp_micros": int(frame["timestamp_micros"]),
                "frame_index": int(frame["frame_index"]),
            }
        )

    atomic_savez(
        ego_dir / f"{segment_id}.npz",
        schema_version=np.asarray(EGO_SCHEMA_VERSION),
        segment_id=np.asarray(segment_id),
        timestamp_micros=np.asarray(
            [f["timestamp_micros"] for f in episode["frames"]], dtype=np.int64
        ),
        ego_obs=episode["ego_obs"],
        # Provenance: this table uses the episode's own endpoint, not the
        # rolling lookahead the Waymo extractor defaults to.
        goal_mode=np.asarray("endpoint"),
    )
    return entries


def episode_record(episode: dict, town: str, seed: int) -> dict:
    """One row of ``episodes.jsonl``: everything about a segment but its pixels.

    The sky is the reason this file exists.  Weather is the axis CARLA adds that
    a recorded log cannot, so it has to be queryable after the fact -- to
    condition on it, to ablate it, or to explain a validation number by the
    conditions the segment was collected under.  Without this the sky is applied
    and then lost.
    """
    return {
        "segment_id": episode["segment_id"],
        "town": town,
        "seed": int(seed),
        "frames": len(episode["frames"]),
        "weather": episode["weather"],
        "ego_blueprint": episode["ego_blueprint"],
        "ego_length": float(episode["ego_length"]),
        "ego_width": float(episode["ego_width"]),
        "mean_speed": float(episode["ego_obs"][:, 2].mean()) * MAX_SPEED,
        "mean_agents": float(np.mean([len(f["agents"]) for f in episode["frames"]])),
        # Rejection happens post-warm-up; this is the worst the ego strayed
        # while recording, so mid-episode excursions stay queryable.
        "max_offroad": max(_nearest_lane_distance(f["roads"]) for f in episode["frames"]),
        # Where the segment opened, in world metres, so segments sharing a spawn
        # can be checked for overlap after the fact.
        "start_xy": [float(v) for v in episode.get("start_xy", (float("nan"),) * 2)],
    }


def episode_seed(seed: int, town: str, episode: int, tag: str = "") -> int:
    """A stable per-episode seed.

    Deliberately not ``hash()``: Python salts string hashing per process, so a
    tuple hash would hand the same episode a different sky on every run and
    quietly break ``--resume``'s promise to reproduce the slot it skipped.

    ``tag`` defaults to the split directory's name and exists to stop a silent
    leak.  Everything about an episode -- the sky, the Traffic Manager seed, the
    spawn point -- follows from this number, so collecting a validation split
    with the same ``--seed`` as training would reproduce the *same episodes*,
    not merely similar ones.  Folding the split name in makes the two disjoint
    by default rather than by remembering.
    """
    return zlib.crc32(f"{seed}:{tag}:{town}:{episode}".encode()) & 0xFFFFFFFF


def _existing_segments(output: Path) -> set[str]:
    manifest = output / "processed" / "manifest.jsonl"
    if not manifest.is_file():
        return set()
    segments = set()
    for line in manifest.read_text().splitlines():
        if line.strip():
            segments.add(json.loads(line)["segment_id"])
    return segments


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True, help="split directory to write")
    parser.add_argument("--town", action="append", dest="towns", default=None)
    parser.add_argument("--episodes", type=int, default=100, help="segments per town")
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="write every Nth frame of a segment. Every tick is still simulated -- the ego "
        "track must stay at 10 Hz for the finite differences behind obs[6..8] -- so this "
        "trades frames for scenes at fixed storage, which is the axis that matters",
    )
    parser.add_argument(
        "--segment-frames",
        type=int,
        default=EPISODE_FRAMES,
        help="frames per segment, which is also the goal window. Changing it moves obs[0] off "
        "the distribution the frozen planning head trained on; to drive for longer, raise "
        "--segments-per-spawn instead",
    )
    parser.add_argument(
        "--camera-prime",
        type=int,
        default=2,
        help="ticks to start rendering before each kept frame so Unreal's temporal "
        "anti-aliasing has history; 0 captures cold and is measurably more aliased",
    )
    parser.add_argument(
        "--min-segment-displacement",
        type=float,
        default=25.0,
        help="drop a segment whose ego has moved less than this (metres) since the last kept "
        "segment of the same spawn; 0 disables. Guards against a stopped ego turning one "
        "scene into --segments-per-spawn copies of itself",
    )
    parser.add_argument(
        "--segments-per-spawn",
        type=int,
        default=1,
        help="consecutive segments recorded from one spawn, giving a longer continuous drive "
        "and amortising the warm-up across all of them",
    )
    parser.add_argument("--vehicles", type=int, default=60)
    parser.add_argument("--walkers", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=8000)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--dt", type=float, default=DEFAULT_DT)
    parser.add_argument("--warmup", type=int, default=30, help="ticks to discard before recording")
    parser.add_argument("--radius", type=float, default=DEFAULT_RADIUS)
    parser.add_argument("--road-resolution", type=float, default=1.0)
    parser.add_argument("--ego-blueprint", default="vehicle.lincoln.mkz_2020")
    parser.add_argument(
        "--max-offroad",
        type=float,
        default=4.0,
        help="reject an episode whose ego settles further than this from a lane centreline "
        "(metres); spawning dense traffic shoves cars onto pavements and grass",
    )
    parser.add_argument(
        "--max-stray-fraction",
        type=float,
        default=0.25,
        help="reject an episode when more than this fraction of traffic has left the driving "
        "lanes after the warm-up; a healthy fleet measures 0.00",
    )
    parser.add_argument(
        "--max-attempts", type=int, default=4, help="respawn attempts before skipping an episode"
    )
    parser.add_argument(
        "--split-tag",
        default=None,
        help="folded into the per-episode seed so splits differ; defaults to the output "
        "directory's name, which keeps training and validation disjoint automatically",
    )
    parser.add_argument("--allow-elevated-towns", action="store_true",
                        help="permit towns the flat-ground rasterizer cannot represent")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--resume", action="store_true", help="skip segments already in the manifest")
    group.add_argument("--overwrite", action="store_true", help="discard an existing split")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    towns = tuple(args.towns) if args.towns else DEFAULT_TOWNS
    if not args.allow_elevated_towns:
        elevated = [town for town in towns if town not in FLAT_TOWNS]
        if elevated:
            raise SystemExit(
                f"{', '.join(elevated)} have elevation the flat-ground rasterizer cannot "
                "represent; roads render at z=0 and agent boxes sit on z=0. Pass "
                "--allow-elevated-towns to record them anyway."
            )
    if args.frame_stride < 1 or args.segments_per_spawn < 1 or args.segment_frames < 2:
        raise SystemExit("--frame-stride and --segments-per-spawn must be >= 1, --segment-frames >= 2")
    if args.segment_frames != EPISODE_FRAMES:
        print(
            f"warning: --segment-frames {args.segment_frames} is not the simulator's "
            f"{EPISODE_FRAMES}. The segment is the goal window, so obs[0] will start at roughly "
            f"{args.segment_frames / EPISODE_FRAMES:.1f}x what the frozen planning head trained "
            "on. Use --segments-per-spawn for a longer drive instead.",
            file=sys.stderr,
        )
    if abs(args.dt - DEFAULT_DT) > 1e-9:
        print(
            f"warning: --dt {args.dt} differs from the simulator's 0.1; the finite-differenced "
            "ego accelerations will land in a different part of the planning head's distribution",
            file=sys.stderr,
        )

    output: Path = args.output
    done: set[str] = set()
    if output.exists():
        if args.resume:
            done = _existing_segments(output)
        elif args.overwrite:
            # Guard against --overwrite pointed at something that is not one of
            # ours: only a split this script wrote gets deleted.
            if not (output / "processed" / "manifest.jsonl").is_file() and any(output.iterdir()):
                raise SystemExit(
                    f"{output} is not empty and has no processed/manifest.jsonl, so it does not "
                    "look like a split this script wrote; refusing to delete it"
                )
            existing = len(done := _existing_segments(output))
            print(f"--overwrite: discarding {existing} segments under {output}", flush=True)
            for name in ("processed", "ego_state"):
                shutil.rmtree(output / name, ignore_errors=True)
            (output / "episodes.jsonl").unlink(missing_ok=True)
            done = set()
        else:
            raise SystemExit(f"{output} already exists; pass --resume or --overwrite")

    split_tag = args.split_tag if args.split_tag is not None else output.name
    collector = Collector(args)
    manifest_path = output / "processed" / "manifest.jsonl"
    episodes_path = output / "episodes.jsonl"
    written = 0
    per_spawn = args.segments_per_spawn
    try:
        for town in towns:
            collector.load_town(town)
            # One spawn yields `per_spawn` consecutive segments, so the loop steps
            # over spawns while --episodes keeps meaning segments per town.
            for base in range(0, args.episodes, per_spawn):
                wanted = [
                    f"{town}_{base + offset:05d}"
                    for offset in range(min(per_spawn, args.episodes - base))
                ]
                # The first segment of a spawn is always kept, so its presence
                # is what marks the spawn done. Testing all of them would retry
                # for ever whenever the displacement filter dropped one.
                if wanted[0] in done:
                    continue
                seed = episode_seed(args.seed, town, base, split_tag)
                recorded = None
                for attempt in range(args.max_attempts):
                    # Each attempt draws its own spawn point, deterministically,
                    # so --resume still reproduces whatever landed in this slot.
                    try:
                        recorded = collector.record_episode(base, random.Random(seed + attempt))
                        break
                    except (OffRoadEgo, DegenerateTraffic) as error:
                        print(f"[{town}] {wanted[0]}: {error}; respawning", flush=True)
                if recorded is None:
                    print(
                        f"[{town}] {wanted[0]}: no on-road spawn in {args.max_attempts} attempts; "
                        "skipping",
                        flush=True,
                    )
                    continue
                for segment in recorded:
                    if segment["segment_id"] in done or segment["segment_id"] not in wanted:
                        continue
                    entries = write_episode(
                        segment,
                        output,
                        collector.rig,
                        collector.intrinsics,
                        collector.extrinsics,
                        collector.image_sizes,
                    )
                    manifest_path.parent.mkdir(parents=True, exist_ok=True)
                    with manifest_path.open("a") as handle:
                        for entry in entries:
                            handle.write(json.dumps(entry) + "\n")
                        handle.flush()
                        os.fsync(handle.fileno())
                    # The segment row lands after its frames, so an interrupted
                    # run never claims a segment the manifest does not have.
                    record = episode_record(segment, town, seed)
                    with episodes_path.open("a") as handle:
                        handle.write(json.dumps(record) + "\n")
                        handle.flush()
                        os.fsync(handle.fileno())
                    written += len(entries)
                    # Mean speed is reported because a pinned ego is otherwise
                    # invisible here: it still yields well-formed frames.
                    speed, agents = record["mean_speed"], record["mean_agents"]
                    print(
                        f"[{town}] {segment['segment_id']}: {len(entries)} frames, "
                        f"{agents:.0f} agents in view, mean speed {speed:5.2f} m/s, "
                        f"{written} total" + ("   <-- stationary" if speed < 0.5 else ""),
                        flush=True,
                    )
    finally:
        collector.restore()
    kept = len(kept_indices(args.segment_frames, args.frame_stride))
    print(json.dumps({"segments": written // max(kept, 1), "frames": written}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
