"""Offline checks for the CARLA sim-to-real producer.

None of these need a running CARLA server.  ``carla.Map(name, xodr_content)``
builds a full road network from an ``.xodr`` string in-process, which is what
lets the coordinate-frame test -- the one that actually matters -- run in CI.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from data_utils.carla_sim2real import EPISODE_FRAMES, REAL_HEIGHT, REAL_WIDTH
from data_utils.carla_sim2real.collect import (
    PEDESTRIAN,
    VEHICLE,
    Weather,
    _agent_type,
    _box_center,
    episode_seed,
    kept_indices,
    write_episode,
)
from data_utils.carla_sim2real.ego import episode_ego_obs, pose_matrices
from data_utils.carla_sim2real.rig import (
    SENSOR_TO_CV,
    mount,
    rig_array,
    rig_cameras,
    sensor_fov_deg,
    sensor_intrinsics,
    source_calibration,
)
from data_utils.carla_sim2real.roads import RoadIndex, town_roads, world_to_ego
from data_utils.carla_sim2real.split_distillation import split_segments
from data_utils.waymo_sim2real.processed import (
    EGO_OBS_DIM,
    SIM_HEIGHT,
    SIM_WIDTH,
    load_ego_state,
    load_processed,
    validate_processed,
)
from data_utils.waymo_sim2real.render_roads import RENDER_LANE_AREA, ROAD_LANE, prepare_runtime_roads
from pufferlib.ocean.drive.raster_ref import WAYMO_RIG, rig_tensor


CARLA_XODR = Path("/home/tjhu78u/CARLA_0_9_16/CarlaUE4/Content/Carla/Maps/OpenDrive")
PY123D = Path("data_utils/carla/carla_py123d")


def test_distillation_split_is_town_stratified_deterministic_and_leak_free():
    episodes = [
        {"segment_id": f"{town}_{index:02d}", "town": town}
        for town in ("Town01", "Town02", "Town10HD")
        for index in range(10)
    ]
    first = split_segments(episodes, train_per_town=7, validation_per_town=3, seed=42)
    second = split_segments(episodes, train_per_town=7, validation_per_town=3, seed=42)
    assert first == second
    assert not first["training"] & first["validation"]
    for town in ("Town01", "Town02", "Town10HD"):
        assert sum(segment.startswith(town) for segment in first["training"]) == 7
        assert sum(segment.startswith(town) for segment in first["validation"]) == 3


def _carla_map(town: str):
    carla = pytest.importorskip("carla")
    path = CARLA_XODR / f"{town}.xodr"
    if not path.is_file():
        pytest.skip(f"no CARLA OpenDRIVE for {town} at {path}")
    return carla.Map(town, path.read_text())


# ---------------------------------------------------------------------------
# Coordinate frame
# ---------------------------------------------------------------------------


def test_extracted_roads_land_on_the_repositorys_own_carla_geometry():
    """CARLA is left-handed, this repository is not, and y is the whole difference.

    ``carla_py123d/Town01.json`` was produced by an external tool in the
    right-handed OpenDRIVE frame; the live ``carla.Map`` reports the left-handed
    world frame.  If the negation in ``town_roads`` were missing or doubled, the
    two point sets would be mirrored about y and this would blow up -- which is
    the cheapest possible proof of the convention, and it needs no server.
    """
    reference_path = PY123D / "Town01.json"
    if not reference_path.is_file():
        pytest.skip(f"missing {reference_path}")
    reference = json.loads(reference_path.read_text())
    expected = np.asarray(
        [[p["x"], p["y"]] for road in reference["roads"] for p in road["geometry"]], dtype=np.float64
    )

    roads = town_roads(_carla_map("Town01"), resolution=1.0)
    assert len(roads)
    ours = np.concatenate([roads[:, 0:2], roads[:, 2:4]]).astype(np.float64)

    # Bulk extent first: a mirrored y would show up here before any matching.
    assert ours[:, 0].min() == pytest.approx(expected[:, 0].min(), abs=5.0)
    assert ours[:, 0].max() == pytest.approx(expected[:, 0].max(), abs=5.0)
    assert ours[:, 1].min() == pytest.approx(expected[:, 1].min(), abs=5.0)
    assert ours[:, 1].max() == pytest.approx(expected[:, 1].max(), abs=5.0)

    # Then per-point: every reference vertex must have one of ours nearby. The
    # two samplings differ, so this bounds distance, not correspondence -- the
    # residual is the 1 m sampling grid, hence sub-metre rather than zero.
    # Chunked because the full pairwise matrix would be half a gigabyte.
    sample = expected[:: max(1, len(expected) // 2000)]
    distance = np.concatenate(
        [
            np.linalg.norm(sample[start : start + 200, None, :] - ours[None, :, :], axis=2).min(axis=1)
            for start in range(0, len(sample), 200)
        ]
    )
    assert np.percentile(distance, 95) < 1.0, f"95th percentile offset {np.percentile(distance, 95)}"
    assert distance.max() < 3.0, f"worst offset {distance.max()}"


def test_world_to_ego_puts_the_heading_on_positive_x():
    center = np.asarray([10.0, -4.0])
    yaw = math.radians(30.0)
    ahead = center + 5.0 * np.asarray([math.cos(yaw), math.sin(yaw)])
    left = center + 3.0 * np.asarray([-math.sin(yaw), math.cos(yaw)])
    assert world_to_ego(ahead, center, yaw) == pytest.approx([5.0, 0.0], abs=1e-9)
    assert world_to_ego(left, center, yaw) == pytest.approx([0.0, 3.0], abs=1e-9)
    assert world_to_ego(center, center, yaw) == pytest.approx([0.0, 0.0], abs=1e-9)


def test_box_center_negates_carla_y_and_applies_the_actor_offset():
    transform = SimpleNamespace(
        location=SimpleNamespace(x=100.0, y=50.0, z=0.0), rotation=SimpleNamespace(yaw=90.0)
    )
    # A +1 m forward offset in the actor frame, with the actor facing CARLA +y.
    x, y = _box_center(transform, (1.0, 0.0, 0.7))
    assert x == pytest.approx(100.0, abs=1e-6)
    assert y == pytest.approx(-51.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Rig
# ---------------------------------------------------------------------------


def test_rig_is_the_simulators_own_waymo_rig():
    rig = rig_array()
    assert rig.shape == (3, 20) and rig.dtype == np.float32
    assert np.allclose(rig, rig_tensor(WAYMO_RIG).numpy())
    assert (rig[:, 16] == SIM_WIDTH).all() and (rig[:, 17] == SIM_HEIGHT).all()


def test_carla_fov_reproduces_the_render_intrinsics_at_four_times_the_resolution():
    """The captured image and the abstract render must be one pinhole camera.

    CARLA derives its projection from the horizontal FOV alone and centres the
    principal point, exactly as ``Camera.intrinsics()`` does, so passing this
    angle at 384x256 gives precisely 4x the 96x64 focal length.
    """
    for camera in rig_cameras():
        fx_render, fy_render, cx_render, cy_render = camera.intrinsics()
        fx_real, fy_real, cx_real, cy_real = sensor_intrinsics(camera)
        assert fx_real == pytest.approx(4.0 * fx_render)
        assert fy_real == pytest.approx(4.0 * fy_render)
        assert cx_real == pytest.approx(4.0 * cx_render) == pytest.approx(REAL_WIDTH / 2)
        assert cy_real == pytest.approx(4.0 * cy_render) == pytest.approx(REAL_HEIGHT / 2)

        # CARLA rebuilds fx from the angle; that round trip must be lossless.
        fov = sensor_fov_deg(camera)
        rebuilt = (REAL_WIDTH / 2.0) / math.tan(math.radians(fov) / 2.0)
        assert rebuilt == pytest.approx(fx_real, rel=1e-9)


def test_camera_mounts_negate_carla_s_left_handed_yaw():
    """A sign error here silently swaps front_left and front_right."""
    cameras = {camera.name: camera for camera in rig_cameras()}
    box_offset = (0.05, 0.0)

    (x, _y, z), (pitch, yaw, roll) = mount(cameras["front"], box_offset)
    assert (pitch, yaw, roll) == (0.0, 0.0, 0.0)
    assert z == pytest.approx(cameras["front"].pos[2])
    assert x == pytest.approx(box_offset[0] + cameras["front"].pos[0])

    # The rig yaws front_left to the ego's left (+44.6); CARLA's positive yaw
    # turns right, so the mounted sensor must carry the negation.
    assert mount(cameras["front_left"], box_offset)[1][1] == pytest.approx(-44.6)
    assert mount(cameras["front_right"], box_offset)[1][1] == pytest.approx(44.7)
    # y likewise: the rig puts front_left at positive (left) y, CARLA at negative.
    assert cameras["front_left"].pos[1] > 0 > mount(cameras["front_left"], box_offset)[0][1]


def test_source_calibration_round_trips_to_the_rig():
    """The stored extrinsics must be the ones the rig was built from.

    ``preprocess`` derives ``rig[:, :9]`` as ``SENSOR_TO_CV @ extrinsic[:3,:3].T``;
    running that on what we store has to give the rig back, or the geometry
    audits read a calibration the renderer never used.
    """
    intrinsics, extrinsics, sizes = source_calibration()
    rig = rig_array()
    assert intrinsics.shape == (3, 9) and extrinsics.shape == (3, 4, 4) and sizes.shape == (3, 2)
    assert (sizes == np.asarray([REAL_HEIGHT, REAL_WIDTH])).all()
    for index in range(3):
        recovered = SENSOR_TO_CV @ extrinsics[index, :3, :3].T
        assert np.allclose(recovered, rig[index, :9].reshape(3, 3), atol=1e-6)
        assert np.allclose(extrinsics[index, :3, 3], rig[index, 9:12], atol=1e-6)
        # No lens distortion is enabled on CARLA's RGB camera.
        assert (intrinsics[index, 4:] == 0).all()


# ---------------------------------------------------------------------------
# Roads
# ---------------------------------------------------------------------------


def test_crop_keeps_lane_areas_last_so_the_ground_renders_black():
    """``raster.cu:619`` reads only the final road row to pick the ground colour."""
    roads = np.asarray(
        [
            [0.0, 0.0, 1.0, 0.0, 4.5, ROAD_LANE],
            [0.0, 2.0, 1.0, 2.0, 0.15, 5.0],
            [0.0, 4.0, 1.0, 4.0, 0.25, 6.0],
        ],
        dtype=np.float32,
    )
    cropped = RoadIndex(roads).crop(np.zeros(2), 0.0, 100.0)
    assert len(cropped) == 3
    assert int(cropped[-1, 5]) == ROAD_LANE
    assert int(prepare_runtime_roads(cropped)[-1, 5]) == RENDER_LANE_AREA


def test_crop_rejects_segments_outside_the_radius():
    roads = np.asarray(
        [
            [0.0, 0.0, 1.0, 0.0, 4.5, ROAD_LANE],
            [500.0, 0.0, 501.0, 0.0, 4.5, ROAD_LANE],
        ],
        dtype=np.float32,
    )
    cropped = RoadIndex(roads).crop(np.zeros(2), 0.0, 220.0)
    assert len(cropped) == 1
    assert cropped[0, 0] == pytest.approx(0.0)


def test_town_roads_emits_canonical_types_only():
    roads = town_roads(_carla_map("Town01"), resolution=2.0)
    assert roads.dtype == np.float32 and roads.shape[1] == 6
    assert set(np.unique(roads[:, 5]).astype(int)) <= {4, 5, 6, 8}
    # A town without lane centrelines would render as black ground everywhere.
    assert (roads[:, 5] == ROAD_LANE).sum() > 100
    assert int(roads[-1, 5]) == ROAD_LANE


# ---------------------------------------------------------------------------
# Ego observations
# ---------------------------------------------------------------------------


def _straight_episode(speed: float = 10.0, frames: int = EPISODE_FRAMES):
    times = (np.arange(frames) * 100_000).astype(np.int64)
    centers = np.stack([speed * 0.1 * np.arange(frames), np.zeros(frames)], axis=1)
    return centers, np.zeros(frames), times


def test_endpoint_goal_decays_to_zero_unlike_the_rolling_lookahead():
    """This is the distribution the frozen head actually trained on.

    ``drive_3cam.ini`` sets ``goal_behavior = 0``, so the simulator's goal is a
    fixed world point that the ego drives into over the 91-step episode, not a
    lookahead that stays 30 m ahead forever.
    """
    centers, yaws, times = _straight_episode()
    obs = episode_ego_obs(centers, yaws, times, length=4.9, width=2.0)
    assert obs.shape == (EPISODE_FRAMES, EGO_OBS_DIM) and obs.dtype == np.float32
    assert np.all(np.diff(obs[:, 0]) < 1e-9)
    assert obs[-1, 0] == pytest.approx(0.0, abs=1e-6)
    # 90 m of travel at 0.005 m^-1.
    assert obs[0, 0] == pytest.approx(0.45, abs=1e-3)
    assert abs(obs[:, 1]).max() < 1e-6


def test_ego_observation_slots_match_the_simulator_normalizers():
    centers, yaws, times = _straight_episode(speed=10.0)
    obs = episode_ego_obs(centers, yaws, times, length=4.9, width=2.0)
    assert obs[5, 2] == pytest.approx(10.0 / 100.0)
    assert obs[5, 3] == pytest.approx(2.0 / 15.0)
    assert obs[5, 4] == pytest.approx(4.9 / 30.0)
    assert obs[5, 5] == 0.0 and obs[5, 9] == 0.0
    assert obs[5, 10] == pytest.approx(1.0 / 3.0)
    # A straight constant-speed drive has no steering, no acceleration.
    assert abs(obs[5, 6]) < 1e-6 and abs(obs[5, 7]) < 1e-6 and abs(obs[5, 8]) < 1e-6


def test_pose_matrices_encode_yaw_as_a_rotation_block():
    poses = pose_matrices(np.asarray([[1.0, 2.0]]), np.asarray([math.radians(90.0)]))
    assert poses.shape == (1, 4, 4)
    assert poses[0, :2, 3] == pytest.approx([1.0, 2.0])
    assert math.atan2(poses[0, 1, 0], poses[0, 0, 0]) == pytest.approx(math.radians(90.0))


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def _synthetic_episode(frames: int = 3) -> dict:
    rng = np.random.default_rng(0)
    centers, yaws, times = _straight_episode(frames=frames)
    return {
        "segment_id": "Town01_00000",
        "frames": [
            {
                "frame_index": index,
                "timestamp_micros": int(times[index]),
                "real_images": rng.integers(
                    0, 256, (3, REAL_HEIGHT, REAL_WIDTH, 3), dtype=np.uint8
                ),
                "agents": np.asarray(
                    [[12.0, 1.5, 1.0, 0.0, 4.7, 2.0, 1.6, float(VEHICLE)]], dtype=np.float32
                ),
                "roads": np.asarray(
                    [
                        [0.0, 3.0, 10.0, 3.0, 0.15, 5.0],
                        [-20.0, 0.0, 60.0, 0.0, 4.5, float(ROAD_LANE)],
                    ],
                    dtype=np.float32,
                ),
            }
            for index in range(frames)
        ],
        "ego_obs": episode_ego_obs(centers, yaws, times, length=4.9, width=2.0),
        "weather": {},
    }


def test_written_episode_satisfies_the_waymo_processed_schema(tmp_path: Path):
    """The whole point: the downstream half must not be able to tell the source."""
    episode = _synthetic_episode()
    intrinsics, extrinsics, sizes = source_calibration()
    entries = write_episode(episode, tmp_path, rig_array(), intrinsics, extrinsics, sizes)

    assert len(entries) == len(episode["frames"])
    for entry in entries:
        sample = load_processed(tmp_path / "processed" / entry["file"])
        validate_processed(sample)
        assert sample["real_images"].shape == (3, REAL_HEIGHT, REAL_WIDTH, 3)
        assert np.allclose(sample["ego"], [0.0, 0.0, 1.0, 0.0, -1.0])
        assert sample["rig"].shape == (3, 20)
        # Renderable straight out of the file, lane area last.
        runtime = prepare_runtime_roads(sample["roads"])
        assert int(runtime[-1, 5]) == RENDER_LANE_AREA

    state = load_ego_state(tmp_path / "ego_state" / "Town01_00000.npz")
    assert state["ego_obs"].shape == (len(episode["frames"]), EGO_OBS_DIM)
    assert list(state["timestamp_micros"]) == [e["timestamp_micros"] for e in entries]


# ---------------------------------------------------------------------------
# Determinism and actor typing
# ---------------------------------------------------------------------------


def test_episode_seed_is_stable_across_processes():
    """``--resume`` promises the skipped slot is the one a full run would write.

    ``hash()`` on a str is salted per process, so it cannot back that promise.
    """
    assert episode_seed(0, "Town01", 7) == episode_seed(0, "Town01", 7)
    assert episode_seed(0, "Town01", 7) != episode_seed(0, "Town01", 8)
    assert episode_seed(0, "Town01", 7) != episode_seed(1, "Town01", 7)
    assert episode_seed(0, "Town01", 7) != episode_seed(0, "Town02", 7)
    assert episode_seed(0, "Town01", 7) == 1392679768

    # The split tag is what keeps validation from reproducing training: every
    # episode -- sky, Traffic Manager seed, spawn point -- follows from this
    # number, so the same seed in both splits would collect the same episodes.
    assert episode_seed(0, "Town01", 7, "training") != episode_seed(0, "Town01", 7, "validation")
    assert episode_seed(0, "Town01", 7, "training") == episode_seed(0, "Town01", 7, "training")


def test_actor_type_maps_two_wheelers_to_cyclist_and_ignores_sensors():
    def actor(type_id, **attributes):
        return SimpleNamespace(type_id=type_id, attributes=attributes)

    assert _agent_type(actor("vehicle.tesla.model3", number_of_wheels="4")) == VEHICLE
    assert _agent_type(actor("vehicle.bh.crossbike", number_of_wheels="2")) == 3
    assert _agent_type(actor("walker.pedestrian.0001")) == PEDESTRIAN
    assert _agent_type(actor("sensor.camera.rgb")) == 0
    assert _agent_type(actor("controller.ai.walker")) == 0
    assert _agent_type(actor("traffic.traffic_light")) == 0


def test_weather_sampling_stays_inside_carla_s_ranges_and_favours_daylight():
    import random

    altitudes = []
    for seed in range(400):
        weather = Weather.sample(random.Random(seed))
        assert 0.0 <= weather.cloudiness <= 100.0
        assert 0.0 <= weather.precipitation <= 100.0
        assert weather.precipitation_deposits <= max(weather.precipitation, 1e-6)
        assert 0.0 <= weather.fog_density <= 100.0
        assert 0.0 <= weather.sun_azimuth_angle <= 360.0
        # Never true night: the cameras would carry no signal while the abstract
        # render is unchanged, which is an unlearnable pair.
        assert weather.sun_altitude_angle >= -10.0
        altitudes.append(weather.sun_altitude_angle)
    assert np.mean(np.asarray(altitudes) > 20.0) > 0.7


def test_nearest_lane_distance_reads_the_ego_frame_road_array():
    from data_utils.carla_sim2real.collect import _nearest_lane_distance

    roads = np.asarray(
        [
            [3.0, 4.0, 4.0, 4.0, 4.5, float(ROAD_LANE)],  # 5 m from the origin
            [0.0, 1.0, 1.0, 1.0, 0.15, 5.0],  # a painted line, not a lane
        ],
        dtype=np.float32,
    )
    assert _nearest_lane_distance(roads) == pytest.approx(5.0)
    # A scene with no lane at all must not read as "on road".
    assert _nearest_lane_distance(roads[1:]) == math.inf


def test_episode_record_captures_the_sky_and_the_offroad_excursion():
    """Weather is the axis CARLA adds that a log cannot; it must survive to disk.

    It is applied per episode and would otherwise be lost the moment the frames
    are written, leaving no way to condition on it or ablate it later.
    """
    from data_utils.carla_sim2real.collect import episode_record

    episode = _synthetic_episode(frames=2)
    episode["weather"] = {"sun_altitude_angle": 35.0, "precipitation": 12.0}
    episode["ego_blueprint"] = "vehicle.lincoln.mkz_2020"
    episode["ego_length"], episode["ego_width"] = 4.892, 1.837
    episode["start_xy"] = (12.0, -3.0)
    # Frame roads sit 20 m ahead in _synthetic_episode, so the ego is off-lane.
    record = episode_record(episode, "Town01", seed=123)

    assert record["segment_id"] == "Town01_00000"
    assert record["town"] == "Town01" and record["seed"] == 123
    assert record["weather"]["sun_altitude_angle"] == 35.0
    assert record["ego_length"] == pytest.approx(4.892)
    assert record["frames"] == 2 and record["mean_agents"] == pytest.approx(1.0)
    assert math.isfinite(record["max_offroad"])
    assert record["start_xy"] == [12.0, -3.0]
    # Must be JSON-serialisable: it is written to episodes.jsonl verbatim.
    assert json.loads(json.dumps(record))["segment_id"] == "Town01_00000"


def test_kept_indices_thins_writes_without_touching_the_simulated_span():
    from data_utils.carla_sim2real.collect import kept_indices

    assert kept_indices(91, 1) == list(range(91))
    assert kept_indices(91, 10) == [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    assert kept_indices(91, 91) == [0]
    with pytest.raises(ValueError):
        kept_indices(1, 1)
    with pytest.raises(ValueError):
        kept_indices(91, 0)


def test_striding_never_changes_an_ego_value():
    """The ego table is built at 10 Hz and only then thinned.

    ``np.gradient`` is time-aware, so rebuilding from a thinned track would not
    rescale the derivatives -- it would *smooth* them. A brake pulse that lives
    between two kept frames is simply invisible at 1 s spacing, and obs[7] is
    exactly the slot the planning head reads to know the car is braking.
    """
    from data_utils.carla_sim2real.collect import kept_indices

    frames, stride = 91, 10
    times = (np.arange(frames) * 100_000).astype(np.int64)
    speed = np.full(frames, 8.0)
    speed[43:48] = 2.0  # a hard brake entirely between kept frames 40 and 50
    centers = np.stack([np.cumsum(speed * 0.1), np.zeros(frames)], axis=1)
    yaws = np.zeros(frames)

    full = episode_ego_obs(centers, yaws, times, length=4.9, width=2.0)
    thinned = full[kept_indices(frames, stride)]

    # The invariant the collector must hold: thinning is pure row selection.
    assert thinned.shape == (10, EGO_OBS_DIM)
    assert np.array_equal(thinned, full[::stride])

    # And the reason it must: rebuilding from the thinned track loses the brake.
    naive = episode_ego_obs(
        centers[::stride], yaws[::stride], times[::stride], length=4.9, width=2.0
    )
    # At 10 Hz the brake saturates obs[7]; at 1 s spacing only the residue in the
    # position integral survives, understating it more than threefold and
    # placing it on the wrong frame.
    assert abs(full[:, 7]).max() == pytest.approx(0.625, abs=1e-3)
    assert abs(naive[:, 7]).max() == pytest.approx(0.1875, abs=1e-3)
    assert abs(full[:, 7]).max() > 3 * abs(naive[:, 7]).max()


def test_deduplicate_drops_repeats_of_a_stopped_ego_but_keeps_the_stop():
    """A stopped ego must cost one scene, not --segments-per-spawn copies.

    Displacement accumulates from the last *kept* segment, so the window where
    the car arrives at the light survives and only its duplicates are dropped --
    stopped traffic is about a fifth of Waymo's training set and deleting it
    would bias the data away from exactly the case the goal term exercises.
    """
    from data_utils.carla_sim2real.collect import Collector

    def segments(points):
        return [{"segment_id": f"T_{i:05d}", "start_xy": p} for i, p in enumerate(points)]

    collector = Collector.__new__(Collector)  # no server needed for this method
    collector.town = "Town10HD"
    collector.args = SimpleNamespace(min_segment_displacement=25.0)

    # drives 70 m, then stops for two windows, then pulls away 40 m
    kept = collector._deduplicate(segments([(0.0, 0.0), (70.0, 0.0), (70.0, 0.0), (110.0, 0.0)]))
    assert [s["segment_id"] for s in kept] == ["T_00000", "T_00001", "T_00003"]

    # crawling in traffic: nothing after the first is a new scene
    kept = collector._deduplicate(segments([(0.0, 0.0), (5.0, 0.0), (9.0, 0.0), (12.0, 0.0)]))
    assert [s["segment_id"] for s in kept] == ["T_00000"]

    collector.args = SimpleNamespace(min_segment_displacement=0.0)
    assert len(collector._deduplicate(segments([(0.0, 0.0)] * 4))) == 4


def test_camera_render_window_covers_the_first_frame():
    """Which ticks the cameras must be rendering for, given stride and priming.

    Keyed on the distance to the *next* kept frame rather than the previous one.
    A condition written the other way round leaves step 0 unsubscribed, and the
    collector then blocks on an empty queue instead of capturing it.
    """

    def rendered(span, stride, prime):
        return [step for step in range(span) if (-step) % stride <= prime]

    assert rendered(30, 10, 0) == [0, 10, 20]
    assert rendered(30, 10, 2) == [0, 8, 9, 10, 18, 19, 20, 28, 29]
    # Every kept frame is always inside the render window, whatever the priming.
    for prime in range(4):
        window = set(rendered(91, 10, prime))
        assert set(kept_indices(91, 10)) <= window
    # No thinning means no gaps to prime across.
    assert rendered(5, 1, 0) == [0, 1, 2, 3, 4]
