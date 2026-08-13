from unittest.mock import patch

import pytest
import torch

from pufferlib.pufferl import load_config


@pytest.mark.parametrize(
    "env_name",
    (
        "puffer_drive_cam",
        "puffer_drive_3cam",
        "puffer_teddy_cam",
        "puffer_teddy_3cam",
        "puffer_giga_cam",
        "puffer_giga_3cam",
    ),
)
def test_camera_configs_enable_lane_area(env_name):
    with patch("sys.argv", ["pufferl.py"]):
        config = load_config(env_name)
    assert config["env"]["draw_lane_area"] is True
    assert config["env"]["lane_width"] == pytest.approx(4.5)


@pytest.mark.parametrize(
    "module_name",
    (
        "pufferlib.ocean.drive.raster_ref",
        "pufferlib.teddy.drive.raster_ref",
        "pufferlib.giga.drive.raster_ref",
    ),
)
def test_lane_area_and_edge_palette(module_name):
    module = __import__(module_name, fromlist=["raster_ref"])
    assert module.DEFAULT_PALETTE.road_color(module.RENDER_LANE_AREA) == (0.32, 0.32, 0.34)
    assert module.DEFAULT_PALETTE.road_color(module.RENDER_YELLOW_ROAD_EDGE) == (1.0, 0.82, 0.0)


@pytest.mark.parametrize(
    "module_name",
    (
        "pufferlib.ocean.drive.raster_ref",
        "pufferlib.teddy.drive.raster_ref",
        "pufferlib.giga.drive.raster_ref",
    ),
)
def test_lane_area_tag_switches_nonroad_ground_to_black(module_name):
    reference = __import__(module_name, fromlist=["raster_ref"])
    agents = torch.zeros((0, 8))
    egos = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    # Behind the camera: the strip itself is invisible, so any image difference
    # comes from the tag selecting the black non-road ground.
    tagged = torch.tensor([[-60.0, 0.0, -5.0, 0.0, 4.5, float(reference.RENDER_LANE_AREA)]])
    old_ground = reference.render(agents, torch.zeros((0, 6)), egos)
    black_ground = reference.render(agents, tagged, egos)
    assert black_ground[0, 0, :, -1, 0].sum() < old_ground[0, 0, :, -1, 0].sum()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
@pytest.mark.parametrize("package", ("ocean", "teddy", "giga"))
def test_cuda_matches_reference_for_lane_area_palette(package):
    reference = __import__(f"pufferlib.{package}.drive.raster_ref", fromlist=["raster_ref"])
    raster_cuda = __import__(f"pufferlib.{package}.drive.raster_cuda", fromlist=["raster_cuda"])

    # Markings/edges precede the opaque lane area, matching fill_render_roads().
    roads = torch.tensor(
        [
            [5.0, 2.25, 60.0, 2.25, 0.25, float(reference.RENDER_YELLOW_ROAD_EDGE)],
            [5.0, -2.25, 60.0, -2.25, 0.25, float(reference.RENDER_YELLOW_ROAD_EDGE)],
            [5.0, 0.0, 60.0, 0.0, 4.5, float(reference.RENDER_LANE_AREA)],
        ]
    )
    agents = torch.zeros((0, 8))
    egos = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    expected = reference.render(agents, roads, egos)
    actual = raster_cuda.render(agents.cuda(), roads.cuda(), egos.cuda()).cpu()
    assert (expected.int() - actual.int()).abs().max() <= 1


@pytest.mark.parametrize("package", ("ocean", "teddy", "giga"))
def test_lane_area_segments_overlap_at_joints(package):
    """`fill_render_roads` must extend each lane-area segment past its endpoints.

    The strips are drawn one per polyline segment. Meeting exactly at a joint
    leaves an uncovered wedge on the outside of every bend, and WOMD's centrelines
    are decimated enough (3.9 m between points at the median) for those wedges to
    show the ground through the drivable surface. Overlapping the strips instead
    covers the wedge, and bridges the gaps where a lane feature ends short of its
    successor.
    """
    import numpy as np

    drive_module = __import__(f"pufferlib.{package}.drive.drive", fromlist=["Drive"])
    reference_module = __import__(f"pufferlib.{package}.drive.raster_ref", fromlist=["raster_ref"])
    Drive = drive_module.Drive

    try:
        env = Drive(
            num_agents=8,
            num_maps=1,
            episode_length=10,
            obs_mode="render_state",
            draw_lane_area=True,
            lane_width=4.5,
            render_road_types=0,
        )
    except FileNotFoundError:
        pytest.skip("Drive map binaries are not available in this checkout")

    try:
        roads = env.render_state[0]["roads"][: env.render_state[0]["num_roads"]]
        lane_area = roads[roads[:, 5] == reference_module.RENDER_LANE_AREA]
        markings = roads[roads[:, 5] != reference_module.RENDER_LANE_AREA]
        assert len(lane_area), "no lane-area segments were emitted"
        lengths = np.hypot(lane_area[:, 2] - lane_area[:, 0], lane_area[:, 3] - lane_area[:, 1])
        # Each strip carries half a width of overlap at either end, so the shortest
        # one drawn spans at least a full strip width however short its segment is.
        assert lengths.min() >= 4.5 - 1e-3
        # Painted features are not widened; only the drivable surface is.
        if len(markings):
            marked = np.hypot(markings[:, 2] - markings[:, 0], markings[:, 3] - markings[:, 1])
            assert marked.min() < 4.5
    finally:
        env.close()
