from unittest.mock import patch

import pytest
import torch

from pufferlib.pufferl import load_config


@pytest.mark.parametrize(
    "env_name",
    ("puffer_teddy_cam", "puffer_teddy_3cam", "puffer_giga_cam", "puffer_giga_3cam"),
)
def test_camera_configs_enable_lane_area(env_name):
    with patch("sys.argv", ["pufferl.py"]):
        config = load_config(env_name)
    assert config["env"]["draw_lane_area"] is True
    assert config["env"]["lane_width"] == pytest.approx(4.5)


@pytest.mark.parametrize(
    "module_name",
    ("pufferlib.teddy.drive.raster_ref", "pufferlib.giga.drive.raster_ref"),
)
def test_lane_area_and_edge_palette(module_name):
    module = __import__(module_name, fromlist=["raster_ref"])
    assert module.DEFAULT_PALETTE.road_color(module.RENDER_LANE_AREA) == (0.32, 0.32, 0.34)
    assert module.DEFAULT_PALETTE.road_color(module.RENDER_YELLOW_ROAD_EDGE) == (1.0, 0.82, 0.0)


def test_lane_area_tag_switches_nonroad_ground_to_black():
    from pufferlib.teddy.drive import raster_ref as reference

    agents = torch.zeros((0, 8))
    egos = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    # Behind the camera: the strip itself is invisible, so any image difference
    # comes from the tag selecting the black non-road ground.
    tagged = torch.tensor(
        [[-60.0, 0.0, -5.0, 0.0, 4.5, float(reference.RENDER_LANE_AREA)]]
    )
    old_ground = reference.render(agents, torch.zeros((0, 6)), egos)
    black_ground = reference.render(agents, tagged, egos)
    assert black_ground[0, 0, :, -1, 0].sum() < old_ground[0, 0, :, -1, 0].sum()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
def test_cuda_matches_reference_for_lane_area_palette():
    from pufferlib.teddy.drive import raster_cuda
    from pufferlib.teddy.drive import raster_ref as reference

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
