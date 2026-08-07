"""Tests for the perspective rasterizer (Pictura reproduction).

These pin down the reference implementation in `pufferlib/ocean/drive/raster_ref.py`
against analytically predictable geometry. Once the CUDA kernel lands, it is
checked against the same reference here (see `test_cuda_matches_reference`).

Run as a script the way CI runs the other drive tests:
    python tests/test_raster.py
"""

import importlib.util
import math
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pufferlib.ocean.drive import raster_ref as R

EGO_AT_ORIGIN = torch.tensor([[0.0, 0.0, 1.0, 0.0]])  # at origin, facing +x
NO_AGENTS = torch.zeros((0, 8))
NO_ROADS = torch.zeros((0, 6))


def _car(x, y=0.0, heading=(1.0, 0.0), length=4.5, width=2.2, height=1.8):
    return torch.tensor([[x, y, heading[0], heading[1], length, width, height, float(R.VEHICLE)]])


def _lane_lines(offsets=(-1.75, 1.75), out_to=60, step=4, width=0.15):
    seg = []
    for y in offsets:
        for s in range(0, out_to, step):
            seg.append([float(s), y, float(s + step), y, width, float(R.ROAD_LINE)])
    return torch.tensor(seg)


# ---------------------------------------------------------------------------
# Camera rig
# ---------------------------------------------------------------------------


def test_nuplan_rig_matches_paper_table3():
    """Extrinsics and intrinsics must match Pictura Tab. 3 exactly."""
    expected = {
        "front": ((1.66, -0.01, 1.49), 0.0),
        "front_left": ((1.63, 0.12, 1.48), 55.0),
        "front_right": ((1.62, -0.16, 1.49), -55.0),
        "back": ((-0.47, 0.02, 1.43), 180.0),
    }
    assert len(R.NUPLAN_RIG) == 4
    for cam in R.NUPLAN_RIG:
        pos, yaw = expected[cam.name]
        assert cam.pos == pytest.approx(pos)
        assert cam.yaw_deg == pytest.approx(yaw)
        assert cam.pitch_deg == 0.0 and cam.roll_deg == 0.0
        # Shared intrinsics: 1545 px focal length on a 1920x1080 sensor,
        # rendered at 96x54, giving a 63.7 x 38.5 degree field of view.
        assert (cam.focal_px, cam.sensor_width) == (1545.0, 1920)
        assert (cam.width, cam.height) == (96, 54)
        hfov, vfov = cam.fov_deg()
        assert hfov == pytest.approx(63.7, abs=0.05)
        assert vfov == pytest.approx(38.5, abs=0.05)
        assert cam.intrinsics()[0] == pytest.approx(1545.0 * 96 / 1920)


def test_waymo_rig_matches_wod_calibration():
    """The default rig is Waymo's, as calibrated in Perception v1.4.3.

    Positions are the calibration medians moved from WOD's vehicle frame (rear
    axle, ground level) into the box-centre ego frame the rasterizer works in,
    which shifts x by `WAYMO_REAR_AXLE_TO_BOX_CENTER` and leaves y and z alone.
    """
    d = R.WAYMO_REAR_AXLE_TO_BOX_CENTER
    expected = {
        "front": ((1.5440 - d, -0.0237, 2.1157), 0.0),
        "front_left": ((1.4961 - d, 0.0946, 2.1155), 44.6),
        "front_right": ((1.4938 - d, -0.0963, 2.1157), -44.7),
    }
    assert len(R.WAYMO_RIG) == 3
    for cam in R.WAYMO_RIG:
        pos, yaw = expected[cam.name]
        assert cam.pos == pytest.approx(pos, abs=1e-4)
        assert cam.yaw_deg == pytest.approx(yaw)
        # Calibrated pitch and roll are under a degree on every camera, so the
        # rig carries mounting yaw only.
        assert cam.pitch_deg == 0.0 and cam.roll_deg == 0.0
        # Shared intrinsics: 2066.7 px focal length on a 1920x1280 sensor,
        # rendered at 96x64, giving a 49.8 x 34.4 degree field of view.
        assert (cam.focal_px, cam.sensor_width) == (2066.7, 1920)
        assert (cam.width, cam.height) == (96, 64)
        hfov, vfov = cam.fov_deg()
        assert hfov == pytest.approx(49.8, abs=0.05)
        assert vfov == pytest.approx(34.4, abs=0.05)
        assert cam.intrinsics()[0] == pytest.approx(2066.7 * 96 / 1920)
        # Square pixels, which only holds because 96x64 keeps the sensor's 3:2.
        assert cam.width / cam.height == pytest.approx(1920 / 1280)


def test_default_rig_is_the_waymo_front_camera_only():
    assert [c.name for c in R.DEFAULT_RIG] == ["front"]
    assert R.DEFAULT_RIG == R.WAYMO_RIG[:1]


def test_rotation_is_orthonormal_and_correctly_oriented():
    """Ego frame is x forward, y left, z up; camera frame is x right, y down, z forward."""
    for cam in R.NUPLAN_RIG:
        rot = cam.rotation()
        assert torch.allclose(rot @ rot.T, torch.eye(3), atol=1e-6)
        assert torch.linalg.det(rot) == pytest.approx(1.0, abs=1e-6)

    front = R.NUPLAN_RIG[0].rotation()
    assert torch.allclose(front[0], torch.tensor([0.0, -1.0, 0.0]), atol=1e-6)  # right = -y
    assert torch.allclose(front[1], torch.tensor([0.0, 0.0, -1.0]), atol=1e-6)  # down  = -z
    assert torch.allclose(front[2], torch.tensor([1.0, 0.0, 0.0]), atol=1e-6)  # fwd   = +x

    # Yawing left by 55 degrees swings the forward axis toward +y (left).
    fl = R.NUPLAN_RIG[1].rotation()
    assert fl[2][1] == pytest.approx(math.sin(math.radians(55.0)), abs=1e-6)

    back = R.NUPLAN_RIG[3].rotation()
    assert torch.allclose(back[2], torch.tensor([-1.0, 0.0, 0.0]), atol=1e-6)


def test_rig_tensor_layout():
    rig = R.rig_tensor(R.NUPLAN_RIG)
    assert rig.shape == (4, R.RIG_STRIDE)
    cam = R.NUPLAN_RIG[0]
    fx, fy, cx, cy = cam.intrinsics()
    assert torch.allclose(rig[0, :9], cam.rotation().reshape(-1), atol=1e-6)
    assert torch.allclose(rig[0, 9:12], torch.tensor(cam.pos), atol=1e-6)
    assert rig[0, 12].item() == pytest.approx(fx)
    assert rig[0, 16].item() == pytest.approx(cam.width)


def test_rig_from_config_json_and_presets():
    assert [c.name for c in R.rig_from_config("nuplan")] == [c.name for c in R.NUPLAN_RIG]
    assert [c.name for c in R.rig_from_config("waymo")] == [c.name for c in R.WAYMO_RIG]
    assert [c.name for c in R.rig_from_config(None)] == ["front"]
    spec = '[{"name": "front", "pos": [1.66, -0.01, 1.49], "yaw_deg": 0.0}]'
    rig = R.rig_from_config(spec)
    assert len(rig) == 1 and rig[0].pos == pytest.approx((1.66, -0.01, 1.49))


# ---------------------------------------------------------------------------
# Projection geometry
# ---------------------------------------------------------------------------


def _occupied_bbox(image, background):
    """Rows/cols where `image` differs from a background-only render."""
    diff = (image.int() - background.int()).abs().sum(1)[0, 0]
    ys, xs = torch.where(diff > 0)
    assert len(ys) > 0, "nothing was drawn"
    return int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())


@pytest.mark.parametrize("distance", [10.0, 20.0, 30.0])
def test_box_footprint_matches_pinhole_prediction(distance):
    """A box at a known range must land where the pinhole model says it does."""
    cam = R.DEFAULT_RIG[0]
    fx, fy, cx, cy = cam.intrinsics()
    cam_z, cam_x = cam.pos[2], cam.pos[0]

    length, width, height = 4.5, 2.2, 1.8
    img = R.render(_car(distance, length=length, width=width, height=height), NO_ROADS, EGO_AT_ORIGIN)
    bg = R.render(NO_AGENTS, NO_ROADS, EGO_AT_ORIGIN)
    r0, r1, c0, c1 = _occupied_bbox(img, bg)

    # Near and far faces of the box, in camera-relative depth.
    z_near = distance - length / 2 - cam_x
    z_far = distance + length / 2 - cam_x
    # The nearest face is the widest, so it sets the sides, and it is the closest
    # to the ground line, so it sets the base.
    assert c0 == pytest.approx(cx - fx * (width / 2) / z_near, abs=1.5)
    assert c1 == pytest.approx(cx + fx * (width / 2) / z_near, abs=1.5)
    assert r1 == pytest.approx(cy + fy * cam_z / z_near, abs=1.5)
    # Waymo's camera is mounted above a car's roof, so the top face is in view and
    # the silhouette's upper edge is the box's far upper edge, not its near one.
    assert cam_z > height
    assert r0 == pytest.approx(cy + fy * (cam_z - height) / z_far, abs=1.5)


def test_box_apparent_size_scales_inversely_with_range():
    bg = R.render(NO_AGENTS, NO_ROADS, EGO_AT_ORIGIN)
    widths = []
    for distance in (10.0, 20.0, 40.0):
        img = R.render(_car(distance), NO_ROADS, EGO_AT_ORIGIN)
        _, _, c0, c1 = _occupied_bbox(img, bg)
        widths.append(c1 - c0)
    assert widths[0] > widths[1] > widths[2]
    # Halving the range roughly doubles the apparent width.
    assert widths[0] / widths[1] == pytest.approx(2.0, rel=0.3)


def test_lateral_offset_maps_to_correct_side():
    """+y is left in the ego frame, and the camera's right axis is -y."""
    bg = R.render(NO_AGENTS, NO_ROADS, EGO_AT_ORIGIN)
    cx = R.DEFAULT_RIG[0].intrinsics()[2]

    left = R.render(_car(20.0, y=4.0), NO_ROADS, EGO_AT_ORIGIN)
    _, _, c0, c1 = _occupied_bbox(left, bg)
    assert (c0 + c1) / 2 < cx

    right = R.render(_car(20.0, y=-4.0), NO_ROADS, EGO_AT_ORIGIN)
    _, _, c0, c1 = _occupied_bbox(right, bg)
    assert (c0 + c1) / 2 > cx


def test_object_behind_camera_is_not_drawn():
    img = R.render(_car(-20.0), NO_ROADS, EGO_AT_ORIGIN)
    bg = R.render(NO_AGENTS, NO_ROADS, EGO_AT_ORIGIN)
    assert torch.equal(img, bg)


def test_ego_pose_is_applied():
    """Rotating the ego by 90 degrees moves a box from ahead to out of view."""
    ahead = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    turned = torch.tensor([[0.0, 0.0, 0.0, 1.0]])  # facing +y
    car = _car(20.0)
    bg = R.render(NO_AGENTS, NO_ROADS, ahead)
    assert not torch.equal(R.render(car, NO_ROADS, ahead), bg)
    assert torch.equal(R.render(car, NO_ROADS, turned), R.render(NO_AGENTS, NO_ROADS, turned))


# ---------------------------------------------------------------------------
# Visibility
# ---------------------------------------------------------------------------


def test_near_box_fully_occludes_far_box():
    """A nearer, larger silhouette must hide a smaller one directly behind it.

    Two things about the scene are deliberate. The occluder has to out-top the
    camera for full occlusion to be possible at all: from 2.12 m the rig looks
    over an ordinary 1.8 m car and the roof of anything behind it stays visible
    however close the near one is, so the occluder here is a 3 m box.

    And the check is "hidden", not "bit-identical". Each box face is two triangles
    sharing a diagonal, and their analytic coverage does not sum to one along it,
    so a seam one pixel wide runs across every face and lets a few levels of
    whatever is behind bleed through. That is a property of the coverage-as-alpha
    scheme, not of this scene: the seam is visible on a lone box against the sky
    at either rig, and it is only chance that nothing sat behind it before.
    """
    near = _car(10.0, height=3.0)
    both = torch.cat([near, _car(30.0)], dim=0)
    only_near = R.render(near, NO_ROADS, EGO_AT_ORIGIN).int()
    with_far = R.render(both, NO_ROADS, EGO_AT_ORIGIN).int()

    diff = (only_near - with_far).abs().sum(2)[0, 0]
    assert (diff > 0).sum() <= 4, "the far box is showing through, not just the seam"
    assert diff.max() <= 8, "the leak is too strong to be the coverage seam"


def test_ground_plane_occludes_distant_boxes():
    """The ground intercept acts as the depth bound for a flat world.

    A box farther than the ground point seen through a pixel cannot be visible in
    that direction, so a distant box must not bleed into the near foreground.
    """
    cam = R.DEFAULT_RIG[0]
    fx, fy, cx, cy = cam.intrinsics()
    img = R.render(_car(60.0), NO_ROADS, EGO_AT_ORIGIN)
    bg = R.render(NO_AGENTS, NO_ROADS, EGO_AT_ORIGIN)
    r0, r1, _, _ = _occupied_bbox(img, bg)
    # Ground depth equals 60 m at this row; nothing from that box may appear below it.
    horizon_row_for_60m = cy + fy * cam.pos[2] / (60.0 - cam.pos[0])
    assert r1 <= horizon_row_for_60m + 1.5


def test_coverage_stays_inside_triangle_bounding_box():
    """Regression: sliver triangles must not leak coverage across the image.

    When a triangle is a fraction of a pixel thick its two long edges are nearly
    collinear. Dilating both by the half pixel the antialiasing ramp uses makes
    their half-planes overlap in a narrow wedge reaching far past the shared
    vertex, which previously painted stray pixels tens of columns away.
    """
    cam = R.DEFAULT_RIG[0]
    fx, fy, cx, cy = cam.intrinsics()
    rot, pos = cam.rotation(), torch.tensor(cam.pos)

    tris, _ = R._agent_triangles(_car(30.0), EGO_AT_ORIGIN[0], R.DEFAULT_PALETTE)
    screen, depth = R._project(R._to_camera(tris, rot, pos), fx, fy, cx, cy)

    ys, xs = torch.meshgrid(
        torch.arange(cam.height) + 0.5, torch.arange(cam.width) + 0.5, indexing="ij"
    )
    cov, _ = R._coverage_and_depth(screen, depth, xs.reshape(1, -1), ys.reshape(1, -1))

    for i in range(tris.shape[0]):
        hit = torch.where(cov[i] > 0)[0]
        if len(hit) == 0:
            continue
        rows, cols = hit // cam.width, hit % cam.width
        assert cols.min() >= screen[i, :, 0].min() - 1.0
        assert cols.max() <= screen[i, :, 0].max() + 1.0
        assert rows.min() >= screen[i, :, 1].min() - 1.0
        assert rows.max() <= screen[i, :, 1].max() + 1.0


def test_opaque_fragments_are_not_dropped():
    """Regression: a fully covered pixel must take the primitive's colour.

    Computing transmittance as cumprod / (1 - alpha) divides by zero exactly
    where a fragment is opaque, which silently erased every box interior.
    """
    img = R.render(_car(12.0), NO_ROADS, EGO_AT_ORIGIN)[0, 0]
    bg = R.render(NO_AGENTS, NO_ROADS, EGO_AT_ORIGIN)[0, 0]
    changed = (img.int() - bg.int()).abs().sum(0) > 0
    # The silhouette must be a solid block, not just an outline.
    assert changed.sum() > 200
    rows, cols = torch.where(changed)
    filled = changed[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]
    assert filled.float().mean() > 0.9


# ---------------------------------------------------------------------------
# Antialiasing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", [(32, 21), (64, 42), (96, 64), (192, 128)])
def test_thin_markings_survive_low_resolution(size):
    """Analytic coverage is what keeps 0.15 m lane lines intact at 96x64.

    Without it, a marking thinner than a pixel either snaps to a full pixel or
    disappears depending on where the sample lands. The paper calls this out as
    the reason training can run at low resolution at all (Fig. 10).
    """
    width, height = size
    cam = R.Camera("front", R.DEFAULT_RIG[0].pos, width=width, height=height)
    img = R.render(NO_AGENTS, _lane_lines(), EGO_AT_ORIGIN, cameras=[cam])
    bg = R.render(NO_AGENTS, NO_ROADS, EGO_AT_ORIGIN, cameras=[cam])
    painted = (img.int() - bg.int()).abs().sum(1)[0, 0] > 0
    assert painted.sum() > 0, "lane markings vanished entirely"
    # Markings belong on the road surface, below the horizon.
    rows = torch.where(painted)[0]
    assert rows.min() >= height // 2 - 1


def test_marking_coverage_is_fractional_not_binary():
    """Sub-pixel markings should produce intermediate values, not hard on/off."""
    img = R.render(NO_AGENTS, _lane_lines(), EGO_AT_ORIGIN)[0, 0]
    bg = R.render(NO_AGENTS, NO_ROADS, EGO_AT_ORIGIN)[0, 0]
    painted = (img.int() - bg.int()).abs().sum(0)
    values = painted[painted > 0]
    assert values.float().std() > 1.0, "coverage looks binary; antialiasing is not active"


def test_near_markings_are_thicker_than_far_ones():
    """Perspective alone tapers markings with depth (paper Fig. 3i)."""
    img = R.render(NO_AGENTS, _lane_lines(), EGO_AT_ORIGIN)[0, 0]
    bg = R.render(NO_AGENTS, NO_ROADS, EGO_AT_ORIGIN)[0, 0]
    painted = (img.int() - bg.int()).abs().sum(0) > 0
    height = painted.shape[0]
    near = painted[int(height * 0.85) :].sum()
    far = painted[height // 2 : int(height * 0.6)].sum()
    assert near > far


# ---------------------------------------------------------------------------
# Contract
# ---------------------------------------------------------------------------


def test_output_shape_dtype_and_determinism():
    egos = torch.tensor([[0.0, 0.0, 1.0, 0.0], [5.0, 2.0, 0.0, 1.0]])
    agents = torch.cat([_car(15.0), _car(25.0, y=3.0)], dim=0)
    roads = _lane_lines()

    img = R.render(agents, roads, egos, cameras=R.NUPLAN_RIG)
    assert img.shape == (2, 4, 3, 54, 96)
    assert img.dtype == torch.uint8
    assert torch.equal(img, R.render(agents, roads, egos, cameras=R.NUPLAN_RIG))


def test_empty_scene_is_background_only():
    img = R.render(NO_AGENTS, NO_ROADS, EGO_AT_ORIGIN)[0, 0]
    # Sky above the horizon, asphalt below, each uniform across a row.
    assert img[:, 5, :].float().std(dim=-1).max() < 1.0
    assert img[:, 45, :].float().std(dim=-1).max() < 1.0
    assert not torch.equal(img[:, 5, 0], img[:, 45, 0])


def test_cameras_in_a_rig_must_share_resolution():
    rig = [R.Camera("a", (0, 0, 1.5), width=96, height=54), R.Camera("b", (0, 0, 1.5), width=64, height=36)]
    with pytest.raises(ValueError, match="share a resolution"):
        R.render(NO_AGENTS, NO_ROADS, EGO_AT_ORIGIN, cameras=rig)


def _cuda_rasterizer_available():
    # `torch.ops` is a lazy namespace and answers hasattr for any name, so probe
    # for the module itself instead.
    return torch.cuda.is_available() and importlib.util.find_spec("pufferlib.ocean.drive.raster_cuda") is not None


@pytest.mark.skipif(not _cuda_rasterizer_available(), reason="CUDA rasterizer not built yet (Phase 3)")
def test_cuda_matches_reference():
    """Acceptance gate for the CUDA kernel: within one uint8 level of reference."""
    agents = torch.cat([_car(8.0 + 4 * i, y=float(i - 2)) for i in range(5)], dim=0)
    roads = _lane_lines()
    egos = torch.tensor([[0.0, 0.0, 1.0, 0.0]])

    want = R.render(agents, roads, egos)
    from pufferlib.ocean.drive import raster_cuda

    got = raster_cuda.render(agents.cuda(), roads.cuda(), egos.cuda()).cpu()
    assert (want.int() - got.int()).abs().max() <= 1


def _dense_scene(seed=0):
    """A scene dense enough to crowd the horizon with distant markings.

    Sparse test scenes hide the failure modes that matter: many far segments
    compress into the couple of pixel rows below the horizon, where fragment
    retention and depth-tie ordering decide the result.
    """
    g = torch.Generator().manual_seed(seed)
    seg = []
    for lane in range(-3, 4):
        y = lane * 1.75
        for s in range(0, 200, 3):
            seg.append([float(s), y, float(s + 3), y, 0.15, float(R.ROAD_LINE)])
    for cross in range(20, 200, 20):
        seg.append([float(cross), -8.0, float(cross), 8.0, 0.25, float(R.ROAD_EDGE)])
    roads = torch.tensor(seg)

    n = 24
    xs = torch.rand(n, generator=g) * 90 + 6
    ys = (torch.rand(n, generator=g) - 0.5) * 12
    th = (torch.rand(n, generator=g) - 0.5) * 0.6
    types = torch.tensor([R.VEHICLE, R.PEDESTRIAN, R.CYCLIST])[
        torch.randint(0, 3, (n,), generator=g)
    ].float()
    agents = torch.stack(
        [xs, ys, torch.cos(th), torch.sin(th),
         torch.full((n,), 4.5), torch.full((n,), 2.0), torch.full((n,), 1.7), types],
        dim=-1,
    )
    return agents, roads


@pytest.mark.skipif(not _cuda_rasterizer_available(), reason="CUDA rasterizer not built yet (Phase 3)")
@pytest.mark.parametrize("rig_name", ["front", "waymo", "nuplan"])
def test_cuda_matches_reference_dense_scene(rig_name):
    """Parity on a crowded scene, across every multi-camera rig."""
    from pufferlib.ocean.drive import raster_cuda

    agents, roads = _dense_scene()
    egos = torch.tensor([[0.0, 0.0, 1.0, 0.0, -1.0], [40.0, 1.0, 0.94, 0.34, 3.0]])
    rig = R.rig_from_config(rig_name)

    want = R.render(agents, roads, egos, cameras=rig).int()
    got = raster_cuda.render(agents.cuda(), roads.cuda(), egos.cuda(), cameras=rig).cpu().int()

    diff = (want - got).abs()
    # A handful of pixels differ by 2 where float32 rounding flips an edge coverage
    # ramp; anything larger means a fragment is present in one and not the other.
    assert diff.max() <= 2, f"max diff {int(diff.max())}"
    assert (diff > 1).sum() <= diff.numel() // 1000


@pytest.mark.skipif(not _cuda_rasterizer_available(), reason="CUDA rasterizer not built yet (Phase 3)")
def test_cuda_is_deterministic():
    """Fixed scratch slots plus index tie-breaking make the output reproducible."""
    from pufferlib.ocean.drive import raster_cuda

    agents, roads = _dense_scene(seed=1)
    egos = torch.tensor([[0.0, 0.0, 1.0, 0.0, -1.0]])
    first = raster_cuda.render(agents.cuda(), roads.cuda(), egos.cuda(), cameras=R.NUPLAN_RIG)
    for _ in range(4):
        again = raster_cuda.render(agents.cuda(), roads.cuda(), egos.cuda(), cameras=R.NUPLAN_RIG)
        assert torch.equal(first, again)


@pytest.mark.skipif(not _cuda_rasterizer_available(), reason="CUDA rasterizer not built yet (Phase 3)")
def test_cuda_skips_the_ego_own_box():
    """The fifth ego column removes the ego's own primitive from its own view."""
    from pufferlib.ocean.drive import raster_cuda

    agents, roads = _dense_scene()
    # Put a box exactly at the ego, which would otherwise fill the frame.
    own = torch.tensor([[0.0, 0.0, 1.0, 0.0, 4.5, 2.0, 1.7, float(R.VEHICLE)]])
    agents = torch.cat([own, agents], dim=0)

    skipped = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0]])  # self_index = 0
    kept = torch.tensor([[0.0, 0.0, 1.0, 0.0, -1.0]])
    a, r = agents.cuda(), roads.cuda()
    assert not torch.equal(raster_cuda.render(a, r, skipped.cuda()), raster_cuda.render(a, r, kept.cuda()))
    # With the box skipped, the view must match a scene that never had it.
    without = raster_cuda.render(agents[1:].cuda(), r, torch.tensor([[0.0, 0.0, 1.0, 0.0, -1.0]]).cuda())
    assert torch.equal(raster_cuda.render(a, r, skipped.cuda()), without)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
