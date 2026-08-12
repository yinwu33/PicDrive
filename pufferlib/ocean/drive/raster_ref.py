"""Reference perspective rasterizer for PufferDrive (Pictura reproduction).

Pure PyTorch, batched over egos and cameras. This module is the *semantic
definition* of what the perspective renderer draws; the CUDA kernel must match
its output within a tolerance of one uint8 level. It is deliberately written for
clarity rather than speed -- it exists to be obviously correct and to serve as
ground truth in `tests/test_raster.py`.

Conventions
-----------
Ego frame: x forward, y left, z up, metres. This matches the simulator, where
`heading_x = cos(theta)`, `heading_y = sin(theta)` and world->ego rotation is
`ex = dx*cos + dy*sin`, `ey = -dx*sin + dy*cos` (see `compute_observations` in
drive.h). Camera yaw is about +z, so positive yaw turns left.

Camera frame: the usual computer-vision convention, +x right, +y down, +z
forward, so a point projects as `u = fx*X/Z + cx`, `v = fy*Y/Z + cy`.

Scene layering
--------------
The world is flat, which lets the renderer resolve visibility in three layers
instead of sorting every triangle:

1. Background -- an analytic ray/ground-plane intersection gives each pixel
   either the asphalt colour with an exact depth, or sky with infinite depth.
2. Road markings -- paint lying *on* the ground plane, composited over the
   background. They are coplanar with it, so they are drawn on top rather than
   depth-tested against it, which avoids z-fighting entirely.
3. Agents -- boxes standing above the ground. These carry real depth and are
   composited over layers 1-2 wherever the agent is nearer than the ground point
   seen through that pixel. That test is exact for a flat world: a car 30 m away
   cannot be visible in a direction whose ground intercept is 5 m away.

Within a layer, fragments are combined with front-to-back alpha compositing
using analytic edge coverage as alpha, which is the antialiasing scheme Pictura
relies on to keep thin lane markings intact at 96x54 (paper Fig. 10).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import torch

# Entity type ids, mirroring the #defines in drive.h.
VEHICLE = 1
PEDESTRIAN = 2
CYCLIST = 3
ROAD_LANE = 4
ROAD_LINE = 5
ROAD_EDGE = 6
STOP_SIGN = 7
CROSSWALK = 8
SPEED_BUMP = 9
DRIVEWAY = 10

# Feature widths of the RenderState buffers, mirroring drive.h.
RENDER_AGENT_FEATURES = 8
RENDER_ROAD_FEATURES = 6
RENDER_EGO_FEATURES = 5

# Number of floats per camera in the packed rig tensor the CUDA kernel reads.
RIG_STRIDE = 20

# Fragments retained per pixel per layer, nearest first. Front-to-back compositing
# stops mattering once alpha saturates, so this only bites where many partially
# covering fragments stack up -- in practice the band of distant road markings
# that crowds into the couple of pixel rows just below the horizon. It is part of
# the renderer's definition, not an implementation detail: the CUDA kernel keeps
# the same number so the two produce identical images.
MAX_FRAGMENTS = 16

# Relative slack on the agent layer's ground-intercept bound; see
# `_composite_shaded`. The CUDA kernel uses the same figure.
GROUND_DEPTH_SLACK = 1e-4

# Depth-ordering quantum, in metres. Two faces of a box meet along an edge where
# their depths are equal in exact arithmetic, so ordering the fragments on the raw
# float decides which face shades that edge by rounding -- differently on the CPU
# and in the kernel. Rounding to a millimetre lets the primitive index settle it
# instead, which both implementations agree on, and is far below any depth
# difference the geometry means. The CUDA kernel uses the same figure.
DEPTH_ORDER_QUANTUM = 1e-3


# ---------------------------------------------------------------------------
# Camera rig
# ---------------------------------------------------------------------------


@dataclass
class Camera:
    """One ego-mounted pinhole camera.

    Position is in the ego frame in metres; yaw/pitch/roll are in degrees, with
    positive yaw turning left and positive pitch tilting the view downward.
    """

    name: str
    pos: tuple[float, float, float]
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0
    roll_deg: float = 0.0
    width: int = 96
    height: int = 64
    # Sensor focal length in pixels at `sensor_width`. Both rigs below are
    # specified this way; the field of view quoted alongside them is the derived
    # quantity, so deriving fx from the focal length rather than from a rounded
    # angle reproduces the sensor exactly. Defaults are Waymo's front camera.
    focal_px: float = 2066.7
    sensor_width: int = 1920
    hfov_deg: float | None = None  # overrides focal_px when set
    near: float = 0.15
    far: float = 200.0

    def intrinsics(self) -> tuple[float, float, float, float]:
        """Return (fx, fy, cx, cy) for square pixels at this camera's resolution."""
        if self.hfov_deg is not None:
            fx = (self.width / 2.0) / math.tan(math.radians(self.hfov_deg) / 2.0)
        else:
            fx = self.focal_px * self.width / self.sensor_width
        return fx, fx, self.width / 2.0, self.height / 2.0

    def fov_deg(self) -> tuple[float, float]:
        """Horizontal and vertical field of view, for reporting."""
        fx, fy, cx, cy = self.intrinsics()
        return (
            2.0 * math.degrees(math.atan(cx / fx)),
            2.0 * math.degrees(math.atan(cy / fy)),
        )

    def rotation(self) -> torch.Tensor:
        """Ego->camera rotation, rows being the camera's right/down/forward axes."""
        yaw = math.radians(self.yaw_deg)
        pitch = math.radians(self.pitch_deg)
        roll = math.radians(self.roll_deg)

        # Forward, then the right axis before roll (horizontal, perpendicular to
        # forward). For yaw=0 this gives forward=+x_ego and right=-y_ego, since
        # +y_ego points left.
        fwd = (math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), -math.sin(pitch))
        right0 = (math.sin(yaw), -math.cos(yaw), 0.0)
        down0 = (
            fwd[1] * right0[2] - fwd[2] * right0[1],
            fwd[2] * right0[0] - fwd[0] * right0[2],
            fwd[0] * right0[1] - fwd[1] * right0[0],
        )
        cr, sr = math.cos(roll), math.sin(roll)
        right = tuple(right0[i] * cr + down0[i] * sr for i in range(3))
        down = tuple(-right0[i] * sr + down0[i] * cr for i in range(3))
        return torch.tensor([right, down, fwd], dtype=torch.float32)


# Distance from the rear axle to the centre of the ego bounding box, for the
# Waymo platform (Chrysler Pacifica: 5.18 m long, 1.15 m rear overhang). WOD
# reports extrinsics in a vehicle frame whose origin is the rear axle at ground
# level; this rasterizer works in an ego frame centred on the bounding box,
# because that is what WOMD stores and what `Drive` keeps in `agent->x/y`. z is
# already measured from the road surface in both, so only x shifts.
WAYMO_REAR_AXLE_TO_BOX_CENTER = 1.44


def _box_center_frame(pos: tuple[float, float, float]) -> tuple[float, float, float]:
    """WOD vehicle frame (rear axle, ground level) -> the box-centre ego frame."""
    return (round(pos[0] - WAYMO_REAR_AXLE_TO_BOX_CENTER, 4), pos[1], pos[2])


# Waymo's own rig, medians of the camera calibration in 36 Perception v1.4.3
# segments; the positions below are passed in as calibrated. Shared
# intrinsics: 1920x1280 sensor, 2066.7 px focal length, 49.8 x 34.4 degree FOV,
# rendered at 96x64 -- the sensor's own 3:2, so pixels stay square.
#
# Measured pitch and roll are within a degree of zero on every camera (front is
# 0.08 and -0.17) and the front camera's yaw is 0.14, all of it calibration
# jitter rather than mounting, so only the mounting yaw of the side pair is kept.
#
# The two side cameras (yaw +-90) are left out: they are 1920x886, and a rig has
# to share one resolution. WOD has no rear camera, so unlike NUPLAN_RIG this one
# cannot see behind the vehicle.
WAYMO_RIG: list[Camera] = [
    Camera("front", _box_center_frame((1.5440, -0.0237, 2.1157)), yaw_deg=0.0),
    Camera("front_left", _box_center_frame((1.4961, 0.0946, 2.1155)), yaw_deg=44.6),
    Camera("front_right", _box_center_frame((1.4938, -0.0963, 2.1157)), yaw_deg=-44.7),
]

# The four-camera rig of the Pictura paper (Tab. 3), modelled on nuPlan. Shared
# intrinsics: 1920x1080 sensor, 1545 px focal length, 63.7 x 38.5 degree FOV,
# rendered at 96x54. Only yaw varies between cameras; there is no pitch or roll.
# Positions are as published, i.e. in nuPlan's own rear-axle frame; they are not
# shifted to box centre, so this rig reproduces the paper rather than the sim.
NUPLAN_RIG: list[Camera] = [
    Camera("front", (1.66, -0.01, 1.49), yaw_deg=0.0, focal_px=1545.0, width=96, height=54),
    Camera("front_left", (1.63, 0.12, 1.48), yaw_deg=55.0, focal_px=1545.0, width=96, height=54),
    Camera("front_right", (1.62, -0.16, 1.49), yaw_deg=-55.0, focal_px=1545.0, width=96, height=54),
    Camera("back", (-0.47, 0.02, 1.43), yaw_deg=180.0, focal_px=1545.0, width=96, height=54),
]

# Default rig: Waymo's front camera alone. PufferDrive drives WOMD layouts, so
# rendering them through the sensor that recorded them is what keeps a later
# sim-to-real alignment against Waymo camera data from having to absorb a change
# of projection on top of the appearance gap. Adding the other two is a config
# change, not a kernel change.
DEFAULT_RIG: list[Camera] = WAYMO_RIG[:1]


def rig_from_config(spec) -> list[Camera]:
    """Build a rig from a JSON string or a list of dicts.

    Accepts the form written in drive_cam.ini, e.g.
        [{"name": "front", "pos": [1.66, -0.01, 1.49], "yaw_deg": 0.0}]
    Named presets "waymo" (all three Waymo cameras), "nuplan" (Pictura's four)
    and "front" (the default, Waymo's front camera alone) are also accepted.
    """
    if spec is None:
        return list(DEFAULT_RIG)
    if isinstance(spec, str):
        text = spec.strip().strip('"').strip("'")
        if text.lower() == "waymo":
            return list(WAYMO_RIG)
        if text.lower() == "nuplan":
            return list(NUPLAN_RIG)
        if text.lower() == "front":
            return list(DEFAULT_RIG)
        spec = json.loads(text)
    if isinstance(spec, dict):
        spec = [spec]
    cameras = []
    for entry in spec:
        entry = dict(entry)
        pos = tuple(float(v) for v in entry.pop("pos"))
        cameras.append(Camera(pos=pos, **entry))
    return cameras


def rig_tensor(cameras: list[Camera], device="cpu") -> torch.Tensor:
    """Pack a rig into the flat [num_cameras, RIG_STRIDE] layout the kernel reads.

    Layout: R (9, row-major ego->camera), pos (3), fx, fy, cx, cy, width,
    height, near, far.
    """
    rows = []
    for cam in cameras:
        fx, fy, cx, cy = cam.intrinsics()
        rows.append(
            torch.cat(
                [
                    cam.rotation().reshape(-1),
                    torch.tensor(cam.pos, dtype=torch.float32),
                    torch.tensor(
                        [fx, fy, cx, cy, cam.width, cam.height, cam.near, cam.far],
                        dtype=torch.float32,
                    ),
                ]
            )
        )
    return torch.stack(rows).to(device)


# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------


@dataclass
class Palette:
    """Flat-shaded colours. No textures, no lighting, as in the paper.

    Agent boxes are coloured per face so that orientation reads from colour
    alone, and modulated by depth so that range does too.
    """

    sky: tuple[float, float, float] = (0.62, 0.70, 0.80)
    ground: tuple[float, float, float] = (0.16, 0.16, 0.17)

    agent: dict[int, tuple[float, float, float]] = field(
        default_factory=lambda: {
            VEHICLE: (0.90, 0.26, 0.24),
            PEDESTRIAN: (0.30, 0.85, 0.40),
            CYCLIST: (0.98, 0.78, 0.20),
        }
    )
    # Per-face multipliers, indexed by the face order used in `_agent_triangles`:
    # front, back, left, right, top, bottom. These are not a light model -- they
    # rotate with the object -- but a code that lets the policy read another
    # agent's heading off its faces, so all six only have to stay distinguishable
    # from one another.
    #
    # `back` is the face you look at for the whole of car-following, which makes it
    # the most-seen face in the rig; at 0.42 a red vehicle's rear rendered darker
    # than the asphalt it stood on and its class colour was gone. It sits at 0.72
    # rather than lower because 0.6 would collide with `right`. The tightest
    # remaining pair is back/right at 0.10, wider than the front/top pair this
    # palette already shipped with.
    face_shade: tuple[float, ...] = (1.00, 0.72, 0.86, 0.62, 0.95, 0.32)

    road: dict[int, tuple[float, float, float]] = field(
        default_factory=lambda: {
            ROAD_LANE: (0.55, 0.55, 0.55),
            ROAD_LINE: (0.95, 0.95, 0.95),
            ROAD_EDGE: (0.60, 0.60, 0.64),
            CROSSWALK: (0.92, 0.92, 0.92),
            SPEED_BUMP: (0.95, 0.80, 0.25),
            DRIVEWAY: (0.45, 0.45, 0.48),
        }
    )

    # Aerial perspective: every surface blends toward `sky` with distance, from no
    # blend at the camera to `1 - depth_min_scale` at `depth_falloff` metres. See
    # `_haze`.
    depth_falloff: float = 60.0
    depth_min_scale: float = 0.30

    def agent_color(self, type_id: int) -> tuple[float, float, float]:
        return self.agent.get(int(type_id), self.agent[VEHICLE])

    def road_color(self, type_id: int) -> tuple[float, float, float]:
        return self.road.get(int(type_id), self.road[ROAD_LINE])


DEFAULT_PALETTE = Palette()


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def world_to_ego(points_xy: torch.Tensor, ego: torch.Tensor) -> torch.Tensor:
    """Rotate world-frame XY into the ego frame of a single ego.

    `points_xy` is [..., 2]; `ego` is [x, y, cos_h, sin_h].
    """
    dx = points_xy[..., 0] - ego[0]
    dy = points_xy[..., 1] - ego[1]
    cos_h, sin_h = ego[2], ego[3]
    return torch.stack([dx * cos_h + dy * sin_h, -dx * sin_h + dy * cos_h], dim=-1)


def _road_triangles(roads: torch.Tensor, ego: torch.Tensor, palette: Palette):
    """Expand road segments into ground-plane quads, two triangles each.

    `roads` is [R, 6] = x0, y0, x1, y1, width, type in world frame.
    Returns (verts [T, 3, 3] in the ego frame, colors [T, 3]).
    """
    if roads.numel() == 0:
        return roads.new_zeros((0, 3, 3)), roads.new_zeros((0, 3))

    p0 = world_to_ego(roads[:, 0:2], ego)
    p1 = world_to_ego(roads[:, 2:4], ego)
    half_w = (roads[:, 4] * 0.5).unsqueeze(-1)

    direction = p1 - p0
    length = torch.linalg.norm(direction, dim=-1, keepdim=True).clamp_min(1e-6)
    direction = direction / length
    # Left normal of the segment direction.
    normal = torch.stack([-direction[:, 1], direction[:, 0]], dim=-1)

    a = p0 + normal * half_w
    b = p0 - normal * half_w
    c = p1 - normal * half_w
    d = p1 + normal * half_w

    def lift(xy):
        return torch.cat([xy, xy.new_zeros((xy.shape[0], 1))], dim=-1)

    a, b, c, d = lift(a), lift(b), lift(c), lift(d)
    tris = torch.cat([torch.stack([a, b, c], dim=1), torch.stack([a, c, d], dim=1)], dim=0)

    base = torch.tensor(
        [palette.road_color(int(t)) for t in roads[:, 5].tolist()],
        dtype=roads.dtype,
        device=roads.device,
    )
    return tris, torch.cat([base, base], dim=0)


# Cuboid faces as indices into the 8 corners produced by `_agent_triangles`.
# Corner order is (front/back, left/right, bottom/top) with bit weights 4/2/1.
_BOX_FACES = (
    ((4, 6, 7), (4, 7, 5)),  # front  (+x local)
    ((2, 0, 1), (2, 1, 3)),  # back   (-x local)
    ((6, 2, 3), (6, 3, 7)),  # left   (+y local)
    ((0, 4, 5), (0, 5, 1)),  # right  (-y local)
    ((1, 5, 7), (1, 7, 3)),  # top    (+z)
    ((0, 2, 6), (0, 6, 4)),  # bottom (z = 0)
)


def _agent_triangles(agents: torch.Tensor, ego: torch.Tensor, palette: Palette):
    """Expand agent boxes into cuboids, twelve triangles each.

    `agents` is [A, 8] = x, y, cos_h, sin_h, length, width, height, type in
    world frame. Returns (verts [T, 3, 3] in the ego frame, colors [T, 3]).
    """
    if agents.numel() == 0:
        return agents.new_zeros((0, 3, 3)), agents.new_zeros((0, 3))

    center = world_to_ego(agents[:, 0:2], ego)
    # Relative heading of the agent in the ego frame.
    cos_r = agents[:, 2] * ego[2] + agents[:, 3] * ego[3]
    sin_r = agents[:, 3] * ego[2] - agents[:, 2] * ego[3]

    half_l = agents[:, 4] * 0.5
    half_w = agents[:, 5] * 0.5
    height = agents[:, 6]

    corners = []
    for fx_sign in (-1.0, 1.0):  # back, front
        for fy_sign in (-1.0, 1.0):  # right, left
            for fz in (0.0, 1.0):  # bottom, top
                lx = fx_sign * half_l
                ly = fy_sign * half_w
                corners.append(
                    torch.stack(
                        [
                            center[:, 0] + lx * cos_r - ly * sin_r,
                            center[:, 1] + lx * sin_r + ly * cos_r,
                            height * fz,
                        ],
                        dim=-1,
                    )
                )
    corners = torch.stack(corners, dim=1)  # [A, 8, 3]

    base = torch.tensor(
        [palette.agent_color(int(t)) for t in agents[:, 7].tolist()],
        dtype=agents.dtype,
        device=agents.device,
    )

    tris, cols = [], []
    for face_idx, face in enumerate(_BOX_FACES):
        shade = palette.face_shade[face_idx]
        for i, j, k in face:
            tris.append(torch.stack([corners[:, i], corners[:, j], corners[:, k]], dim=1))
            cols.append(base * shade)
    return torch.cat(tris, dim=0), torch.cat(cols, dim=0)


def _to_camera(tris_ego: torch.Tensor, rot: torch.Tensor, cam_pos: torch.Tensor) -> torch.Tensor:
    """Transform ego-frame triangle vertices into the camera frame."""
    return (tris_ego - cam_pos) @ rot.T


def _clip_near(tris: torch.Tensor, colors: torch.Tensor, near: float):
    """Clip camera-frame triangles against the near plane z = near.

    Triangles straddling the plane are re-cut so that geometry directly under
    the camera still renders. Fully-behind triangles are dropped.

    Clipping against one plane yields at most two triangles, and each source
    triangle keeps a fixed pair of output rows, so the result stays in primitive
    order. That is what the compositor's depth tie-break is defined on, and it is
    the order the CUDA kernel reproduces with its scratch slot pairs; compacting
    the unused rows away preserves it.
    """
    if tris.shape[0] == 0:
        return tris, colors

    count_in = tris.shape[0]
    out = tris.new_zeros((2 * count_in, 3, 3))
    filled = torch.zeros(2 * count_in, dtype=torch.bool, device=tris.device)
    inside = tris[..., 2] >= near
    count = inside.sum(dim=1)

    def intersect(p_in, p_out):
        """Point where segment p_in->p_out crosses z = near."""
        # p_out is behind the plane and p_in in front of it, so the denominator is
        # negative by construction. Clamping its magnitude has to keep that sign:
        # clamp_min turned every real crossing into a division by 1e-9, which threw
        # the clipped vertex out past 1e11 and collapsed the triangle into a
        # zero-area sliver on the horizon. Straddling geometry then disappeared
        # instead of being re-cut -- most visibly the lane-area quad the camera
        # sits on, which blanked the road right in front of the vehicle.
        den = p_out[..., 2] - p_in[..., 2]
        den = torch.where(den > -1e-9, torch.full_like(den, -1e-9), den)
        t = (near - p_in[..., 2]) / den
        return p_in + t.unsqueeze(-1) * (p_out - p_in)

    sel = (count == 3).nonzero().flatten()
    if sel.numel():
        out[2 * sel] = tris[sel]
        filled[2 * sel] = True

    # One vertex inside: the triangle shrinks to a smaller triangle.
    sel = (count == 1).nonzero().flatten()
    if sel.numel():
        t, m = tris[sel], inside[sel]
        idx = m.float().argmax(dim=1)
        rows = torch.arange(t.shape[0], device=tris.device)
        a = t[rows, idx]
        b = t[rows, (idx + 1) % 3]
        d = t[rows, (idx + 2) % 3]
        out[2 * sel] = torch.stack([a, intersect(a, b), intersect(a, d)], dim=1)
        filled[2 * sel] = True

    # Two vertices inside: the triangle becomes a quad, emitted as two triangles.
    sel = (count == 2).nonzero().flatten()
    if sel.numel():
        t, m = tris[sel], inside[sel]
        idx = (~m).float().argmax(dim=1)  # the single outside vertex
        rows = torch.arange(t.shape[0], device=tris.device)
        a = t[rows, idx]
        b = t[rows, (idx + 1) % 3]
        d = t[rows, (idx + 2) % 3]
        ab, ad = intersect(b, a), intersect(d, a)
        out[2 * sel] = torch.stack([b, d, ad], dim=1)
        out[2 * sel + 1] = torch.stack([b, ad, ab], dim=1)
        filled[2 * sel] = True
        filled[2 * sel + 1] = True

    return out[filled], colors.repeat_interleave(2, dim=0)[filled]


def _project(tris_cam: torch.Tensor, fx, fy, cx, cy):
    """Perspective-project camera-frame triangles to pixel coordinates.

    Returns (screen [T, 3, 2], depth [T, 3]).
    """
    z = tris_cam[..., 2].clamp_min(1e-6)
    u = fx * tris_cam[..., 0] / z + cx
    v = fy * tris_cam[..., 1] / z + cy
    return torch.stack([u, v], dim=-1), tris_cam[..., 2]


def _coverage_and_depth(screen, depth, px, py, eps=1e-9):
    """Analytic edge coverage and interpolated depth for every triangle/pixel pair.

    Coverage is the product over the three edges of the edge's signed distance in
    pixels, offset by half a pixel and clamped to [0, 1]. That is the standard
    single-sample analytic-coverage approximation: it antialiases edges at one
    sample per pixel, which is what lets thin lane markings survive at low
    resolution instead of breaking into stair-stepped fragments.

    Returns (coverage [T, P], depth [T, P]).
    """
    a, b, c = screen[:, 0], screen[:, 1], screen[:, 2]

    area = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    # Normalise winding so that interior edge functions are positive.
    sign = torch.where(area >= 0, 1.0, -1.0).unsqueeze(-1)
    area_abs = area.abs().clamp_min(eps).unsqueeze(-1)

    cov = None
    bary = []
    for p, q in ((b, c), (c, a), (a, b)):
        # Edge function for the edge opposite the remaining vertex, as the cross
        # product (q - p) x (X - p). Evaluated at the opposite vertex this equals
        # the signed triangle area, so normalising by `sign` makes it positive
        # throughout the interior regardless of winding.
        ex = q[:, 0:1] - p[:, 0:1]
        ey = q[:, 1:2] - p[:, 1:2]
        e = (ex * (py - p[:, 1:2]) - ey * (px - p[:, 0:1])) * sign
        bary.append(e / area_abs)
        # Signed distance in pixels, used for the coverage ramp.
        dist = e / torch.sqrt(ex * ex + ey * ey).clamp_min(eps)
        edge_cov = (dist + 0.5).clamp(0.0, 1.0)
        cov = edge_cov if cov is None else cov * edge_cov

    # Restrict coverage to the triangle's bounding box, dilated by the same half
    # pixel the edge ramps use. Without this, slivers leak: when a triangle is a
    # fraction of a pixel thick, its two long edges are nearly collinear, and
    # dilating both by half a pixel makes their half-planes overlap in a narrow
    # wedge that extends tens of pixels past the shared vertex. A real rasterizer
    # never sees those pixels because it iterates the bounding box, so bounding
    # here is also what keeps this reference in step with the CUDA kernel.
    umin = screen[..., 0].min(dim=1).values.unsqueeze(-1) - 0.5
    umax = screen[..., 0].max(dim=1).values.unsqueeze(-1) + 0.5
    vmin = screen[..., 1].min(dim=1).values.unsqueeze(-1) - 0.5
    vmax = screen[..., 1].max(dim=1).values.unsqueeze(-1) + 0.5
    in_box = (px >= umin) & (px <= umax) & (py >= vmin) & (py <= vmax)
    cov = torch.where(in_box, cov, torch.zeros_like(cov))

    w0, w1, w2 = bary
    # Perspective-correct depth. Screen-space barycentrics interpolate 1/z
    # linearly, not z, and the error in interpolating z directly grows with how
    # much depth the triangle spans on screen. It was invisible while every
    # primitive was a short road segment or a car-sized box; the ground quad the
    # camera stands on runs from the near plane to tens of metres, and shading it
    # from an affine depth painted the haze of its far end across the near end,
    # leaving the surface under the vehicle flat and a shade apart from the quads
    # beyond it.
    inv = w0 / depth[:, 0:1] + w1 / depth[:, 1:2] + w2 / depth[:, 2:3]
    z = torch.where(inv > eps, 1.0 / inv.clamp_min(eps), torch.full_like(inv, float("inf")))
    # Degenerate (zero-area) triangles contribute nothing.
    cov = torch.where(area.abs().unsqueeze(-1) < eps, torch.zeros_like(cov), cov)
    return cov, z


def _background(rot, cam_pos, fx, fy, cx, cy, width, height, palette, device, dtype):
    """Sky above the horizon, asphalt below, with exact ground depth per pixel.

    Returns (color [P, 3], depth [P]) where sky depth is +inf.
    """
    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype) + 0.5,
        torch.arange(width, device=device, dtype=dtype) + 0.5,
        indexing="ij",
    )
    px, py = xs.reshape(-1), ys.reshape(-1)

    dir_cam = torch.stack([(px - cx) / fx, (py - cy) / fy, torch.ones_like(px)], dim=-1)
    dir_ego = dir_cam @ rot  # rot is ego->camera, so its transpose maps back

    # Ground plane is z = 0 in the ego frame; the camera sits above it.
    denom = dir_ego[:, 2]
    t = torch.where(denom < -1e-6, -cam_pos[2] / denom, torch.full_like(denom, float("inf")))
    depth = torch.where(torch.isfinite(t), t * dir_cam[:, 2], torch.full_like(t, float("inf")))

    ground = torch.tensor(palette.ground, device=device, dtype=dtype)
    sky = torch.tensor(palette.sky, device=device, dtype=dtype)
    hit = torch.isfinite(depth).unsqueeze(-1)
    color = torch.where(hit, ground, sky).expand(px.shape[0], 3).clone()

    # Fade the asphalt with distance so the horizon does not read as a hard edge.
    scale = _depth_scale(depth, palette).unsqueeze(-1)
    color = torch.where(hit, _haze(color, scale, sky), color)
    return color, depth


def _depth_scale(depth, palette):
    """Depth-aware brightness: near is bright, far fades toward the horizon."""
    s = 1.0 - depth.clamp(min=0.0) / palette.depth_falloff
    return s.clamp(palette.depth_min_scale, 1.0)


def _haze(color, scale, sky):
    """Fade a surface toward the horizon with distance (aerial perspective).

    Every layer has to fade the same way, or contrast between two of them does not
    merely weaken with range -- it crosses zero and inverts. Fading fragments
    toward black while the ground faded toward `sky` did exactly that: the two met
    at 24 m for a road edge and 32 m for a lane line, where the marking took the
    asphalt's own colour and vanished, and past that markings read as dark lines
    on bright asphalt. Blending both toward `sky` leaves the difference between
    any two surfaces at `(a - b) * scale`, which decays to `depth_min_scale` but
    keeps its sign at every range.
    """
    return color * scale + sky * (1.0 - scale)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _merge_coplanar(cov, depth, colors):
    """Fold the coplanar fragments of one colour into a single surface per pixel.

    Road primitives tile a surface rather than stack on it: two abutting lane-area
    quads split the pixels along their shared joint between them, and so do the two
    triangles each quad is made of. Compositing those as independent fragments
    leaves 1 - (1 - a)(1 - b) of the pixel to whatever is behind, so a dark seam is
    drawn along every joint and every quad diagonal and the drivable area reads as
    a mosaic of tiles instead of one surface. Adding the coverages is exact where
    the primitives tile and saturates where they overlap, which is what an opaque
    ground surface wants either way.

    The surface's depth is the coverage-weighted mean of its fragments. That is
    exact when they are coplanar, which road primitives are, and unlike picking the
    nearest fragment it moves continuously with coverage: a fragment whose coverage
    rounds to zero on one device and to 1e-7 on another cannot swing the whole
    surface's haze to its own depth. The sum is carried by the lowest-indexed
    covering fragment of the colour, so the fragment ordering and its tie-break
    stay what the compositor and the CUDA kernel already agree on.

    Returns (coverage, depth) in the input's [T, P] shape, with everything but each
    surface's carrier zeroed out.
    """
    if cov.shape[0] == 0:
        return cov, depth

    groups, inverse = torch.unique(colors, dim=0, return_inverse=True)
    shape = (groups.shape[0], cov.shape[1])
    covering = cov > 0
    total = cov.new_zeros(shape).index_add_(0, inverse, cov)
    weighted = cov.new_zeros(shape).index_add_(
        0, inverse, cov * torch.where(covering, depth, torch.zeros_like(depth))
    )

    order = torch.arange(cov.shape[0], device=cov.device).unsqueeze(1).expand_as(cov)
    unset = torch.iinfo(torch.int64).max
    candidate = torch.where(covering, order, torch.full_like(order, unset))
    first = order.new_full(shape, unset).index_reduce_(0, inverse, candidate, "amin")
    carrier = order == first[inverse]

    merged_cov = torch.where(carrier, total.clamp(max=1.0)[inverse], torch.zeros_like(cov))
    merged_depth = torch.where(carrier, (weighted / total.clamp_min(1e-9))[inverse], depth)
    return merged_cov, merged_depth


def render(
    agents: torch.Tensor,
    roads: torch.Tensor,
    egos: torch.Tensor,
    cameras: list[Camera] | None = None,
    palette: Palette | None = None,
    pixel_chunk: int = 4096,
) -> torch.Tensor:
    """Render one scene from every ego, through every camera in the rig.

    Parameters
    ----------
    agents : [A, 8] world-frame agent boxes (x, y, cos_h, sin_h, l, w, h, type).
    roads  : [R, 6] world-frame road segments (x0, y0, x1, y1, width, type).
    egos   : [E, 4] or [E, 5] ego poses (x, y, cos_h, sin_h[, self_index]).
             `self_index` is the ego's own row in `agents`, which is skipped so
             the camera does not stare at the vehicle it is mounted on; -1 or an
             absent column means nothing is skipped.
    cameras: rig; defaults to the paper's front camera alone.

    Returns
    -------
    uint8 tensor [E, num_cameras, 3, H, W].
    """
    cameras = cameras or DEFAULT_RIG
    palette = palette or DEFAULT_PALETTE
    device, dtype = egos.device, egos.dtype

    heights = {cam.height for cam in cameras}
    widths = {cam.width for cam in cameras}
    if len(heights) != 1 or len(widths) != 1:
        raise ValueError("All cameras in a rig must share a resolution")
    height, width = heights.pop(), widths.pop()

    out = torch.empty((egos.shape[0], len(cameras), 3, height, width), dtype=torch.uint8, device=device)

    ys, xs = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype) + 0.5,
        torch.arange(width, device=device, dtype=dtype) + 0.5,
        indexing="ij",
    )
    px_all, py_all = xs.reshape(1, -1), ys.reshape(1, -1)

    for e in range(egos.shape[0]):
        ego = egos[e]
        # Drop this ego's own box: a camera mounted on a car does not see it, and
        # leaving it in would fill the frame with a zero-distance primitive.
        visible = agents
        if egos.shape[1] > 4:
            self_index = int(ego[4].item())
            if 0 <= self_index < agents.shape[0]:
                keep = torch.ones(agents.shape[0], dtype=torch.bool, device=agents.device)
                keep[self_index] = False
                visible = agents[keep]

        road_tris, road_cols = _road_triangles(roads, ego, palette)
        agent_tris, agent_cols = _agent_triangles(visible, ego, palette)

        for c, cam in enumerate(cameras):
            rot = cam.rotation().to(device=device, dtype=dtype)
            cam_pos = torch.tensor(cam.pos, device=device, dtype=dtype)
            fx, fy, cx, cy = cam.intrinsics()

            bg_color, bg_depth = _background(
                rot, cam_pos, fx, fy, cx, cy, width, height, palette, device, dtype
            )
            sky = torch.tensor(palette.sky, device=device, dtype=dtype)

            layers = []
            for tris_ego, cols in ((road_tris, road_cols), (agent_tris, agent_cols)):
                tris_cam = _to_camera(tris_ego, rot, cam_pos)
                tris_cam, cols_c = _clip_near(tris_cam, cols, cam.near)
                layers.append((tris_cam, cols_c))

            image = torch.empty((height * width, 3), device=device, dtype=dtype)
            for start in range(0, height * width, pixel_chunk):
                stop = min(start + pixel_chunk, height * width)
                px, py = px_all[:, start:stop], py_all[:, start:stop]
                acc = bg_color[start:stop]
                depth_limit = bg_depth[start:stop]

                for layer_idx, (tris_cam, cols_c) in enumerate(layers):
                    if tris_cam.shape[0] == 0:
                        continue
                    screen, depth_v = _project(tris_cam, fx, fy, cx, cy)
                    cov, zbuf = _coverage_and_depth(screen, depth_v, px, py)
                    shaded = cols_c
                    if layer_idx == 0:
                        # Drop what the compositor would discard anyway before
                        # merging, so a fragment beyond the far plane cannot lend
                        # its coverage to a surface that survives.
                        cov = torch.where(
                            (zbuf > cam.far) | (zbuf <= 0), torch.zeros_like(cov), cov
                        )
                        cov, zbuf = _merge_coplanar(cov, zbuf, cols_c)
                    # Road markings are coplanar with the ground, so they are drawn
                    # over it rather than depth-tested against it; agents are.
                    limit = (
                        torch.full_like(depth_limit, float("inf"))
                        if layer_idx == 0
                        else depth_limit
                    )
                    scale = _depth_scale(zbuf, palette)
                    # Road decals are painted in buffer order; agents are depth-sorted.
                    order_key = None
                    if layer_idx == 0:
                        order_key = (
                            torch.arange(cov.shape[0], device=cov.device, dtype=cov.dtype)
                            .unsqueeze(1)
                            .expand_as(cov)
                        )
                    weighted = _composite_shaded(
                        cov, zbuf, shaded, scale, sky, acc, limit, cam.far, order_key
                    )
                    acc = weighted
                image[start:stop] = acc

            frame = image.reshape(height, width, 3).clamp(0.0, 1.0)
            out[e, c] = (frame * 255.0 + 0.5).to(torch.uint8).permute(2, 0, 1)

    return out


def _composite_shaded(
    cov, depth, colors, depth_scale, sky, background, background_depth, far, order_key=None
):
    """`_composite`, with per-fragment aerial perspective applied.

    The haze varies per pixel (it depends on interpolated depth), so the colour
    cannot be folded into the per-triangle table beforehand.

    `order_key` replaces depth as the front-to-back order. The road layer passes
    the primitive index: its fragments are decals on one plane, so which of them is
    on top is a painter's-order decision the buffer already encodes -- markings are
    emitted before the lane area they are painted on -- and their interpolated
    depths agree to within float noise, which would otherwise settle the order
    differently on the CPU and in the kernel.
    """
    if cov.shape[0] == 0:
        return background

    cov = torch.where(depth > far, torch.zeros_like(cov), cov)
    # A box's bottom face lies exactly on the ground plane, so its depth equals the
    # ground intercept to within rounding and an exact comparison decides that face
    # by float noise -- differently in float32 on the CPU and in the kernel. The
    # bound only has to reject boxes genuinely behind the ground point, so give it
    # a relative slack far below anything the geometry cares about.
    limit = background_depth.unsqueeze(0) * (1.0 + GROUND_DEPTH_SLACK)
    cov = torch.where(depth > limit, torch.zeros_like(cov), cov)
    cov = torch.where(depth <= 0, torch.zeros_like(cov), cov)

    # Sort covering fragments nearest first. Pushing non-covering ones to infinity
    # keeps them out of the retained window, and a stable sort breaks depth ties by
    # primitive index -- coplanar road markings routinely land at identical depth,
    # and the CUDA kernel breaks those ties the same way so the two agree
    # regardless of the order its threads visit primitives in.
    key = torch.round(depth / DEPTH_ORDER_QUANTUM) if order_key is None else order_key
    sort_key = torch.where(cov > 0, key, torch.full_like(key, float("inf")))
    order = torch.argsort(sort_key, dim=0, stable=True)
    if order.shape[0] > MAX_FRAGMENTS:
        order = order[:MAX_FRAGMENTS]
    cov_s = torch.gather(cov, 0, order)
    scale_s = torch.gather(depth_scale, 0, order).unsqueeze(-1)
    col_s = _haze(colors[order], scale_s, sky)

    # Transmittance reaching each layer: the product of (1 - alpha) over all
    # strictly nearer layers, i.e. an exclusive prefix product. Computing it as
    # cumprod / one_minus would divide by zero exactly where a fragment is fully
    # opaque, which is the common case for a box interior.
    one_minus = 1.0 - cov_s
    inclusive = torch.cumprod(one_minus, dim=0)
    trans = torch.cat([torch.ones_like(inclusive[:1]), inclusive[:-1]], dim=0)

    weight = (cov_s * trans).unsqueeze(-1)
    out = (weight * col_s).sum(dim=0)
    return out + background * torch.prod(one_minus, dim=0).unsqueeze(-1)
