"""CUDA perspective rasterizer front end.

Thin wrapper over the `pufferlib::drive_raster` op in
`pufferlib/extensions/cuda/raster.cu`. The signature mirrors
`pufferlib.ocean.drive.raster_ref.render` so the two are interchangeable, and
`tests/test_raster.py` holds the kernel to the reference's output.

The kernel runs inside the training process on the GPU's general-purpose compute
cores, so rendered views reach the network without a host round trip.
"""

from __future__ import annotations

import torch

import pufferlib  # noqa: F401  (loads pufferlib._C, registering the op)
from pufferlib.ocean.drive.raster_ref import (
    DEFAULT_RIG,
    NUPLAN_RIG,  # noqa: F401  (re-exported for callers)
    WAYMO_RIG,  # noqa: F401  (re-exported for callers)
    Camera,
    rig_from_config,  # noqa: F401
    rig_tensor,
)

_OP = None


def _op():
    global _OP
    if _OP is None:
        import pufferlib._C  # noqa: F401

        _OP = torch.ops.pufferlib.drive_raster
    return _OP


def render(
    agents: torch.Tensor,
    roads: torch.Tensor,
    egos: torch.Tensor,
    cameras: list[Camera] | None = None,
    rig: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    ego_scene: torch.Tensor | None = None,
    agent_ranges: torch.Tensor | None = None,
    road_ranges: torch.Tensor | None = None,
) -> torch.Tensor:
    """Render from every ego, through every camera in the rig.

    Parameters match `raster_ref.render`: `agents` is [A, 8], `roads` is [R, 6],
    and `egos` is [E, 4] or [E, 5] where the fifth column is the ego's own row in
    `agents`, skipped so the camera does not see the vehicle carrying it.

    Several environments can be rendered in one launch by concatenating their
    scenes and describing the split: `agent_ranges` and `road_ranges` are int32
    prefix offsets of length num_scenes + 1, and `ego_scene` says which scene each
    ego belongs to. Omitting them treats the input as a single scene. Batching
    matters at training scale, where a step spans a hundred or more environments
    and per-scene launches would cost more than the rendering.

    Pass `rig` to reuse a packed rig tensor across calls, and `out` to render into
    an existing uint8 [E, C, 3, H, W] buffer.
    """
    cameras = cameras or DEFAULT_RIG
    device = egos.device
    if not device.type == "cuda":
        raise ValueError("raster_cuda.render requires CUDA tensors; use raster_ref for CPU")

    heights = {cam.height for cam in cameras}
    widths = {cam.width for cam in cameras}
    if len(heights) != 1 or len(widths) != 1:
        raise ValueError("All cameras in a rig must share a resolution")
    height, width = heights.pop(), widths.pop()

    if rig is None:
        rig = rig_tensor(cameras, device=device)
    rig = rig.to(device=device, dtype=torch.float32).contiguous()

    agents = agents.to(device=device, dtype=torch.float32).contiguous()
    roads = roads.to(device=device, dtype=torch.float32).contiguous()
    egos = egos.to(device=device, dtype=torch.float32).contiguous()

    if ego_scene is None:
        ego_scene = torch.zeros(egos.shape[0], dtype=torch.int32, device=device)
        agent_ranges = torch.tensor([0, agents.shape[0]], dtype=torch.int32, device=device)
        road_ranges = torch.tensor([0, roads.shape[0]], dtype=torch.int32, device=device)
    ego_scene = ego_scene.to(device=device, dtype=torch.int32).contiguous()
    agent_ranges = agent_ranges.to(device=device, dtype=torch.int32).contiguous()
    road_ranges = road_ranges.to(device=device, dtype=torch.int32).contiguous()

    if out is None:
        out = torch.empty(
            (egos.shape[0], len(cameras), 3, height, width), dtype=torch.uint8, device=device
        )
    _op()(agents, roads, egos, rig, ego_scene, agent_ranges, road_ranges, out)
    return out
