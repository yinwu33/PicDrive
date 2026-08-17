"""Freeze the closed-loop policy into an interpreter-independent bundle.

The bundle originally existed to cross an interpreter boundary: CARLA 0.9.15
publishes no Python 3.12 wheel, so the leaderboard had to run in a second
Python 3.10 venv without ``pufferlib``.  That boundary is gone -- the repository
venv is Python 3.10.16 and runs the leaderboard itself -- but the bundle is
kept, because a frozen graph is what makes a score reproducible: it pins the
policy at export time instead of letting it drift with the working tree.

Both halves are traced to TorchScript, which carries its own graph and weights
and needs nothing but ``torch`` to load:

    encoder.ts   uint8 [1, 3, 3, 256, 384]                  -> scene [1, 256]
    planner.ts   (scene [1, 256], ego [1, 24], h, c)        -> (logits [1, 12], h, c)

The LSTM state is an explicit input and output instead of module state, because
the leaderboard calls the agent once per tick and owns no notion of a rollout;
the agent carries ``(h, c)`` between calls and zeroes them per route, which is
exactly what :class:`RecurrentPlanningRuntime` does inside its own loop.

``bundle.json`` carries everything else the agent would otherwise have to
recompute: the camera rig, the conditioning vector, and the checkpoint digests
that say which policy this is.

    source scripts/define_env.sh
    "$PICDRIVE_PYTHON" -m data_utils.sim2real.carla.b2d.export \
        --student artifacts/carla_sim2real/runs/puffer_giga_3cam_distillation/deployment.pt \
        --checkpoint experiments/skynet/puffer_giga_3cam_20260813_105022_0ejcqldx.pt \
        --output artifacts/carla_sim2real/b2d_bundle
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from data_utils.sim2real.waymo.giga_conditioning import (
    GIGA_EGO_OBS_DIM,
    conditioning_to_raw,
    good_conditioning,
    nominal_conditioning,
    segment_conditioning,
)
from data_utils.sim2real.waymo.real_perception import load_deployment_bundle
from data_utils.sim2real.waymo.teacher import FrozenPlanningHead, load_frozen_planning_head, sha256_file

from .. import REAL_HEIGHT, REAL_WIDTH
from ..control import carla_mount
from ..rig import rig_cameras, sensor_fov_deg


# A traced graph is only valid for the shapes it saw. Anything the agent feeds
# that disagrees with these is a silent wrong answer, so both sides assert them.
TRACE_BATCH = 1
NUM_CAMERAS = 3
CAMERA_ORDER = ("front", "front_left", "front_right")


class PlannerStep(nn.Module):
    """One recurrent planning tick with the LSTM state passed in and out.

    :class:`FrozenPlanningHead.forward` deliberately starts every sample from
    ``h0 = c0 = 0`` -- distillation scores each paired frame as a first frame.
    A rollout must not do that, so this mirrors ``RecurrentPlanningRuntime.step``
    instead: trunk, then one ``LSTMCell`` advance from the caller's state.
    """

    def __init__(self, head: FrozenPlanningHead):
        super().__init__()
        if head.recurrent is None:
            raise ValueError("closed-loop export requires a recurrent planning checkpoint")
        self.ego_encoder = head.ego_encoder
        self.trunk = head.trunk
        self.recurrent = head.recurrent
        self.actor = head.actor
        self.eval()
        self.requires_grad_(False)

    def forward(
        self,
        scene: torch.Tensor,
        ego: torch.Tensor,
        hidden: torch.Tensor,
        cell: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self.trunk(torch.cat([scene, self.ego_encoder(ego)], dim=1))
        hidden, cell = self.recurrent(encoded, (hidden, cell))
        return self.actor(hidden), hidden, cell


def _conditioning(kind: str, seed: int) -> np.ndarray:
    if kind == "nominal":
        return nominal_conditioning()
    if kind == "good":
        return good_conditioning()
    return segment_conditioning(kind, seed)


def _rig_spec() -> list[dict[str, object]]:
    """The rig as plain numbers, so the agent never imports ``pufferlib``.

    ``mount`` needs the ego bounding-box offset, which only exists once the
    hero is spawned, so the CARLA-frame conversion is left to the agent and
    only the source camera parameters travel in the bundle.
    """

    cameras = rig_cameras()
    if len(cameras) != NUM_CAMERAS:
        raise ValueError(f"expected {NUM_CAMERAS} cameras in the rig, got {len(cameras)}")
    return [
        {
            "id": name,
            "pos": [float(value) for value in camera.pos],
            "pitch_deg": float(camera.pitch_deg),
            "yaw_deg": float(camera.yaw_deg),
            "roll_deg": float(camera.roll_deg),
            "fov_deg": float(sensor_fov_deg(camera)),
            "width": REAL_WIDTH,
            "height": REAL_HEIGHT,
            # Only for the record: what the sensor would sit at on a vehicle
            # with no bounding-box offset, which is what the agent adds back.
            "carla_mount": [list(part) for part in carla_mount(camera.pos, camera.pitch_deg, camera.yaw_deg, camera.roll_deg)],
        }
        for name, camera in zip(CAMERA_ORDER, cameras)
    ]


def _check(name: str, traced: torch.jit.ScriptModule, eager, inputs, tolerance: float) -> float:
    """Trace equality is the whole contract of this export, so measure it."""

    with torch.inference_mode():
        want = eager(*inputs)
        got = traced(*inputs)
    if isinstance(want, torch.Tensor):
        want, got = (want,), (got,)
    error = max(float((a.float() - b.float()).abs().max().item()) for a, b in zip(want, got))
    if not error <= tolerance:
        raise RuntimeError(f"{name} trace disagrees with the eager module by {error:.3e}")
    return error


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student", required=True, help="deployment.pt written by train_distillation")
    parser.add_argument("--checkpoint", required=True, help="giga policy checkpoint with recurrent state")
    parser.add_argument("--output", required=True, help="bundle directory to write")
    parser.add_argument(
        "--conditioning",
        default="good",
        help="'good', 'nominal', or a segment id hashed with --conditioning-seed",
    )
    parser.add_argument("--conditioning-seed", type=int, default=42)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--dt", type=float, default=0.1, help="policy control period in seconds")
    parser.add_argument("--tolerance", type=float, default=1e-4)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is unavailable")
    device = torch.device(args.device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)

    student = load_deployment_bundle(args.student, device).eval()
    head = load_frozen_planning_head(args.checkpoint, device, require_recurrent=True)
    if head.action_dims != [12]:
        raise SystemExit(f"expected one 12-way jerk action, got {head.action_dims}")
    if head.ego_features != GIGA_EGO_OBS_DIM:
        raise SystemExit(f"checkpoint wants {head.ego_features} ego values, not {GIGA_EGO_OBS_DIM}")
    planner = PlannerStep(head).to(device)
    hidden_size = int(head.recurrent.hidden_size)

    images = torch.randint(
        0, 256, (TRACE_BATCH, NUM_CAMERAS, 3, REAL_HEIGHT, REAL_WIDTH), dtype=torch.uint8, device=device
    )
    scene = torch.zeros((TRACE_BATCH, head.scene_dim), device=device)
    ego = torch.zeros((TRACE_BATCH, GIGA_EGO_OBS_DIM), device=device)
    state = torch.zeros((TRACE_BATCH, hidden_size), device=device)

    with torch.inference_mode():
        traced_encoder = torch.jit.trace(student, images, check_trace=False)
        traced_planner = torch.jit.trace(planner, (scene, ego, state, state), check_trace=False)
    traced_encoder = torch.jit.freeze(traced_encoder.eval())
    traced_planner = torch.jit.freeze(traced_planner.eval())

    # Freezing folds constants, so equality has to be re-measured after it, and
    # on inputs the trace never saw.
    probe_images = torch.randint(
        0, 256, images.shape, dtype=torch.uint8, device=device
    )
    probe_scene = torch.randn_like(scene)
    probe_ego = torch.randn_like(ego)
    probe_state = torch.randn_like(state)
    encoder_error = _check("encoder", traced_encoder, student, (probe_images,), args.tolerance)
    planner_error = _check(
        "planner", traced_planner, planner, (probe_scene, probe_ego, probe_state, probe_state), args.tolerance
    )

    conditioning = _conditioning(args.conditioning, args.conditioning_seed)
    torch.jit.save(traced_encoder, output / "encoder.ts")
    torch.jit.save(traced_planner, output / "planner.ts")
    manifest = {
        "schema_version": 1,
        "student": str(Path(args.student).resolve()),
        "student_sha256": sha256_file(args.student),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": sha256_file(args.checkpoint),
        "scene_dim": int(head.scene_dim),
        "ego_dim": GIGA_EGO_OBS_DIM,
        "action_dims": list(head.action_dims),
        "hidden_size": hidden_size,
        "dt": float(args.dt),
        "image_height": REAL_HEIGHT,
        "image_width": REAL_WIDTH,
        "camera_order": list(CAMERA_ORDER),
        "rig": _rig_spec(),
        "conditioning": args.conditioning,
        "conditioning_seed": args.conditioning_seed,
        "conditioning_vector": [float(value) for value in conditioning],
        "conditioning_raw": [float(value) for value in conditioning_to_raw(conditioning)],
        "trace_error": {"encoder": encoder_error, "planner": planner_error},
        "torch_version": torch.__version__,
    }
    (output / "bundle.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"wrote {output}/encoder.ts, planner.ts, bundle.json")
    print(f"trace error: encoder {encoder_error:.3e}, planner {planner_error:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
