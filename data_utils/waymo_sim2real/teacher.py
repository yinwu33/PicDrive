"""Frozen ``DriveCam`` teacher: scene encoder and planning head.

Two stages need the pinned simulation policy for different reasons.
``extract_teacher_features`` runs its perception half to cache the 256-D scene
feature; ``train_distillation`` runs its planning half to score how differently
the frozen planner acts on the student's feature.  Both go through this module
so a single checkpoint SHA256 pins the whole network.

The split point is ``scene_encoder``.  Everything before it is perception and is
replaced by the real-image student; everything after it -- the ego encoder, the
shared trunk and the actor -- is the planning head the paper keeps frozen.
"""

from __future__ import annotations

import hashlib
import warnings
from pathlib import Path
from types import SimpleNamespace

import gymnasium
import torch
from torch import nn

from pufferlib.ocean.torch import DriveCam

from .processed import SIM_HEIGHT, SIM_WIDTH, TEACHER_FEATURE_DIM


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_state(checkpoint: str | Path) -> dict[str, torch.Tensor]:
    """Unwrap a checkpoint down to the bare ``DriveCam`` state dict.

    A run with ``rnn_name = Recurrent`` saves the policy nested under a
    ``policy.`` prefix alongside the recurrent core's own ``lstm.``/``cell.``
    tensors.  The scene encoder this module exists to serve is upstream of that
    core and is bit-identical either way, so the prefix is stripped rather than
    treated as a different network.

    The recurrent tensors are dropped, and that is *not* free:
    ``FrozenPlanningHead`` runs ``backbone -> actor``, so for a recurrent
    checkpoint the planning term is evaluated through a feed-forward slice of a
    policy whose real actions also depend on LSTM state.  Both the teacher and
    student branches lose the same core, so ``plan_kl`` remains a valid measure
    of how far a feature error moves the wheel -- it is just not the deployed
    policy's action distribution.  Callers are warned so this never happens
    silently.
    """
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise TypeError(f"{checkpoint} contains {type(state).__name__}, expected a state dict")
    state = {key.removeprefix("module."): value for key, value in state.items()}
    recurrent = sorted(key for key in state if key.startswith(("lstm.", "cell.")))
    if recurrent:
        warnings.warn(
            f"{checkpoint} is a recurrent checkpoint; dropping {len(recurrent)} tensors "
            f"({', '.join(recurrent[:3])}...). The 256-D scene feature is unaffected, but the "
            "planning head runs backbone->actor without the LSTM, so plan_kl and plan_agreement "
            "are measured through a feed-forward slice of the policy.",
            RuntimeWarning,
            stacklevel=3,
        )
        state = {key: value for key, value in state.items() if key not in set(recurrent)}
    return {key.removeprefix("policy."): value for key, value in state.items()}


def load_teacher(checkpoint: str | Path, device: torch.device | str = "cpu") -> DriveCam:
    """Instantiate the pinned teacher, shaped from its own weights.

    Every width is read back from the checkpoint rather than assumed, so a
    ``strict=True`` load is a real assertion that this is the network the
    features were extracted with instead of a shape coincidence.
    """
    state = _load_state(checkpoint)
    try:
        input_size = state["cam_proj.0.weight"].shape[0]
        scene_dim, fused_width = state["scene_encoder.weight"].shape
        ego_features = state["ego_encoder.0.weight"].shape[1]
        hidden_size, trunk_input = state["backbone.0.weight"].shape
        action_dim = state["actor.weight"].shape[0]
        cnn_channels = state["cnn.12.weight"].shape[0]
    except KeyError as error:
        raise ValueError(f"{checkpoint} is not a DriveCam checkpoint: missing {error}") from error

    if fused_width % input_size:
        raise ValueError(f"scene_encoder input {fused_width} is not a multiple of {input_size}")
    num_cameras = fused_width // input_size
    if scene_dim != TEACHER_FEATURE_DIM:
        raise ValueError(f"{checkpoint} emits a {scene_dim}-D scene feature, expected {TEACHER_FEATURE_DIM}")
    if trunk_input != scene_dim + state["ego_encoder.2.weight"].shape[0]:
        raise ValueError(f"{checkpoint} trunk input {trunk_input} does not match scene + ego widths")
    backbone_layers = sum(1 for key in state if key.startswith("backbone.") and key.endswith(".weight"))

    env = SimpleNamespace(
        num_cameras=num_cameras,
        height=SIM_HEIGHT,
        width=SIM_WIDTH,
        image_bytes=num_cameras * 3 * SIM_HEIGHT * SIM_WIDTH,
        ego_dim=ego_features,
        single_action_space=gymnasium.spaces.MultiDiscrete([action_dim]),
    )
    teacher = DriveCam(
        env,
        input_size=input_size,
        hidden_size=hidden_size,
        cnn_channels=cnn_channels,
        scene_dim=scene_dim,
        ego_dim=state["ego_encoder.2.weight"].shape[0],
        backbone_layers=backbone_layers,
    )
    teacher.load_state_dict(state, strict=True)
    teacher.requires_grad_(False)
    teacher.eval()
    return teacher.to(device)


def scene_features(teacher: DriveCam, images: torch.Tensor) -> torch.Tensor:
    """Run the teacher's perception half on ``[batch, camera, 3, H, W]`` uint8."""
    batch, cameras = images.shape[:2]
    features = teacher.cnn(images.reshape(batch * cameras, 3, SIM_HEIGHT, SIM_WIDTH).float() / 255.0)
    features = teacher.cam_proj(features.flatten(1)).reshape(batch, -1)
    return teacher.scene_encoder(features)


class FrozenPlanningHead(nn.Module):
    """The teacher's planning half, held frozen for both branches.

    This is the module the paper keeps fixed while adapting perception: it turns
    a scene feature and the ego vector into action logits, so a divergence
    measured at its output is a divergence in behaviour rather than in latent
    coordinates.  Ablating the freeze cost the paper 2.7 HD-Score points, so the
    parameters are detached here and ``train()`` is a no-op.
    """

    def __init__(self, teacher: DriveCam):
        super().__init__()
        self.ego_encoder = teacher.ego_encoder
        self.trunk = teacher.backbone
        self.actor = teacher.actor
        self.value_fn = teacher.value_fn
        self.action_dims = list(teacher.atn_dim)
        self.ego_features = int(teacher.ego_features)
        self.scene_dim = int(teacher.scene_encoder.out_features)
        self.requires_grad_(False)
        super().train(False)

    def train(self, mode: bool = True) -> "FrozenPlanningHead":
        # The distillation target must not move when the student switches modes.
        return self

    def forward(self, scene: torch.Tensor, ego: torch.Tensor) -> list[torch.Tensor]:
        if scene.shape[-1] != self.scene_dim:
            raise ValueError(f"scene feature must be {self.scene_dim}-D, got {scene.shape[-1]}")
        if ego.shape[-1] != self.ego_features:
            raise ValueError(f"ego vector must be {self.ego_features}-D, got {ego.shape[-1]}")
        hidden = self.trunk(torch.cat([scene, self.ego_encoder(ego)], dim=1))
        return list(torch.split(self.actor(hidden), self.action_dims, dim=1))
