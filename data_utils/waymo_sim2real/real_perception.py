"""Pretrained real-image encoder for Waymo-to-simulation distillation.

The frozen simulation teacher produces one 256-D scene feature from three
96x64 abstract camera renders.  :class:`RealPerception` consumes the matching
three 384x256 RGB images and emits a feature in exactly that space. The
ImageNet-pretrained backbone is shared across cameras; fusion happens only
after each view has been encoded, matching the camera-sharing structure of
``DriveCam`` while giving the real branch substantially more capacity.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn
import torch.nn.functional as F

try:
    from torchvision.models import ConvNeXt_Tiny_Weights, convnext_tiny
except ImportError as error:  # pragma: no cover - exercised only in incomplete environments
    raise ImportError(
        "real-perception training requires torchvision; install torchvision==0.19.1 "
        "for this repository's torch==2.4.1 environment"
    ) from error


@dataclass(frozen=True)
class RealPerceptionConfig:
    """Serializable architecture configuration."""

    num_cameras: int = 3
    feature_dim: int = 256
    backbone: str = "convnext_tiny"
    fusion_dim: int = 1024
    fusion_dropout: float = 0.1

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


class RealPerception(nn.Module):
    """Encode three real RGB views into the frozen teacher's 256-D space.

    Input is uint8 or float ``[batch, camera, 3, height, width]`` in the fixed
    camera order ``front, front_left, front_right``. Float input is assumed to
    be in ``[0, 1]``; uint8 input is scaled internally.
    """

    PRETRAINED_SOURCE = "torchvision:ConvNeXt_Tiny_Weights.IMAGENET1K_V1"
    PRETRAINED_SHA256 = "983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d"

    def __init__(self, config: RealPerceptionConfig | None = None, *, pretrained: bool = True):
        super().__init__()
        self.config = config or RealPerceptionConfig()
        if self.config.num_cameras < 1:
            raise ValueError("num_cameras must be positive")
        if self.config.backbone != "convnext_tiny":
            raise ValueError(f"unsupported backbone: {self.config.backbone}")

        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None
        source = convnext_tiny(weights=weights)
        self.backbone = source.features
        self.avgpool = source.avgpool
        self.view_norm = source.classifier[0]
        self.backbone_feature_dim = 768
        self.pretrained_source = self.PRETRAINED_SOURCE if pretrained else "random"
        self.pretrained_sha256 = self.PRETRAINED_SHA256 if pretrained else None

        self.camera_embedding = nn.Parameter(
            torch.zeros(self.config.num_cameras, self.backbone_feature_dim)
        )
        fused_width = self.config.num_cameras * self.backbone_feature_dim
        self.fusion = nn.Sequential(
            nn.LayerNorm(fused_width, eps=1e-6),
            nn.Linear(fused_width, self.config.fusion_dim),
            nn.GELU(),
            nn.Dropout(self.config.fusion_dropout),
            nn.Linear(self.config.fusion_dim, self.config.feature_dim),
        )

        self.register_buffer(
            "image_mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1), persistent=True
        )
        self.register_buffer(
            "image_std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1), persistent=True
        )
        # Do not call ``self.apply`` here: that would silently overwrite the
        # official ImageNet backbone. Only the task-specific parameters start
        # from scratch.
        self.fusion.apply(self._initialize)
        nn.init.trunc_normal_(self.camera_embedding, std=0.02)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _normalize(self, images: torch.Tensor) -> torch.Tensor:
        images = images.float().div(255.0) if images.dtype == torch.uint8 else images
        return (images - self.image_mean) / self.image_std

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 5 or images.shape[1] != self.config.num_cameras or images.shape[2] != 3:
            raise ValueError(
                "images must be [batch, camera, 3, height, width], got " f"{tuple(images.shape)}"
            )
        batch, cameras = images.shape[:2]
        x = self._normalize(images.reshape(batch * cameras, 3, images.shape[-2], images.shape[-1]))
        x = self.avgpool(self.backbone(x))
        views = self.view_norm(x).flatten(1).reshape(batch, cameras, -1)
        views = views + self.camera_embedding.unsqueeze(0)
        return self.fusion(views.flatten(1))

    @property
    def trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)


class FeatureAlignmentLoss(nn.Module):
    """Exact feature regression plus an angular alignment term."""

    def __init__(self, cosine_weight: float = 0.1):
        super().__init__()
        if cosine_weight < 0:
            raise ValueError("cosine_weight must be non-negative")
        self.cosine_weight = cosine_weight

    def forward(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        prediction = prediction.float()
        target = target.float()
        mse = F.mse_loss(prediction, target)
        cosine = F.cosine_similarity(prediction, target, dim=-1).mean()
        loss = mse + self.cosine_weight * (1.0 - cosine)
        metrics = {
            "loss": loss.detach(),
            "mse": mse.detach(),
            "rmse": mse.detach().sqrt(),
            "cosine": cosine.detach(),
        }
        return loss, metrics
