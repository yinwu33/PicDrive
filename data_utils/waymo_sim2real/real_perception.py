"""Pretrained real-image encoders for camera-to-simulation distillation.

The frozen simulation teacher produces one 256-D scene feature from three
96x64 abstract camera renders. The real branch consumes the matching three
384x256 RGB images and emits a feature in exactly that space. The default
student is a frozen DINOv2 ViT-S/14 with per-camera scene registers, rank-32
Q/V LoRA, and a trainable transformer fusion head. The older ConvNeXt student
is retained as an ablation.

:class:`DistillationLoss` scores it against the teacher both in feature space
and through the teacher's own frozen planning head, which is the objective
used for frozen-policy sim-to-real adaptation.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


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


@dataclass(frozen=True)
class ViTRealPerceptionConfig:
    """Serializable DrivoR-style DINOv2 student configuration."""

    num_cameras: int = 3
    feature_dim: int = 256
    backbone: str = "vit_small_patch14_reg4_dinov2.lvd142m"
    backbone_repo: str = "timm/vit_small_patch14_reg4_dinov2.lvd142m"
    backbone_revision: str = "main"
    image_height: int = 256
    image_width: int = 384
    num_scene_tokens: int = 16
    lora_rank: int = 32
    fusion_layers: int = 2
    fusion_heads: int = 6
    fusion_mlp_ratio: int = 4
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
        self.pretrained_revision = None

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

    def freeze_backbone_stages(self, stages: int) -> int:
        """Freeze the first ``stages`` ConvNeXt stages; return frozen parameters.

        Full fine-tuning overfits quickly here: the paired set covers a few
        hundred distinct scenes however many frames it holds, and consecutive
        frames are near-duplicates. Holding the early, generic stages fixed is
        the cheapest counterpart to the LoRA the paper puts on its own backbone.
        """
        if stages < 0:
            raise ValueError("stages must be non-negative")
        frozen = 0
        for index, stage in enumerate(self.backbone):
            if index >= stages:
                break
            for parameter in stage.parameters():
                parameter.requires_grad_(False)
                frozen += parameter.numel()
        return frozen

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

    @property
    def frozen_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if not parameter.requires_grad)


class QVLoRALinear(nn.Module):
    """Frozen ViT QKV projection with trainable low-rank Q and V updates.

    DrivoR adapts all attention blocks this way. Keeping the original projection
    as a real submodule makes a checkpoint self-contained while ``requires_grad``
    guarantees that only the low-rank matrices move.
    """

    def __init__(self, base: nn.Linear, rank: int):
        super().__init__()
        if rank < 1:
            raise ValueError("LoRA rank must be positive")
        if base.out_features != 3 * base.in_features:
            raise ValueError("QVLoRALinear expects a combined [Q,K,V] projection")
        self.base = base
        self.base.requires_grad_(False)
        self.dim = base.in_features
        self.rank = rank
        self.lora_q_a = nn.Linear(self.dim, rank, bias=False)
        self.lora_q_b = nn.Linear(rank, self.dim, bias=False)
        self.lora_v_a = nn.Linear(self.dim, rank, bias=False)
        self.lora_v_b = nn.Linear(rank, self.dim, bias=False)
        nn.init.kaiming_uniform_(self.lora_q_a.weight, a=5**0.5)
        nn.init.kaiming_uniform_(self.lora_v_a.weight, a=5**0.5)
        nn.init.zeros_(self.lora_q_b.weight)
        nn.init.zeros_(self.lora_v_b.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        query, key, value = self.base(inputs).split(self.dim, dim=-1)
        query = query + self.lora_q_b(self.lora_q_a(inputs))
        value = value + self.lora_v_b(self.lora_v_a(inputs))
        return torch.cat((query, key, value), dim=-1)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dino_weights(
    config: ViTRealPerceptionConfig, weights_path: str | Path | None
) -> tuple[Path, str]:
    """Resolve a pinned Hugging Face snapshot or validate an explicit file."""

    if weights_path is not None:
        path = Path(weights_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"DINOv2 weights do not exist: {path}")
        return path, config.backbone_revision
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as error:  # pragma: no cover - dependency error path
        raise ImportError(
            "DINOv2 initialization requires huggingface-hub; install the sim2real dependencies"
        ) from error
    revision = HfApi().model_info(config.backbone_repo, revision=config.backbone_revision).sha
    if not revision:
        raise RuntimeError(f"could not resolve a revision for {config.backbone_repo}")
    path = Path(
        hf_hub_download(
            repo_id=config.backbone_repo,
            filename="model.safetensors",
            revision=revision,
        )
    )
    return path, revision


class ViTRealPerception(nn.Module):
    """DrivoR-style DINOv2 ViT-S camera encoder emitting a 256-D scene latent.

    The pretrained DINOv2-S/14 backbone is shared across views and frozen. Each
    camera owns 16 learned scene registers, and rank-32 LoRA updates adapt Q and
    V in every transformer block. A lightweight transformer fuses the resulting
    48 camera-aware registers before projecting them into the simulation
    teacher's visual latent space.
    """

    PRETRAINED_SOURCE = "huggingface:timm/vit_small_patch14_reg4_dinov2.lvd142m"

    def __init__(
        self,
        config: ViTRealPerceptionConfig | None = None,
        *,
        pretrained: bool = True,
        weights_path: str | Path | None = None,
        backbone: nn.Module | None = None,
    ):
        super().__init__()
        self.config = config or ViTRealPerceptionConfig()
        if self.config.num_cameras < 1 or self.config.num_scene_tokens < 1:
            raise ValueError("num_cameras and num_scene_tokens must be positive")
        if self.config.feature_dim < 1 or self.config.fusion_layers < 1:
            raise ValueError("feature_dim and fusion_layers must be positive")
        if self.config.lora_rank < 1:
            raise ValueError("lora_rank must be positive")

        padded_height = _round_up(self.config.image_height, 14)
        padded_width = _round_up(self.config.image_width, 14)
        self.padded_image_size = (padded_height, padded_width)
        resolved_revision: str | None = None
        resolved_weights: Path | None = None
        if backbone is None:
            try:
                import timm
            except ImportError as error:  # pragma: no cover - dependency error path
                raise ImportError(
                    "DINO real-perception training requires timm==1.0.19"
                ) from error
            overlay: dict[str, Any] | None = None
            if pretrained:
                resolved_weights, resolved_revision = _dino_weights(self.config, weights_path)
                overlay = {"file": str(resolved_weights)}
            backbone = timm.create_model(
                self.config.backbone,
                pretrained=pretrained,
                pretrained_cfg_overlay=overlay,
                img_size=self.padded_image_size,
                num_classes=0,
            )
        self.backbone = backbone
        self.backbone.requires_grad_(False)
        self.backbone_feature_dim = int(self.backbone.num_features)
        if self.backbone_feature_dim % self.config.fusion_heads:
            raise ValueError(
                f"backbone width {self.backbone_feature_dim} is not divisible by "
                f"{self.config.fusion_heads} fusion heads"
            )
        patch_size = tuple(int(value) for value in self.backbone.patch_embed.patch_size)
        if patch_size != (14, 14):
            raise ValueError(f"DINO student requires patch size 14, got {patch_size}")

        for block in self.backbone.blocks:
            block.attn.qkv = QVLoRALinear(block.attn.qkv, self.config.lora_rank)

        self.scene_tokens = nn.Parameter(
            torch.empty(
                1,
                self.config.num_cameras,
                self.config.num_scene_tokens,
                self.backbone_feature_dim,
            )
        )
        nn.init.normal_(self.scene_tokens, std=1e-6)
        layer = nn.TransformerEncoderLayer(
            d_model=self.backbone_feature_dim,
            nhead=self.config.fusion_heads,
            dim_feedforward=self.config.fusion_mlp_ratio * self.backbone_feature_dim,
            dropout=self.config.fusion_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(
            layer,
            num_layers=self.config.fusion_layers,
            norm=nn.LayerNorm(self.backbone_feature_dim),
        )
        self.projection = nn.Linear(self.backbone_feature_dim, self.config.feature_dim)
        self.fusion.apply(self._initialize)
        self.projection.apply(self._initialize)

        self.register_buffer(
            "image_mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1), persistent=True
        )
        self.register_buffer(
            "image_std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1), persistent=True
        )
        self.pretrained_source = self.PRETRAINED_SOURCE if pretrained else "random"
        self.pretrained_sha256 = None if resolved_weights is None else _sha256(resolved_weights)
        self.pretrained_revision = resolved_revision
        self.backbone.eval()

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def train(self, mode: bool = True) -> ViTRealPerception:
        super().train(mode)
        # Frozen DINO weights must not acquire stochastic train-mode behaviour.
        # LoRA contains only Linear layers, so keeping this subtree in eval mode
        # does not disable any trainable operation.
        self.backbone.eval()
        return self

    def _normalize_and_pad(self, images: torch.Tensor) -> torch.Tensor:
        images = images.float().div(255.0) if images.dtype == torch.uint8 else images.float()
        images = (images - self.image_mean) / self.image_std
        pad_height = self.padded_image_size[0] - images.shape[-2]
        pad_width = self.padded_image_size[1] - images.shape[-1]
        if pad_height < 0 or pad_width < 0:
            raise ValueError(
                f"images exceed configured size {(self.config.image_height, self.config.image_width)}"
            )
        top, left = pad_height // 2, pad_width // 2
        return F.pad(images, (left, pad_width - left, top, pad_height - top))

    def _forward_backbone(self, images: torch.Tensor, scene_tokens: torch.Tensor) -> torch.Tensor:
        x = self.backbone.patch_embed(images)
        x = self.backbone._pos_embed(x)
        x = torch.cat((scene_tokens, x), dim=1)
        x = self.backbone.patch_drop(x)
        x = self.backbone.norm_pre(x)
        x = self.backbone.blocks(x)
        return self.backbone.norm(x)[:, : self.config.num_scene_tokens]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 5 or images.shape[1:3] != (self.config.num_cameras, 3):
            raise ValueError(
                "images must be [batch, camera, 3, height, width], got " f"{tuple(images.shape)}"
            )
        if tuple(images.shape[-2:]) != (self.config.image_height, self.config.image_width):
            raise ValueError(
                f"images must be {self.config.image_height}x{self.config.image_width}, "
                f"got {tuple(images.shape[-2:])}"
            )
        batch, cameras = images.shape[:2]
        x = self._normalize_and_pad(images.reshape(batch * cameras, 3, *images.shape[-2:]))
        registers = self.scene_tokens.expand(batch, -1, -1, -1).reshape(
            batch * cameras, self.config.num_scene_tokens, self.backbone_feature_dim
        )
        registers = self._forward_backbone(x, registers).reshape(
            batch, cameras * self.config.num_scene_tokens, self.backbone_feature_dim
        )
        fused = self.fusion(registers)
        return self.projection(fused.mean(dim=1))

    @property
    def trainable_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    @property
    def frozen_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if not parameter.requires_grad)


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def load_deployment_bundle(
    path: str | Path,
    device: torch.device | str = "cpu",
) -> nn.Module:
    """Load a self-contained encoder written by ``train_distillation``."""

    bundle = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(bundle, dict) or int(bundle.get("schema_version", 0)) != 1:
        raise ValueError(f"{path} is not a supported real-perception deployment bundle")
    architecture = bundle.get("architecture")
    config_values = dict(bundle.get("model_config", {}))
    if architecture == "dino_vit_small":
        model: nn.Module = ViTRealPerception(
            ViTRealPerceptionConfig(**config_values),
            pretrained=False,
        )
    elif architecture == "convnext_tiny":
        model = RealPerception(RealPerceptionConfig(**config_values), pretrained=False)
    else:
        raise ValueError(f"{path} has unknown architecture {architecture!r}")
    model.load_state_dict(bundle["model"], strict=True)
    model.pretrained_source = str(bundle.get("backbone_initialization", "unknown"))
    model.pretrained_sha256 = bundle.get("backbone_checkpoint_sha256")
    model.pretrained_revision = bundle.get("backbone_revision")
    model.eval()
    return model.to(device)


class DistillationLoss(nn.Module):
    """The paper's adaptation objective: feature alignment plus planning agreement.

    ``lambda * ||E_real - E_sim||^2 + L_plan``.  The feature term is a sum over
    the feature dimension rather than a mean, which is the convention the
    published ``lambda = 1`` balances against; ``mse`` is reported separately as
    the per-dimension mean so it stays comparable across feature widths.

    ``L_plan`` is the KL between the frozen planner's action distributions on the
    teacher and student features.  It is what makes the objective care about
    behaviour: the feature term alone weights every latent direction by its
    coordinate scale, while the planner weights each direction by how much it
    actually moves the wheel.  Dropping it leaves nothing tying the student to
    the closed-loop policy that was distilled in simulation.

    ``target_scale`` optionally divides the residual by a per-dimension standard
    deviation, so no single high-variance coordinate dominates. The prediction
    handed to the planning head is always the raw, unscaled feature.
    """

    def __init__(
        self,
        feature_weight: float = 1.0,
        cosine_weight: float = 0.1,
        plan_weight: float = 1.0,
        plan_head: nn.Module | None = None,
        temperature: float = 1.0,
        target_scale: torch.Tensor | None = None,
    ):
        super().__init__()
        if feature_weight < 0 or cosine_weight < 0 or plan_weight < 0:
            raise ValueError("loss weights must be non-negative")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if plan_weight > 0 and plan_head is None:
            raise ValueError("a non-zero plan weight needs the frozen planning head")
        self.feature_weight = feature_weight
        self.cosine_weight = cosine_weight
        self.plan_weight = plan_weight
        self.temperature = temperature
        self.plan_head = plan_head
        if target_scale is None:
            self.register_buffer("target_scale", None, persistent=False)
        else:
            scale = target_scale.detach().float().clone()
            if (scale <= 0).any():
                raise ValueError("target_scale must be strictly positive")
            self.register_buffer("target_scale", scale, persistent=True)

    @property
    def uses_ego(self) -> bool:
        return self.plan_weight > 0 and self.plan_head is not None

    def _plan_terms(
        self, prediction: torch.Tensor, target: torch.Tensor, ego: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        temperature = self.temperature
        # The planner is a frozen reference, so it runs in full precision even
        # when the student's backbone is under autocast: a KL read off bf16
        # logits is noise at the scale this loss operates on.
        with torch.autocast(device_type=prediction.device.type, enabled=False):
            student_logits = self.plan_head(prediction.float(), ego.float())
            with torch.no_grad():
                teacher_logits = self.plan_head(target.float(), ego.float())
            divergence = prediction.new_zeros(())
            matches = prediction.new_zeros(())
            for student, teacher in zip(student_logits, teacher_logits):
                student_log = F.log_softmax(student / temperature, dim=-1)
                teacher_log = F.log_softmax(teacher / temperature, dim=-1)
                divergence = divergence + F.kl_div(
                    student_log, teacher_log, log_target=True, reduction="batchmean"
                ) * (temperature * temperature)
                matches = matches + (student.argmax(-1) == teacher.argmax(-1)).float().mean()
        return divergence, matches / max(1, len(student_logits))

    def forward(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor,
        ego: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        prediction = prediction.float()
        target = target.float()
        residual = prediction - target
        if self.target_scale is not None:
            residual = residual / self.target_scale
        squared = residual.pow(2)
        feature_loss = squared.sum(dim=-1).mean()
        mse = F.mse_loss(prediction, target)
        cosine = F.cosine_similarity(prediction, target, dim=-1).mean()

        loss = self.feature_weight * feature_loss + self.cosine_weight * (1.0 - cosine)
        metrics = {
            "feature_loss": feature_loss.detach(),
            "mse": mse.detach(),
            "cosine": cosine.detach(),
        }
        if self.uses_ego:
            if ego is None:
                raise ValueError("the planning loss needs the ego observation vector")
            divergence, agreement = self._plan_terms(prediction, target, ego)
            loss = loss + self.plan_weight * divergence
            metrics["plan_kl"] = divergence.detach()
            metrics["plan_agreement"] = agreement.detach()
        metrics["loss"] = loss.detach()
        return loss, metrics
