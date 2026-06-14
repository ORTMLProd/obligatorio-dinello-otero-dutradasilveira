"""Multi-frame clip classifier: frozen ResNet18 backbone + mean-pool + MLP head.

Visual-only (Fase 3.5). The backbone is a frozen ImageNet ResNet18 used as a per-frame
feature extractor; only the head trains. ``build_transforms`` produces the train (augmented)
and eval (deterministic) image transforms; the eval transform is serialized with the bundle
so training and serving preprocess identically (invariant 3).

Implementation note: ``torchvision.models`` is imported lazily (inside functions that use it)
to avoid triggering Metal/MPS dispatch-queue initialization at module import time, which
causes a segfault when XGBoost's OpenMP threads are loaded in the same pytest process on
macOS ARM (Apple Silicon). ``torchvision.transforms`` is safe to import at module level.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision import transforms


def pick_device() -> torch.device:
    """Best available device: MPS (Apple Silicon) > CUDA > CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_transforms(augment: bool, frame_size: int, mean: list[float], std: list[float]):
    """Image transform for a single frame (numpy HWC uint8 RGB → normalized CHW tensor)."""
    normalize = transforms.Normalize(mean=mean, std=std)
    if augment:
        return transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.RandomResizedCrop(frame_size, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize(frame_size),
            transforms.CenterCrop(frame_size),
            transforms.ToTensor(),
            normalize,
        ]
    )


class ClipClassifier(nn.Module):
    """Frozen ResNet18 per frame → temporal pool over K frames → MLP head → logits."""

    def __init__(
        self,
        n_classes: int,
        hidden: int = 256,
        dropout: float = 0.3,
        pooling: str = "mean",
        pretrained: bool = True,
    ) -> None:
        # Lazy import: keeps torchvision.models out of the module namespace until
        # a model is actually instantiated (avoids MPS init at collection time).
        from torchvision import models
        from torchvision.models import ResNet18_Weights

        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
        self.feature_dim = backbone.fc.in_features  # 512
        backbone.fc = nn.Identity()
        for p in backbone.parameters():
            p.requires_grad_(False)
        backbone.eval()
        self.backbone = backbone
        self.pooling = pooling
        self.head = nn.Sequential(
            nn.Linear(self.feature_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def train(self, mode: bool = True) -> ClipClassifier:
        # Keep the frozen backbone in eval mode so BatchNorm running stats stay fixed.
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        # clips: (B, K, 3, H, W)
        b, k = clips.shape[0], clips.shape[1]
        frames = clips.reshape(b * k, *clips.shape[2:])
        with torch.no_grad():
            feats = self.backbone(frames)  # (B*K, 512)
        feats = feats.reshape(b, k, -1)
        pooled = feats.max(dim=1).values if self.pooling == "max" else feats.mean(dim=1)
        return self.head(pooled)


def build_clip_model(
    n_classes: int,
    hidden: int = 256,
    dropout: float = 0.3,
    pooling: str = "mean",
    backbone: str = "resnet18",
    pretrained: bool = True,
) -> ClipClassifier:
    """Build the clip classifier. Only ``resnet18`` is supported for now."""
    if backbone != "resnet18":
        raise ValueError(f"unsupported backbone {backbone!r}; only 'resnet18'")
    return ClipClassifier(n_classes, hidden, dropout, pooling, pretrained)
