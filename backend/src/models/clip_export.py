"""Serialize / load the clip model bundle and run inference from it (Fase 3.5).

The bundle stores only the trained head ``state_dict`` plus metadata (the frozen ResNet18
backbone is rebuilt from torchvision's ImageNet weights on load). The eval transform is
reconstructed from the serialized normalize stats + frame_size, so training and serving
preprocess identically (invariant 3). ``predict_clip`` is the single shared inference path.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from src.models.clip_model import build_clip_model, build_transforms

BUNDLE_FILE = "clip_model.pt"


@dataclass
class ClipModelMeta:
    backbone: str
    pooling: str
    classes: list[str]
    k: int
    frame_size: int
    hidden: int
    dropout: float
    normalize_mean: list[float]
    normalize_std: list[float]
    model_version: str
    metrics: dict


def save_clip_bundle(model, meta: ClipModelMeta, model_dir: Path) -> Path:
    """Save the head state_dict + metadata to ``model_dir/clip_model.pt``."""
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / BUNDLE_FILE
    torch.save({"head_state_dict": model.head.state_dict(), "meta": asdict(meta)}, path)
    return path


def load_clip_bundle(model_dir: Path, device: torch.device | None = None):
    """Rebuild the model (frozen backbone + trained head) from the bundle. Returns (model, meta)."""
    payload = torch.load(model_dir / BUNDLE_FILE, map_location="cpu", weights_only=False)
    meta = ClipModelMeta(**payload["meta"])
    model = build_clip_model(
        len(meta.classes), meta.hidden, meta.dropout, meta.pooling, meta.backbone
    )
    model.head.load_state_dict(payload["head_state_dict"])
    model.eval()
    if device is not None:
        model.to(device)
    return model, meta


def predict_clip(model, meta: ClipModelMeta, frames, device: torch.device | None = None):
    """Predict one clip. ``frames``: iterable of K numpy HWC uint8 RGB arrays.

    Returns ``(label, proba)`` where proba is a softmax vector ordered as ``meta.classes``.
    """
    transform = build_transforms(False, meta.frame_size, meta.normalize_mean, meta.normalize_std)
    clip = torch.stack([transform(f) for f in frames]).unsqueeze(0)  # (1, K, 3, H, W)
    if device is not None:
        clip = clip.to(device)
    model.eval()
    with torch.no_grad():
        proba = torch.softmax(model(clip), dim=1)[0].cpu().numpy().astype(np.float64)
    return meta.classes[int(proba.argmax())], proba
