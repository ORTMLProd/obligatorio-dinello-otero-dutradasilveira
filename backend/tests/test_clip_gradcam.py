import numpy as np
import torch

from src.models.clip_gradcam import gradcam_clip
from src.models.clip_model import build_clip_model


def test_gradcam_returns_per_frame_heatmaps() -> None:
    model = build_clip_model(5, hidden=32, pretrained=False)
    clip = torch.randn(1, 8, 3, 64, 64)
    heatmaps, cls = gradcam_clip(model, clip)
    assert heatmaps.shape == (8, 64, 64)
    assert heatmaps.min() >= 0.0 and heatmaps.max() <= 1.0
    assert 0 <= cls < 5


def test_gradcam_uses_given_class() -> None:
    model = build_clip_model(5, hidden=32, pretrained=False)
    _, cls = gradcam_clip(model, torch.randn(1, 8, 3, 64, 64), class_index=2)
    assert cls == 2


def test_gradcam_is_deterministic() -> None:
    model = build_clip_model(3, hidden=16, pretrained=False)
    clip = torch.randn(1, 4, 3, 64, 64)
    h1, _ = gradcam_clip(model, clip)
    h2, _ = gradcam_clip(model, clip)
    assert np.allclose(h1, h2)


def test_backbone_stays_frozen_after_gradcam() -> None:
    model = build_clip_model(3, hidden=16, pretrained=False)
    gradcam_clip(model, torch.randn(1, 4, 3, 64, 64))
    assert all(not p.requires_grad for p in model.backbone.parameters())
