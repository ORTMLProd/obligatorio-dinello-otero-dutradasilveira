import numpy as np
import torch

from src.models.clip_model import build_clip_model, build_transforms, pick_device

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def test_pick_device_returns_torch_device() -> None:
    assert isinstance(pick_device(), torch.device)


def test_eval_transform_is_deterministic() -> None:
    frame = (np.arange(64 * 64 * 3, dtype=np.uint8) % 255).reshape(64, 64, 3)
    t = build_transforms(augment=False, frame_size=32, mean=MEAN, std=STD)
    a, b = t(frame), t(frame)
    assert a.shape == (3, 32, 32) and a.dtype == torch.float32
    assert torch.equal(a, b)


def test_augment_transform_outputs_right_shape() -> None:
    frame = (np.arange(64 * 64 * 3, dtype=np.uint8) % 255).reshape(64, 64, 3)
    t = build_transforms(augment=True, frame_size=32, mean=MEAN, std=STD)
    out = t(frame)
    assert out.shape == (3, 32, 32) and out.dtype == torch.float32


def test_forward_shape_and_frozen_backbone() -> None:
    # pretrained=False evita descargar pesos ImageNet en el unit test.
    model = build_clip_model(n_classes=5, hidden=32, pretrained=False)
    clips = torch.randn(2, 8, 3, 224, 224)
    out = model(clips)
    assert out.shape == (2, 5)
    # backbone congelado, cabeza entrenable.
    assert all(not p.requires_grad for p in model.backbone.parameters())
    assert any(p.requires_grad for p in model.head.parameters())


def test_max_pooling_keeps_shape() -> None:
    model = build_clip_model(n_classes=3, hidden=16, pooling="max", pretrained=False)
    out = model(torch.randn(1, 4, 3, 224, 224))
    assert out.shape == (1, 3)
