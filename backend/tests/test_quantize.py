import torch

from src.models.clip_model import build_clip_model
from src.models.quantize import _clip_logits, _state_dict_bytes, quantize_backbone


def test_quantize_backbone_runs_and_shrinks() -> None:
    """Static PTQ must produce a working int8 backbone that is smaller than FP32."""
    model = build_clip_model(5, hidden=32, pretrained=False)
    model.eval()
    calib = [torch.rand(8, 3, 64, 64) for _ in range(2)]

    qbackbone = quantize_backbone(model, 64, calib)

    # The quantized backbone still yields valid clip logits (1, K, 3, H, W) -> (1, n_classes).
    logits = _clip_logits(qbackbone, model.head, torch.rand(1, 8, 3, 64, 64))
    assert logits.shape == (1, 5)
    # int8 weights must be smaller than the FP32 backbone.
    assert _state_dict_bytes(qbackbone) < _state_dict_bytes(model.backbone)
