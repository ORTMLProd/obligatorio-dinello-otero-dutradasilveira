"""Grad-CAM for the clip classifier (Fase 3.5 — visual explainability).

Highlights, per frame, which region of the image supported the predicted class. The clip
model's training forward runs the frozen backbone under ``no_grad``; Grad-CAM needs the graph,
so this replicates the forward (reusing ``model.backbone``/``model.head``) with grad enabled and
a hook on ``layer4`` to capture activations and their gradients. The input frames are marked
``requires_grad`` so the graph reaches ``layer4`` even though the backbone weights are frozen —
nothing is retrained.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def gradcam_clip(model, clip: torch.Tensor, class_index: int | None = None):
    """Per-frame Grad-CAM heatmaps for one clip.

    Args:
        model: a ``ClipClassifier`` (frozen ResNet18 backbone + head).
        clip: tensor ``(1, K, 3, H, W)`` or ``(K, 3, H, W)``.
        class_index: target class; defaults to the model's predicted class.

    Returns:
        ``(heatmaps, class_index)`` where ``heatmaps`` is a numpy array ``(K, H, W)`` in [0, 1].
    """
    model.eval()
    if clip.dim() == 4:
        clip = clip.unsqueeze(0)
    b, k = clip.shape[0], clip.shape[1]
    h, w = clip.shape[3], clip.shape[4]
    # requires_grad on the input so the graph reaches layer4 through the frozen backbone.
    frames = clip.reshape(b * k, *clip.shape[2:]).detach().requires_grad_(True)

    captured: dict[str, torch.Tensor] = {}

    def forward_hook(_module, _inputs, output):
        captured["activations"] = output
        output.register_hook(lambda grad: captured.__setitem__("gradients", grad))

    handle = model.backbone.layer4.register_forward_hook(forward_hook)
    try:
        with torch.enable_grad():
            feats = model.backbone(frames).reshape(b, k, -1)
            pooled = feats.max(dim=1).values if model.pooling == "max" else feats.mean(dim=1)
            logits = model.head(pooled)
            if class_index is None:
                class_index = int(logits.argmax(dim=1)[0].item())
            model.zero_grad(set_to_none=True)
            logits[0, class_index].backward()
    finally:
        handle.remove()

    activations = captured["activations"]  # (B*K, C, h4, w4)
    gradients = captured["gradients"]  # (B*K, C, h4, w4)
    weights = gradients.mean(dim=(2, 3), keepdim=True)  # (B*K, C, 1, 1)
    cam = F.relu((weights * activations).sum(dim=1, keepdim=True))  # (B*K, 1, h4, w4)
    cam = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)
    cam = cam.reshape(k, h, w).detach().cpu().numpy()

    heatmaps = np.zeros_like(cam)
    for i in range(k):
        lo, hi = cam[i].min(), cam[i].max()
        heatmaps[i] = (cam[i] - lo) / (hi - lo) if hi > lo else np.zeros_like(cam[i])
    model.zero_grad(set_to_none=True)
    return heatmaps, class_index


def overlay_heatmap(frame_rgb: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a [0, 1] ``heatmap`` (jet colormap) over an RGB frame. Returns uint8 RGB.

    The heatmap is resized to the frame size if they differ. ``alpha`` is the heatmap weight.
    """
    import cv2  # local import: only the overlay utility needs OpenCV

    height, width = frame_rgb.shape[:2]
    hm = heatmap
    if hm.shape[:2] != (height, width):
        hm = cv2.resize(hm, (width, height))
    hm_uint8 = np.uint8(255 * np.clip(hm, 0.0, 1.0))
    colored_bgr = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
    colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
    blended = alpha * colored_rgb.astype(np.float32) + (1 - alpha) * frame_rgb.astype(np.float32)
    return blended.clip(0, 255).astype(np.uint8)
