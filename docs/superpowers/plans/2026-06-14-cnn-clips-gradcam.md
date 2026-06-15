# CNN de clips — sub-proyecto 3 (Grad-CAM) · Plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Calcular mapas de calor Grad-CAM por frame para el clip-model (dónde mira el modelo) y una utilidad para superponerlos sobre el frame.

**Architecture:** Un módulo `clip_gradcam.py` con `gradcam_clip` (forward dedicado con gradientes, hook sobre `layer4`, Grad-CAM por cada uno de los K frames) y `overlay_heatmap` (mezcla colormap sobre el frame). Reusa el `ClipClassifier` del sub-proyecto 2 sin reentrenar.

**Tech Stack:** PyTorch (autograd, hooks, F.interpolate), NumPy, OpenCV (overlay), pytest.

**Spec:** `docs/superpowers/specs/2026-06-14-cnn-clips-gradcam-design.md`

**Branch:** `feat/fase-3.5-gradcam` (ya creada; spec commiteado ahí).

---

## Convenciones del repo

- Correr desde `backend/` con `uv run ...`. Tests: `uv run pytest`. Lint: `uv run ruff check . && uv run ruff format --check .`.
- Código/docstrings inglés; commits español, conventional, sin firma de Claude.
- El `conftest.py` ya setea `OMP_NUM_THREADS=1` (evita el segfault torch/xgboost en Apple Silicon) — no tocar.
- Los tests usan `build_clip_model(..., pretrained=False)` (no descarga pesos) y frames sintéticos (sin NDA).

## File Structure

- Create: `backend/src/models/clip_gradcam.py` — `gradcam_clip` + `overlay_heatmap`.
- Create: `backend/tests/test_clip_gradcam.py`.

---

## Task 1: `gradcam_clip` — heatmaps por frame

**Files:**
- Create: `backend/src/models/clip_gradcam.py`
- Test: `backend/tests/test_clip_gradcam.py`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_clip_gradcam.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_clip_gradcam.py -q`
Expected: FAIL `ModuleNotFoundError: No module named 'src.models.clip_gradcam'`.

- [ ] **Step 3: Write minimal implementation**

Create `backend/src/models/clip_gradcam.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/test_clip_gradcam.py -q`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add backend/src/models/clip_gradcam.py backend/tests/test_clip_gradcam.py
git commit -m "feat: Grad-CAM por frame del clip model (forward dedicado, hook layer4)"
```

---

## Task 2: `overlay_heatmap` — superponer el mapa sobre el frame

**Files:**
- Modify: `backend/src/models/clip_gradcam.py` (add `overlay_heatmap`)
- Test: `backend/tests/test_clip_gradcam.py` (add tests)

- [ ] **Step 1: Write the failing test**

Add to `backend/tests/test_clip_gradcam.py` (add `from src.models.clip_gradcam import overlay_heatmap` to the imports):

```python
def test_overlay_returns_rgb_image() -> None:
    frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    heatmap = np.random.rand(64, 64).astype(np.float32)
    out = overlay_heatmap(frame, heatmap)
    assert out.shape == (64, 64, 3) and out.dtype == np.uint8


def test_overlay_resizes_mismatched_heatmap() -> None:
    frame = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    heatmap = np.random.rand(7, 7).astype(np.float32)  # smaller than the frame
    out = overlay_heatmap(frame, heatmap)
    assert out.shape == (64, 64, 3)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_clip_gradcam.py -k overlay -q`
Expected: FAIL `ImportError: cannot import name 'overlay_heatmap'`.

- [ ] **Step 3: Write minimal implementation**

Add to `backend/src/models/clip_gradcam.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/test_clip_gradcam.py -q`
Expected: PASS (6 passed).

- [ ] **Step 5: Run full suite + lint**

Run: `cd backend && uv run pytest -q && uv run ruff check . && uv run ruff format --check .`
Expected: todo verde (la suite no debe segfaultear gracias al `OMP_NUM_THREADS=1` del conftest). Si ruff format pide cambios: `uv run ruff format .`.

- [ ] **Step 6: Commit**

```bash
git add backend/src/models/clip_gradcam.py backend/tests/test_clip_gradcam.py
git commit -m "feat: overlay_heatmap para Grad-CAM (colormap sobre el frame)"
```

---

## Self-Review (hecho)

- **Cobertura del spec:** `gradcam_clip` por frame con forward dedicado + hook layer4 (Task 1), `overlay_heatmap` (Task 2). Determinismo y backbone-sigue-congelado testeados. NDA: tests con frames sintéticos, nada real commiteado. ✓
- **Detalle técnico crítico:** `frames.requires_grad_(True)` para que el grafo llegue a `layer4` a través del backbone congelado (si no, `backward` falla). Cubierto en el código. ✓
- **Placeholders:** ninguno; código/comandos reales. ✓
- **Consistencia de tipos:** `gradcam_clip(model, clip, class_index) -> (heatmaps, class_index)`, `overlay_heatmap(frame_rgb, heatmap, alpha) -> np.ndarray`. Consistentes entre tasks y tests. ✓
- **Out of scope:** API (4), frontend (5).
