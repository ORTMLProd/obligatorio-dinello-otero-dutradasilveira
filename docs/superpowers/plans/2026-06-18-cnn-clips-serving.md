# CNN de clips — sub-proyecto 4 (Serving) · Plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Exponer el clip-model + Grad-CAM en la API: `POST /predict/clip` recibe un video, extrae frames (idéntico a training), predice la clase + probabilidades y devuelve los overlays de Grad-CAM en base64.

**Architecture:** Un módulo de serving (`clip_inference.py`) hace video→frames→predicción→overlays reusando `extract_clip_frames`/`predict_clip`/`gradcam_clip`/`overlay_heatmap`. El endpoint nuevo lo expone; el clip-bundle se carga en el `lifespan`. torch/torchvision/opencv pasan a deps de prod; el Dockerfile pre-cachea los pesos ResNet18 y setea `OMP_NUM_THREADS=1` (evita el clash con xgboost en runtime).

**Tech Stack:** FastAPI (UploadFile), PyTorch, OpenCV, base64, pytest.

**Spec:** `docs/superpowers/specs/2026-06-18-cnn-clips-serving-design.md`

**Branch:** `feat/fase-3.5-serving` (ya creada; spec commiteado ahí).

---

## Convenciones del repo

- Correr desde `backend/` con `uv run ...`. Tests: `uv run pytest`. Lint: `uv run ruff check . && uv run ruff format --check .`.
- Código/docstrings inglés; commits español, conventional, sin firma de Claude.
- El `conftest.py` setea `OMP_NUM_THREADS=1` (evita el segfault torch/xgboost) — no tocar.
- Tests con **video sintético** (sin NDA) y `pretrained=False` (sin descargar pesos).
- Antes de cada commit: `uv run pytest -q` + ruff.

---

## File Structure

- Modify: `backend/pyproject.toml` — torch/torchvision/opencv-python a `[project]` (prod); vaciar grupo `cnn`.
- Modify: `backend/Dockerfile` — pre-cache de pesos ResNet18 + `ENV OMP_NUM_THREADS=1` y `TORCH_HOME`.
- Modify: `backend/src/config.py` — `clip_model_dir` + `resolved_clip_model_dir`.
- Modify: `backend/src/api/schemas.py` — `GradcamFrame`, `ClipPredictResponse`.
- Create: `backend/src/serving/__init__.py`, `backend/src/serving/clip_inference.py` — `frames_from_video`, `serve_clip`.
- Create: `backend/src/api/routers/clip_predict.py` — endpoint `/predict/clip`.
- Modify: `backend/src/api/main.py` — cargar clip-bundle en lifespan + incluir router.
- Modify: `docker-compose.yml` — `API_CLIP_MODEL_DIR` en el servicio `api`.
- Create tests: `test_clip_inference.py`, `test_clip_predict.py`, y un assert en `test_config` (si existe) / nuevo.

---

## Task 1: Deps a prod + Dockerfile + config `clip_model_dir`

**Files:**
- Modify: `backend/pyproject.toml`
- Modify: `backend/Dockerfile`
- Modify: `backend/src/config.py`
- Test: `backend/tests/test_config_clip.py`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_config_clip.py`:

```python
from src.config import get_settings


def test_settings_has_clip_model_dir() -> None:
    s = get_settings()
    assert s.clip_model_dir  # default no vacío
    assert s.resolved_clip_model_dir().name == "clips-v1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_config_clip.py -q`
Expected: FAIL `AttributeError: 'Settings' object has no attribute 'clip_model_dir'`.

- [ ] **Step 3: Move torch/torchvision/opencv to prod deps**

In `backend/pyproject.toml`, add to `[project] dependencies` (after `prometheus-client`):

```toml
    # Clip model serving (Fase 3.5): the API runs the CNN + Grad-CAM on uploaded videos.
    "torch>=2.4",
    "torchvision>=0.19",
    "opencv-python>=4.10",
```

Then remove the now-duplicated entries: delete the `cnn` group entirely (it only had torch/torchvision), and remove `"opencv-python>=4.10"` from the `data` group (keep `SoccerNet` there). Run `cd backend && uv sync --all-groups`.

- [ ] **Step 4: Add `clip_model_dir` to `Settings`**

In `backend/src/config.py`, inside `class Settings`, after the `model_dir` field + `resolved_model_dir`:

```python
    # Directory holding the exported clip model bundle (clip_model.pt). In Docker set
    # API_CLIP_MODEL_DIR to the mounted path.
    clip_model_dir: str = "models/clips-v1"

    def resolved_clip_model_dir(self) -> Path:
        path = Path(self.clip_model_dir)
        return path if path.is_absolute() else _REPO_ROOT / path
```

- [ ] **Step 5: Pre-cache ResNet18 weights + OMP in the Dockerfile**

In `backend/Dockerfile`, change the runtime stage so that after `ENV PATH=...` and `USER nonroot`/`WORKDIR /app`, the weights are pre-downloaded. Replace the runtime block from `ENV PATH=...` onward with:

```dockerfile
ENV PATH="/app/.venv/bin:$PATH" \
    TORCH_HOME="/app/.torch" \
    OMP_NUM_THREADS=1
USER nonroot
WORKDIR /app
# Pre-cache the frozen ResNet18 ImageNet weights so the API loads the clip model offline.
RUN python -c "import torchvision; torchvision.models.resnet18(weights='IMAGENET1K_V1')"
EXPOSE 8000
CMD ["fastapi", "run", "src/api/main.py", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 6: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/test_config_clip.py -q`
Expected: PASS (1 passed).

- [ ] **Step 7: Commit**

```bash
git add backend/pyproject.toml backend/uv.lock backend/Dockerfile backend/src/config.py backend/tests/test_config_clip.py
git commit -m "feat: torch a prod + clip_model_dir + Dockerfile (pesos + OMP) para serving"
```

---

## Task 2: Schemas de la respuesta del clip

**Files:**
- Modify: `backend/src/api/schemas.py`
- Test: `backend/tests/test_clip_schemas.py`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_clip_schemas.py`:

```python
from src.api.schemas import ClipPredictResponse, GradcamFrame


def test_clip_response_serializes() -> None:
    resp = ClipPredictResponse(
        predicted_label="corner",
        probabilities={"corner": 0.7, "goal": 0.3},
        model_version="clips-v1-aug",
        gradcam=[GradcamFrame(frame_index=0, image_base64="abc")],
    )
    body = resp.model_dump()
    assert body["predicted_label"] == "corner"
    assert body["gradcam"][0]["frame_index"] == 0
    assert body["gradcam"][0]["image_base64"] == "abc"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_clip_schemas.py -q`
Expected: FAIL `ImportError: cannot import name 'ClipPredictResponse'`.

- [ ] **Step 3: Add the schemas**

Append to `backend/src/api/schemas.py`:

```python
class GradcamFrame(BaseModel):
    """One Grad-CAM overlay: the frame index and a base64-encoded JPG of the overlay."""

    model_config = ConfigDict(extra="forbid")
    frame_index: int
    image_base64: str


class ClipPredictResponse(BaseModel):
    """Prediction for an uploaded video clip: class, probabilities and Grad-CAM overlays."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=())
    predicted_label: str
    probabilities: dict[str, float]
    model_version: str
    gradcam: list[GradcamFrame]
```

(`BaseModel` and `ConfigDict` are already imported at the top of `schemas.py`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/test_clip_schemas.py -q`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add backend/src/api/schemas.py backend/tests/test_clip_schemas.py
git commit -m "feat: schemas ClipPredictResponse + GradcamFrame"
```

---

## Task 3: `frames_from_video` (extracción en serving)

**Files:**
- Create: `backend/src/serving/__init__.py`
- Create: `backend/src/serving/clip_inference.py`
- Test: `backend/tests/test_clip_inference.py`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_clip_inference.py`:

```python
import cv2
import numpy as np

from src.serving.clip_inference import frames_from_video


def _video_bytes(path, n_frames=40, fps=10, size=48) -> bytes:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (size, size))
    for i in range(n_frames):
        writer.write(np.full((size, size, 3), i * 6 % 256, dtype=np.uint8))
    writer.release()
    return path.read_bytes()


def test_frames_from_video_returns_k_frames(tmp_path) -> None:
    data = _video_bytes(tmp_path / "clip.avi")
    frames = frames_from_video(data, k=8, frame_size=32, suffix=".avi")
    assert len(frames) == 8
    assert all(f.shape == (32, 32, 3) and f.dtype == np.uint8 for f in frames)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_clip_inference.py -q`
Expected: FAIL `ModuleNotFoundError: No module named 'src.serving'`.

- [ ] **Step 3: Write minimal implementation**

Create `backend/src/serving/__init__.py` (empty file):

```python
```

Create `backend/src/serving/clip_inference.py`:

```python
"""Serving the clip model on uploaded videos (Fase 3.5).

Extracts K frames from the uploaded video the SAME way as training (reusing
``extract_clip_frames``), runs the clip model and Grad-CAM, and returns the prediction plus
base64 JPG overlays. Frames are processed in memory; the temp video file is deleted (no
images are persisted — data policy).
"""

from __future__ import annotations

import base64
import tempfile
from pathlib import Path

import numpy as np

from src.data.frames import extract_clip_frames, video_duration_ms


def frames_from_video(
    video_bytes: bytes, k: int, frame_size: int, suffix: str = ".mp4"
) -> list[np.ndarray]:
    """Extract K evenly-spaced frames spanning the whole uploaded clip. Deletes the temp file."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(video_bytes)
        tmp_path = Path(handle.name)
    try:
        duration_ms = video_duration_ms(tmp_path)
        return extract_clip_frames(tmp_path, duration_ms // 2, duration_ms, k, frame_size)
    finally:
        tmp_path.unlink(missing_ok=True)


def _jpg_base64(frame_rgb: np.ndarray) -> str:
    import cv2

    bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    ok, buffer = cv2.imencode(".jpg", bgr)
    if not ok:
        raise ValueError("failed to encode overlay as JPG")
    return base64.b64encode(buffer.tobytes()).decode("ascii")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/test_clip_inference.py -q`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git add backend/src/serving/__init__.py backend/src/serving/clip_inference.py backend/tests/test_clip_inference.py
git commit -m "feat: frames_from_video (extraccion en serving, anti-skew)"
```

---

## Task 4: `serve_clip` (predicción + overlays)

**Files:**
- Modify: `backend/src/serving/clip_inference.py`
- Test: `backend/tests/test_clip_inference.py`

- [ ] **Step 1: Write the failing test**

Add to `backend/tests/test_clip_inference.py` (add imports: `import base64`, `import torch`, `from src.models.clip_export import ClipModelMeta`, `from src.models.clip_model import build_clip_model`, `from src.serving.clip_inference import serve_clip`):

```python
def test_serve_clip_returns_prediction_and_overlays(tmp_path) -> None:
    classes = ["background", "card", "corner", "goal", "substitution"]
    meta = ClipModelMeta(
        backbone="resnet18", pooling="mean", classes=classes, k=8, frame_size=32,
        hidden=32, dropout=0.3, normalize_mean=[0.485, 0.456, 0.406],
        normalize_std=[0.229, 0.224, 0.225], model_version="clips-test", metrics={},
    )
    model = build_clip_model(len(classes), hidden=32, pooling="mean", pretrained=False)
    data = _video_bytes(tmp_path / "clip.avi")

    label, proba, overlays = serve_clip(model, meta, data, torch.device("cpu"), suffix=".avi")
    assert label in classes
    assert abs(float(proba.sum()) - 1.0) < 1e-5
    assert len(overlays) == meta.k
    # cada overlay es base64 decodificable y no vacío.
    assert all(len(base64.b64decode(o)) > 0 for o in overlays)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_clip_inference.py -k serve_clip -q`
Expected: FAIL `ImportError: cannot import name 'serve_clip'`.

- [ ] **Step 3: Write minimal implementation**

Add to `backend/src/serving/clip_inference.py` (add these imports at the top: `import torch`, `from src.models.clip_export import ClipModelMeta, predict_clip`, `from src.models.clip_gradcam import gradcam_clip, overlay_heatmap`, `from src.models.clip_model import build_transforms`):

```python
def serve_clip(model, meta: ClipModelMeta, video_bytes: bytes, device, suffix: str = ".mp4"):
    """Full serving path: video → frames → prediction + per-frame Grad-CAM overlays.

    Returns ``(label, proba, overlays)`` where ``overlays`` is a list of base64 JPG strings,
    one per frame, of the Grad-CAM heatmap blended on that frame.
    """
    frames = frames_from_video(video_bytes, meta.k, meta.frame_size, suffix=suffix)
    label, proba = predict_clip(model, meta, frames, device)

    # Build the clip tensor with the same eval transform (anti-skew) for Grad-CAM.
    transform = build_transforms(False, meta.frame_size, meta.normalize_mean, meta.normalize_std)
    clip = torch.stack([transform(f) for f in frames]).unsqueeze(0)
    if device is not None:
        clip = clip.to(device)
    heatmaps, _ = gradcam_clip(model, clip, class_index=meta.classes.index(label))

    overlays = [_jpg_base64(overlay_heatmap(frames[i], heatmaps[i])) for i in range(len(frames))]
    return label, proba, overlays
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/test_clip_inference.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add backend/src/serving/clip_inference.py backend/tests/test_clip_inference.py
git commit -m "feat: serve_clip (prediccion + overlays Grad-CAM base64)"
```

---

## Task 5: Endpoint `/predict/clip` + carga en lifespan + compose

**Files:**
- Create: `backend/src/api/routers/clip_predict.py`
- Modify: `backend/src/api/main.py`
- Modify: `docker-compose.yml`
- Test: `backend/tests/test_clip_predict.py`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_clip_predict.py`:

```python
import cv2
import numpy as np
import torch
from fastapi.testclient import TestClient

from src.api.main import app
from src.models.clip_export import ClipModelMeta
from src.models.clip_model import build_clip_model

CLASSES = ["background", "card", "corner", "goal", "substitution"]


def _video_bytes(path) -> bytes:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (48, 48))
    for i in range(40):
        writer.write(np.full((48, 48, 3), i * 6 % 256, dtype=np.uint8))
    writer.release()
    return path.read_bytes()


def _inject_clip_model() -> None:
    app.state.clip_model = build_clip_model(len(CLASSES), hidden=32, pretrained=False)
    app.state.clip_meta = ClipModelMeta(
        backbone="resnet18", pooling="mean", classes=CLASSES, k=8, frame_size=32,
        hidden=32, dropout=0.3, normalize_mean=[0.485, 0.456, 0.406],
        normalize_std=[0.229, 0.224, 0.225], model_version="clips-test", metrics={},
    )
    app.state.clip_device = torch.device("cpu")


def test_predict_clip_returns_prediction_and_gradcam(tmp_path) -> None:
    _inject_clip_model()
    try:
        data = _video_bytes(tmp_path / "clip.avi")
        resp = TestClient(app).post(
            "/predict/clip", files={"video": ("clip.avi", data, "video/x-msvideo")}
        )
    finally:
        app.state.clip_model = app.state.clip_meta = app.state.clip_device = None
    assert resp.status_code == 200
    body = resp.json()
    assert body["predicted_label"] in CLASSES
    assert len(body["gradcam"]) == 8
    assert set(body["probabilities"]) == set(CLASSES)


def test_predict_clip_503_when_no_model(tmp_path) -> None:
    app.state.clip_model = app.state.clip_meta = app.state.clip_device = None
    data = _video_bytes(tmp_path / "clip.avi")
    resp = TestClient(app).post(
        "/predict/clip", files={"video": ("clip.avi", data, "video/x-msvideo")}
    )
    assert resp.status_code == 503
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && uv run pytest tests/test_clip_predict.py -q`
Expected: FAIL (404 on `/predict/clip` because the router isn't wired yet, or ImportError).

- [ ] **Step 3: Create the router**

Create `backend/src/api/routers/clip_predict.py`:

```python
"""Clip inference endpoint: upload a video → class + probabilities + Grad-CAM (Fase 3.5)."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, UploadFile

from src.api.schemas import ClipPredictResponse, GradcamFrame
from src.monitoring.metrics import record_prediction
from src.serving.clip_inference import serve_clip

router = APIRouter(tags=["inference"])


def _require_clip(request: Request):
    model = getattr(request.app.state, "clip_model", None)
    meta = getattr(request.app.state, "clip_meta", None)
    if model is None or meta is None:
        raise HTTPException(
            status_code=503, detail="No clip model loaded — train and mount models/clips-v1."
        )
    return model, meta, getattr(request.app.state, "clip_device", None)


@router.post("/predict/clip", response_model=ClipPredictResponse)
async def predict_clip_endpoint(request: Request, video: UploadFile) -> ClipPredictResponse:
    """Classify an uploaded video clip and return per-frame Grad-CAM overlays (base64 JPG)."""
    model, meta, device = _require_clip(request)
    data = await video.read()
    if not data:
        raise HTTPException(status_code=422, detail="empty video upload")
    suffix = Path(video.filename or "").suffix or ".mp4"
    try:
        label, proba, overlays = serve_clip(model, meta, data, device, suffix=suffix)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"could not read video: {exc}") from exc
    record_prediction(label, meta.model_version)
    return ClipPredictResponse(
        predicted_label=label,
        probabilities=dict(zip(meta.classes, proba.tolist(), strict=True)),
        model_version=meta.model_version,
        gradcam=[GradcamFrame(frame_index=i, image_base64=b) for i, b in enumerate(overlays)],
    )
```

- [ ] **Step 4: Wire the router + load the clip bundle in `main.py`**

In `backend/src/api/main.py`:

Add imports near the others:
```python
from src.api.routers import clip_predict
from src.models.clip_export import load_clip_bundle
from src.models.clip_model import pick_device
```

In the `lifespan`, after the window-bundle `try/except` block and before `yield`, add:
```python
    # Clip model (Fase 3.5). Loaded best-effort; /predict/clip returns 503 if absent.
    clip_dir = get_settings().resolved_clip_model_dir()
    try:
        device = pick_device()
        app.state.clip_model, app.state.clip_meta = load_clip_bundle(clip_dir, device)
        app.state.clip_device = device
        logger.info("loaded clip model from %s (%s)", clip_dir, app.state.clip_meta.model_version)
    except (FileNotFoundError, OSError):
        app.state.clip_model = app.state.clip_meta = app.state.clip_device = None
        logger.warning("no clip model at %s — /predict/clip will return 503", clip_dir)
```

In `create_app`, after `app.include_router(predict.router)`:
```python
    app.include_router(clip_predict.router)
```

- [ ] **Step 5: Add `API_CLIP_MODEL_DIR` to docker-compose**

In `docker-compose.yml`, in the `api` service `environment:` block, add:
```yaml
      API_CLIP_MODEL_DIR: /app/models/clips-v1
```

- [ ] **Step 6: Run test to verify it passes**

Run: `cd backend && uv run pytest tests/test_clip_predict.py -q`
Expected: PASS (2 passed).

- [ ] **Step 7: Run full suite + lint**

Run: `cd backend && uv run pytest -q && uv run ruff check . && uv run ruff format --check .`
Expected: todo verde (sin segfault gracias al conftest). Si ruff format pide cambios: `uv run ruff format .`.

- [ ] **Step 8: Commit**

```bash
git add backend/src/api/routers/clip_predict.py backend/src/api/main.py docker-compose.yml backend/tests/test_clip_predict.py
git commit -m "feat: endpoint /predict/clip (upload de video -> clase + Grad-CAM)"
```

---

## Task 6: Verificación en el stack docker

> Construye la imagen con torch (pesada, varios minutos la primera vez) y verifica el endpoint real. No es test automatizado.

- [ ] **Step 1: Build + up (api)**

Run: `docker compose up -d --build api`
Expected: la imagen builda (instala torch, pre-cachea ResNet18) y el contenedor queda `healthy`.

- [ ] **Step 2: Probar `/predict/clip` con un video real (local, NDA-safe: no se muestra)**

Run (genera un video de prueba y lo postea; imprime solo metadatos, no la imagen):
```bash
cd backend && uv run python -c "
import cv2, numpy as np, tempfile, requests
p = tempfile.mktemp(suffix='.mp4')
w = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*'mp4v'), 10, (224,224))
for i in range(60): w.write(np.random.randint(0,255,(224,224,3),dtype=np.uint8))
w.release()
r = requests.post('http://localhost:8000/predict/clip', files={'video': ('c.mp4', open(p,'rb'), 'video/mp4')})
d = r.json(); print('status', r.status_code, '| pred', d.get('predicted_label'), '| n_overlays', len(d.get('gradcam', [])))
"
```
Expected: `status 200 | pred <clase> | n_overlays 8`.

- [ ] **Step 3: Verificar que `/predict` (xgboost) sigue vivo (coexistencia sin clash)**

Run: `curl -s http://localhost:8000/model-info | head -c 200`
Expected: responde el modelo de ventana (xgboost) sin que el contenedor haya crasheado → confirma que `OMP_NUM_THREADS=1` evita el clash torch/xgboost en runtime.

- [ ] **Step 4: Bajar el stack**

Run: `docker compose down`

---

## Self-Review (hecho)

- **Cobertura del spec:** deps a prod + Dockerfile pesos/OMP + config (Task 1), schemas (Task 2),
  `frames_from_video` anti-skew (Task 3), `serve_clip` predict+gradcam+base64 (Task 4), endpoint +
  lifespan + compose (Task 5), verificación docker con coexistencia xgboost/torch (Task 6). ✓
- **Placeholders:** ninguno; código/comandos reales. ✓
- **Consistencia de tipos:** `frames_from_video(bytes,k,size,suffix)`, `serve_clip(model,meta,bytes,device,suffix)->(label,proba,overlays)`, `ClipPredictResponse{predicted_label,probabilities,model_version,gradcam:[GradcamFrame{frame_index,image_base64}]}`, `resolved_clip_model_dir`. Consistentes entre tasks/tests. ✓
- **NDA:** tests con video sintético; frames en memoria, temp borrado; nada real commiteado. ✓
- **Out of scope:** frontend (sub-proyecto 5).
