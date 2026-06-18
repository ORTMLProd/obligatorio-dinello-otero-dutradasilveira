# Diseño — Fase 3.5, sub-proyecto 4: Serving (upload de video)

Fecha: 2026-06-18 · Estado: aprobado para planificar

## Contexto

Sub-proyecto 4 de la Fase 3.5: **el que conecta el video con el flujo**. Expone el clip-model
(sub-proyecto 2) + Grad-CAM (sub-proyecto 3) en la API: el usuario **sube un video** y recibe la
clase predicha, las probabilidades y los overlays de Grad-CAM. Es el corazón del objetivo
("subir un clip → el modelo dice si es gol/corner/...").

## Decisiones (acordadas)

- **Misma imagen de la API + torch:** `torch`/`torchvision`/`opencv-python` pasan a deps de **prod**;
  el endpoint vive en la misma API. Se setea `OMP_NUM_THREADS=1` en el contenedor para evitar el
  clash de OpenMP entre torch y xgboost (la API ya carga xgboost para el modelo de ventana).
- **Pesos del backbone pre-descargados en el Dockerfile** (un `RUN` cachea ResNet18 ImageNet); el
  bundle sigue chico (~533 KB).
- **Extracción de frames idéntica a training** (anti-skew, invariante 3): se reusa
  `src.data.frames.extract_clip_frames`.
- **Respuesta con los 8 overlays** (uno por frame), evitando el problema de "elegir el frame
  saliente"; el frontend (sub-proyecto 5) los muestra.

## Arquitectura y componentes

```
POST /predict/clip  (multipart: archivo de video)
        │
        ▼
src/serving/clip_inference.py
  ├─ frames_from_video(video_bytes, k, frame_size) -> list[np.ndarray]
  │     (temp file → extract_clip_frames(center=dur/2, clip_ms=dur, k, size) → borra temp)
  └─ serve_clip(model, meta, video_bytes, device) -> (label, proba, overlays)
        (frames → predict_clip → gradcam_clip → overlay_heatmap por frame → JPG base64)
        │
        ▼
ClipPredictResponse { predicted_label, probabilities, model_version, gradcam: [GradcamFrame] }
```

### `clip_inference.py` (nuevo, testeable)
- `frames_from_video(video_bytes, k, frame_size) -> list[np.ndarray]`: escribe los bytes a un
  archivo temporal (`tempfile`), llama `extract_clip_frames(tmp, center_ms=dur/2, clip_ms=dur, k,
  frame_size)` (muestrea K frames a lo largo de **todo** el clip subido, hasta ~30s), y **borra el
  temp** (no se persisten imágenes). `dur` = `video_duration_ms(tmp)`.
- `serve_clip(model, meta, video_bytes, device) -> (label, proba, overlays)`:
  1. `frames = frames_from_video(...)` (numpy HWC uint8 RGB, K frames).
  2. `label, proba = predict_clip(model, meta, frames, device)`.
  3. `clip_tensor` para Grad-CAM: se arma con el transform de eval (mismo de `predict_clip`) →
     `heatmaps, _ = gradcam_clip(model, clip_tensor, class_index=meta.classes.index(label))`.
  4. `overlays = [overlay_heatmap(frames[i], heatmaps[i]) for i in range(K)]` → cada uno a **JPG
     base64** (`cv2.imencode('.jpg', bgr)` → `base64`).
  - Devuelve `(label, proba, overlays_b64)`.

### Schemas (`src/api/schemas.py`)
```python
class GradcamFrame(BaseModel): frame_index: int; image_base64: str
class ClipPredictResponse(BaseModel):
    predicted_label: str
    probabilities: dict[str, float]
    model_version: str
    gradcam: list[GradcamFrame]
```

### Endpoint (`src/api/routers/clip_predict.py`)
- `POST /predict/clip` con `UploadFile` (multipart). Lee los bytes, valida que no esté vacío,
  obtiene `model`/`meta`/`device` de `app.state` (503 si no hay clip-model), llama `serve_clip`,
  arma `ClipPredictResponse`. Reusa el contador Prometheus (`record_prediction(label, version)`).
- Manejo de errores: video ilegible / sin frames → 422 con mensaje claro.

### Carga del modelo (`src/api/main.py`)
- En el `lifespan`: cargar el clip-bundle desde `clip_model_dir` (config) a
  `app.state.clip_model`/`clip_meta`/`clip_device` (best-effort: si no hay bundle, el endpoint
  devuelve 503, igual que el modelo de ventana). Incluir el router nuevo.

### Config (`src/config.py`)
- Setting `clip_model_dir` (default `models/clips-v1`), con `resolved_clip_model_dir()`.

### Dependencias y Docker
- `pyproject.toml`: `torch`, `torchvision`, `opencv-python` a `[project] dependencies` (prod).
- `backend/Dockerfile`: tras instalar deps, `RUN python -c "import torchvision; \
  torchvision.models.resnet18(weights='IMAGENET1K_V1')"` para cachear pesos; `ENV OMP_NUM_THREADS=1`.
- `docker-compose.yml`: el servicio `api` monta `./models:/app/models:ro` (ya lo hace) y suma
  `OMP_NUM_THREADS: "1"`. El clip-bundle vive en `models/clips-v1`.

## Testing
- `clip_inference.frames_from_video`: con un **video sintético** (`cv2.VideoWriter`), devuelve K
  arrays `(frame_size, frame_size, 3)` uint8; el temp se borra (no queda archivo).
- `clip_inference.serve_clip`: con un `build_clip_model(pretrained=False)` + `ClipModelMeta` chico
  y un video sintético, devuelve `(label ∈ classes, proba suma≈1, overlays con len==K)` y cada
  overlay es base64 decodificable a una imagen.
- Endpoint `/predict/clip` (TestClient): inyectar un clip-model tiny en `app.state`; `POST` un
  video sintético (multipart) → 200 + schema correcto (gradcam con K items). `503` si no hay
  clip-model. `422` si el archivo no es un video válido.
- La suite no debe segfaultear (conftest ya setea `OMP_NUM_THREADS=1`).

## Out of scope
- Frontend que consume `/predict/clip` (sub-proyecto 5).
- Async/colas para videos grandes (v0 es síncrono); límites de tamaño de upload finos.

## Riesgos
- **Imagen pesada (torch ~1–1.5 GB):** aceptable para la demo local; documentado.
- **Clash OpenMP en runtime** (torch + xgboost en el mismo proceso): mitigado con `OMP_NUM_THREADS=1`
  en el contenedor. Verificar que `/predict` (xgboost) y `/predict/clip` (torch) coexisten sin crash.
- **Pesos del backbone:** pre-cacheados en el build; si el build no tuviera internet, fallaría —
  documentado.
- **Tamaño/duración del video subido:** se muestrean K frames sea cual sea la duración; videos muy
  largos solo afectan el tiempo de extracción.
