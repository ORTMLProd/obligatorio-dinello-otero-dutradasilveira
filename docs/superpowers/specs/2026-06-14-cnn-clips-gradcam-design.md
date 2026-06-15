# Diseño — Fase 3.5, sub-proyecto 3: Grad-CAM (explicabilidad visual)

Fecha: 2026-06-14 · Estado: aprobado para planificar

## Contexto

Sub-proyecto 3 de la Fase 3.5. El sub-proyecto 2 dejó el clip-model (ResNet18 congelada +
mean-pool + cabeza) y su bundle. Acá agregamos **Grad-CAM**: el mapa de calor que muestra *qué
región* de cada frame sostuvo la predicción de la clase. Cierra el electivo de **explicabilidad**
del lado visual (complementa al SHAP tabular de la Fase 3.2).

## Decisiones (acordadas)

- **Por frame:** se calcula el Grad-CAM para **los 8 frames** del clip; el core devuelve los K
  mapas y la selección (mostrar el más saliente, una tira, etc.) queda downstream (serving/frontend).
- **Capa target:** `layer4` de ResNet18 (último bloque conv, 512 canales, 7×7 para entrada 224).
- **NDA:** los overlays van sobre frames reales → son **contenido NDA**. Este sub-proyecto entrega
  **solo código + tests con frames sintéticos**; la visualización real (overlay sobre un frame de un
  clip) ocurre en serving/frontend, **local**, para quien tenga el NDA. No se commitean overlays ni
  notebooks con frames reales.

## Arquitectura y componentes

```
src/models/clip_gradcam.py
  ├─ gradcam_clip(model, clip, class_index=None) -> (heatmaps, class_index)
  └─ overlay_heatmap(frame_rgb, heatmap, alpha=0.45) -> rgb_overlay
```

### `gradcam_clip(model, clip, class_index=None)`
- Entrada: `model` (ClipClassifier del sub-proyecto 2) y `clip` tensor `(1, K, 3, H, W)` (o `(K, ...)`
  que se expande a batch 1). `class_index` opcional (default: la clase predicha por el modelo).
- **Forward dedicado con gradientes:** el forward de entrenamiento del modelo corre el backbone bajo
  `torch.no_grad()`, así que Grad-CAM **replica el forward mínimamente** reusando `model.backbone` y
  `model.head` (mismos módulos, single source) pero **con el grafo activo**:
  1. Hook forward sobre `model.backbone.layer4` que guarda las activaciones `A` y registra un hook
     sobre `A` para capturar el gradiente `dL/dA` en el backward.
  2. `feats = model.backbone(frames)` (con grad) → `(B*K, 512)` → reshape `(B, K, 512)` → pooling
     (mean/max según `model.pooling`) → `model.head(pooled)` → logits.
  3. `class_index` = argmax de los logits si no se pasó. Backward del logit de esa clase.
  4. Por cada frame: pesos `w_c = GAP_espacial(dL/dA[frame])` (promedio por canal), mapa
     `relu(Σ_c w_c · A[frame, c])` → normalizado a [0,1] → **upsample bilineal** a `(H, W)`.
- **Pesos congelados:** el gradiente fluye hacia las **activaciones** aunque los pesos del backbone
  tengan `requires_grad=False` (el grafo se construye igual fuera de `no_grad`). No se reentrena nada.
- Salida: `heatmaps` numpy `(K, H, W)` en [0,1], y el `class_index` usado.

### `overlay_heatmap(frame_rgb, heatmap, alpha=0.45)`
- Mezcla el `heatmap` (colormap tipo "jet" vía OpenCV) sobre `frame_rgb` (numpy HWC uint8 RGB) y
  devuelve la imagen RGB resultante (uint8). Resize del heatmap al tamaño del frame si difieren.

## Testing (frames sintéticos, sin NDA)

- `gradcam_clip` con un `build_clip_model(..., pretrained=False)` y un clip random `(1, K, 3, 64, 64)`:
  devuelve `heatmaps` de forma `(K, 64, 64)`, dtype float, valores en [0,1]; y un `class_index` válido.
- **Determinismo:** misma entrada → mismos heatmaps (modelo en eval, sin dropout activo).
- `class_index` explícito: pedir una clase fija la usa (no la argmax).
- `overlay_heatmap`: dado un frame `(64,64,3)` uint8 y un heatmap `(64,64)`, devuelve `(64,64,3)` uint8.
- El modelo NO debe quedar con gradientes/estado residual que rompa un forward posterior (se limpian
  los hooks; se puede verificar que `model.backbone` sigue con `requires_grad=False`).

## Out of scope (otros sub-proyectos)

- Exponer Grad-CAM en la API (sub-proyecto 4 — endpoint que devuelve el/los overlay(s)).
- Mostrarlo en el frontend (sub-proyecto 5 — el slot ya reservado en la UI de 3.3).
- Notebook de visualización con frames reales (NDA): si se quisiera, sería local y sin commitear outputs.

## Riesgos

- **Hooks y limpieza:** registrar/quitar los hooks de forward/backward de forma segura (try/finally)
  para no dejar estado que afecte inferencias posteriores.
- **Grad a través de backbone congelado:** asegurarse de NO envolver el forward de Grad-CAM en
  `torch.no_grad()` (si no, no hay grafo). Los `requires_grad=False` de los pesos no impiden el grad
  hacia las activaciones.
- **MPS:** algunas ops de gradiente pueden ser más estables en CPU; el core es device-agnóstico y los
  tests corren en CPU. El clash OpenMP torch/xgboost ya está mitigado (`OMP_NUM_THREADS=1` en conftest).
