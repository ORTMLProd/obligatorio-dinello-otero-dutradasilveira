# Diseño — Fase 3.5, sub-proyecto 2: Modelo (CNN multi-frame de clips)

Fecha: 2026-06-14 · Estado: aprobado para planificar

## Contexto

Sub-proyecto 2 de la Fase 3.5 (integrar video en el flujo). El sub-proyecto 1 dejó un dataset
de **clips** (`data/processed/clips_manifest.parquet`, 1140 clips × 8 frames 224×224, splits por
`game_id`). Acá entrenamos el **clasificador multi-frame** que predice la clase del clip
(`goal`, `corner`, `card`, `substitution`, `background`).

**Visual-only (decisión explícita).** El modelo aprende **solo de los píxeles** de los frames; no
usa las columnas tabulares del manifest. Dos razones: (1) un clip subido por un usuario no trae
contexto de partido (minuto, score, etc.), así que fusionar tabular no serviría al flujo objetivo;
(2) varias features tabulares son **artefactos de construcción del dataset** — al muestrear
`background` les fijamos `team_is_home=-1` y `visible=0`, lo que separa background de eventos casi
perfectamente sin ser señal real (esto explica el dominio de `team_is_home` en el SHAP de la Fase
3.2). El CNN tiene que ganarse la predicción mirando la jugada.

## Decisiones (acordadas)

- **Backbone:** ResNet18 pre-entrenada en ImageNet, **congelada** (sin gradientes). Corre **en
  vivo** cada época (no se cachean embeddings) para permitir data augmentation de imágenes.
- **Forma del modelo:** por clip de K frames → backbone por frame → embedding 512-dim →
  **pooling temporal `mean`** sobre K → **cabeza MLP** (512→256→n_clases, ReLU, dropout 0.3) →
  logits. Solo la cabeza entrena.
- **Augmentation + impacto:** se entrena dos veces (sin/con augmentation) y se reporta el delta de
  macro-F1 (sub-técnica del electivo de optimización, midiendo impacto).
- **Desbalance:** cross-entropy **ponderada** por la distribución de clases del train (invariante 5).
- **Device:** auto (MPS/CUDA/CPU), seedeado (invariante 6).
- **Selección de modelo:** early stopping por **macro-F1 de validación**; test se reporta una vez.

## Arquitectura y componentes

```
configs/train_clips.yaml  (config tipada nueva)
        │
        ▼
src/data/clips_dataset.py   ── Dataset PyTorch: lee frame_paths del manifest, aplica transforms,
                               devuelve (tensor (K,3,224,224), label_idx). Transforms train/eval.
        │
        ▼
src/models/clip_model.py    ── build_clip_model(): nn.Module (backbone congelado + mean-pool +
                               cabeza MLP). build_transforms(augment). pick_device().
        │
        ▼
src/models/train_clips.py   ── loop de entrenamiento (head-only), eval por split, early stopping,
                               logging a MLflow (params, métricas por clase, matriz de confusión),
                               corre baseline (no-aug) vs augmented, exporta el mejor bundle.
        │
        ▼
src/models/clip_export.py   ── ClipModelBundle (serializa head_state_dict + arch + transforms de
                               eval + classes + pooling + k + frame_size + metrics). save/load.
                               predict_clip(model, frames) → (label, proba). Fuente única de
                               inferencia compartida con el serving (invariante 3).
        │
        ▼
models/clips-v1/clip_model.pt   (bundle, gitignored)
report/metrics/confusion_clips.png + MLflow runs
```

### Modelo (`clip_model.py`)
- `build_clip_model(n_classes, backbone="resnet18", hidden=256, dropout=0.3, pooling="mean")`:
  - backbone: `torchvision.models.resnet18(weights=IMAGENET1K_V1)`, sin la FC final (salida
    512-dim del avgpool); `requires_grad_(False)` y `eval()`. Se conserva accesible para el
    Grad-CAM del sub-proyecto 3.
  - forward `(B, K, 3, H, W)` → reshape `(B*K, ...)` → backbone bajo `torch.no_grad()` →
    `(B*K, 512)` → reshape `(B, K, 512)` → mean sobre K → cabeza → `(B, n_classes)`.
- `build_transforms(augment, frame_size, mean, std)`:
  - train+augment: RandomResizedCrop(scale 0.8–1.0) + RandomHorizontalFlip + ColorJitter +
    ToTensor + Normalize(ImageNet).
  - eval / no-augment: Resize/CenterCrop a `frame_size` + ToTensor + Normalize.
- `pick_device()`: MPS si disponible, si no CUDA, si no CPU.

### Dataset (`clips_dataset.py`)
- `ClipsDataset(manifest_df, processed_dir, classes, transform)`: por fila, lee los K `frame_paths`
  (relativos a `processed_dir`), aplica `transform` a cada frame, los apila a `(K, 3, H, W)`,
  devuelve `(clip_tensor, class_idx)`. `classes` ordenado y estable.

### Entrenamiento (`train_clips.py`)
- Carga el `clips_manifest.parquet`, separa por columna `split`. `classes = sorted(unique labels)`.
- Para cada modo en (no-aug, aug): construye DataLoaders, modelo, optimizer (Adam sobre params de
  la cabeza), CE ponderada; entrena con early stopping por val macro-F1; evalúa val+test.
- Loguea a MLflow (experimento `clips-cnn-v1`, runs `clips-noaug` / `clips-aug`): params (backbone,
  pooling, k, lr, epochs, augment, seed, class weights), métricas por clase + macro-F1 + PR-AUC,
  matriz de confusión (reusa `evaluate.save_confusion_matrix_png`), y el delta de augmentation.
- Exporta el **mejor** bundle (mejor val macro-F1) a `models/clips-v1/`.

### Export / contrato de inferencia (`clip_export.py`)
- `ClipModelBundle`: `head_state_dict`, `backbone="resnet18"`, `pooling="mean"`, `classes`,
  `k`, `frame_size`, `normalize_mean/std`, `model_version`, `metrics`.
- `save_clip_bundle(bundle, dir)` / `load_clip_bundle(dir) -> (model, meta)`: reconstruye el
  `nn.Module` (backbone congelado + cabeza desde `state_dict`), listo en `eval()`.
- `predict_clip(model, meta, frames) -> (label, proba)`: aplica el transform de **eval serializado**
  (mismo que en validación → sin training-serving skew, invariante 3) y corre el forward. La usan
  el sanity-check de training y, en el sub-proyecto 4, el serving.

### Config (`train_clips.yaml`)
- `seed`, `backbone`, `pooling`, `k`, `frame_size`, `head{hidden,dropout}`,
  `train{epochs,patience,lr,batch_size}`, `augment`, `normalize{mean,std}`, `mlflow{...}`,
  `paths{model_dir,metrics_dir}`. Tipada con pydantic (sin números mágicos).

### Dependencias
- `torch` + `torchvision` en un grupo nuevo **`cnn`** (training-time). El serving (sub-proyecto 4)
  las moverá/sumará a prod cuando exponga el modelo. Device MPS en Apple Silicon.

## Métricas y evaluación (invariante 5)
- Por clase: precision/recall/F1/support + PR-AUC; macro-F1 como headline. Nunca accuracy a secas.
- Matriz de confusión (test) a MLflow. Comparación no-aug vs aug (delta de macro-F1).

## Testing
- `clips_dataset`: con un mini-manifest + frames JPG sintéticos en tmp, devuelve tensor de forma
  `(K,3,H,W)` y el índice de clase correcto; mapea `frame_paths` relativos bien.
- `clip_model`: forward con input `(B,K,3,224,224)` random → salida `(B,n_classes)`; el backbone
  tiene `requires_grad=False` (congelado); cambiar pooling no rompe la forma.
- `build_transforms`: el transform de **eval es determinista** (dos llamadas → mismo tensor) y el
  de augment introduce variación.
- `clip_export`: `save`→`load`→`predict_clip` reproduce la predicción; el bundle recarga sin la
  data de entrenamiento.
- `train_clips` smoke: con un mini-dataset sintético (pocas clases/clips), una época corre,
  evalúa, exporta y el bundle recargado predice clases válidas (rápido, CPU, sin MLflow).

## Out of scope (otros sub-proyectos)
- Grad-CAM (sub-proyecto 3), endpoint de upload de video + extracción en serving (4), frontend (5).
- Fine-tune del backbone (mejora medible futura), fusión con tabular (no aplica al flujo de upload),
  modelos temporales 3D / attention pooling (posible v2).

## Riesgos
- **Lentitud en MPS:** backbone en vivo sobre 762×8 frames/época. Mitigación: backbone en `no_grad`
  (solo forward), batch razonable, pocas épocas con early stopping. Si fuera inviable, fallback a
  cachear embeddings para el run no-aug.
- **Overfitting (dataset chico, cabeza chica):** dropout + weight de clases + early stopping por val.
- **Clases minoritarias** (goal/card pocas decenas en test): métricas por clase ruidosas; se reporta
  igual con la salvedad documentada.
- **Determinismo en MPS:** seedear torch/numpy/random; algunas ops MPS no son 100% deterministas —
  se loguea el seed y se asume tolerancia.
