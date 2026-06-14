# Diseño — Fase 3.5, sub-proyecto 1: Datos (clips de video)

Fecha: 2026-06-13 · Estado: aprobado para planificar

## Contexto y visión (3.5 completa)

La Fase 3.5 es el corazón del obligatorio: **integrar video en el flujo**. La experiencia
objetivo es: el usuario sube un **clip de hasta ~30s** y el modelo predice **una clase**
(`goal`, `corner`, `card`, `substitution`, `background`). Es **clasificación de clip** (una
etiqueta por video), no un timeline.

Por qué requiere un modelo nuevo: la API actual usa features **ResNet+PCA pre-extraídas** que
**no se pueden reproducir** desde un video arbitrario que suba un usuario. Para servir clips
necesitamos **nuestro propio extractor visual** entrenado sobre frames reales. Y un clip subido
**no tiene contexto tabular del partido** (minuto, score, eventos hasta t), así que el modelo de
clips es **solo-visual**. El modelo tabular+ResNet actual se mantiene para la demo "por ventana".

**Forma del modelo (clip nativo multi-frame):** K frames por clip → backbone CNN compartido
(ResNet18) por frame → K embeddings → pooling temporal (mean/max) → cabeza MLP → clase. Patrón
de *frame-pooling* (tipo TSN), sin 3D-conv. El Grad-CAM sale del backbone sobre el frame más
saliente.

**Descomposición en sub-proyectos** (cada uno: diseño → plan → implementación → PR):
1. **Datos** — descarga de videos + extracción de clips/frames + manifest. **(este spec)**
2. **Modelo** — entrenar el clasificador multi-frame (torch+MPS, MLflow, métricas por clase).
3. **Explicabilidad** — Grad-CAM sobre frames.
4. **Serving** — endpoint de upload de video (extracción idéntica a training) + contrato.
5. **Frontend** — subir video → clase + probabilidades + Grad-CAM.

Este documento cubre **solo el sub-proyecto 1 (Datos)**. Los otros cuatro se diseñan después.

---

## Objetivo del sub-proyecto 1

Producir un **dataset de clips** —cada clip = K frames extraídos de los videos, con su label y
split— reutilizando todo el windowing / labels / splits existente. Lo único nuevo es **bajar
videos** y **extraer frames**.

## Decisiones (acordadas)

- **Escala:** 16 partidos en video (`num_games: 16`). El disco (≈6–16 GB de video 224p) no es
  restricción (150 GB libres); el costo real es tiempo de descarga/entrenamiento.
- **Herramienta de extracción:** `opencv-python` (pip, sin ffmpeg de sistema; *seek* por
  timestamp; robusto en Apple Silicon).
- **Parámetros de clip (config, ajustables):** `K=8` frames equiespaciados, clip de `8s` (±4s
  alrededor del timestamp anotado), frames a `224×224`, formato JPG.
- **Manifest de clips:** nuevo `data/processed/clips_manifest.parquet`; el `manifest.parquet`
  ResNet actual queda intacto.
- **Splits:** los mismos por `game_id` (invariante 1), reusando/regenerando `configs/splits.yaml`.

## Arquitectura y componentes

```
configs/dataset.yaml  (nueva sección `clips` + num_games:16)
        │
        ▼
src/data/download.py  ── (extensión) baja 1_224p.mkv / 2_224p.mkv con SOCCERNET_PASSWORD
        │
        ▼
src/data/frames.py    ── (nuevo) extracción de K frames de un clip [t-4s, t+4s] con OpenCV
        │
        ▼
src/data/build_clips.py ── (nuevo) por ventana (evento + background) → extrae frames →
                            arma clips_manifest.parquet (reusa windows/labels/tabular/splits)
        │
        ▼
data/processed/frames/<game_id>/<window_id>/frame_{0..K-1}.jpg   (gitignored, NDA)
data/processed/clips_manifest.parquet                            (gitignored)
report/clips_summary.json                                        (conteos + hash, versionado)
```

- **`download.py` (extensión):** agregar los archivos de video a la lista descargable cuando una
  flag/sección lo pida; setear `downloader.password = os.environ["SOCCERNET_PASSWORD"]`. Si la
  env var falta y se piden videos → error claro (nunca imprime la password). Idempotente (salta
  videos ya presentes).
- **`src/data/frames.py` (nuevo):** `extract_clip_frames(video_path, center_ms, clip_ms, k, size)
  -> list[np.ndarray]`. Calcula K timestamps equiespaciados en `[center-clip/2, center+clip/2]`,
  clampea a los bordes del video, hace *seek* y lee cada frame con OpenCV, redimensiona a
  `size×size`. Función pura y testeable (con un video sintético chico generado en el test).
- **`src/data/build_clips.py` (nuevo):** orquesta. Reusa `event_windows` /
  `sample_background_positions` / `load_annotations` / `load_splits` / `build_tabular_features`.
  Por cada ventana: extrae K frames del video de su mitad, los guarda en disco, y agrega una fila
  al manifest con `frame_paths[K]` + tabular point-in-time + split. Idempotente (salta clips ya
  extraídos). Determinístico (seed).
- **`configs/dataset.yaml`:** nueva sección `clips` (`enabled`, `k`, `clip_seconds`, `frame_size`,
  `video_files: [1_224p.mkv, 2_224p.mkv]`) y `num_games: 16`. Tipada en `src/data/config.py`.

## Flujo de datos

1. `download` baja labels + features (ya existía) **+ videos** para los 16 partidos.
2. `splits` regenera la asignación `game_id → {train,val,test}` para 16 partidos.
3. `build_clips` recorre las ventanas (mismas reglas que el dataset actual: eventos de las 4
   clases + background submuestreado), extrae K frames por ventana del video correspondiente,
   los guarda y escribe `clips_manifest.parquet` + `clips_summary.json`.

## Política de datos (NDA — vinculante)

- **Nunca** se comitean: videos `.mkv`, frames `.jpg/.png`, ni la password. `data/` y los frames
  están gitignored; la password sale de `SOCCERNET_PASSWORD` (env), nunca se loguea/imprime.
- El repo solo versiona **código, config y manifests/summary** que permiten regenerar todo a
  quien tenga su propia password del NDA.

## Testing

- `frames.extract_clip_frames`: genera un video sintético chico (OpenCV `VideoWriter`, p. ej.
  frames con un número/color por índice), extrae K frames y verifica: cantidad = K, tamaño
  correcto, timestamps equiespaciados y clamp en los bordes.
- `build_clips`: con un mini-fixture (un video sintético + anotaciones), verifica que el manifest
  tiene `frame_paths` con K paths existentes por fila, las columnas esperadas, y que **ningún
  `game_id` cruza splits** (test de leakage, invariante 1).
- Determinismo: misma seed → mismas ventanas/frames seleccionados.

## Consistencia con el pipeline existente

Subir a `num_games: 16` regenera `splits.yaml` (16 partidos). Para mantener coherencia
(invariante 1) entre el modelo de ventana y el de clips, conviene reconstruir el dataset ResNet
y reentrenar/retunear el modelo de ventana sobre los 16 (mecánico, comandos ya existentes;
beneficio: mejora también las clases minoritarias del modelo tabular). Esto se ejecuta como paso
final del sub-proyecto 1, no es desarrollo nuevo.

## Fuera de alcance (otros sub-proyectos)

- Entrenamiento del modelo multi-frame (sub-proyecto 2).
- Grad-CAM (3), endpoint de upload de video y extracción en serving (4), frontend (5).
- Fusión tardía con tabular: no aplica al flujo de upload (visual-only); se podría explorar en el
  modelo de ventana, fuera de 3.5.

## Riesgos

- **Descarga lenta / interrumpida:** idempotencia + reintentos del SoccerNetDownloader; bajamos
  16 partidos (costo moderado).
- **Desfase temporal train/serve:** se entrena con clips de ~8s y se sirve hasta 30s; ambos se
  resuelven muestreando K frames. Limitación documentada; se evalúa en el sub-proyecto 2.
- **OpenCV en Apple Silicon:** `opencv-python` (wheels arm64) es estable; el test con video
  sintético valida la extracción en el entorno real.
