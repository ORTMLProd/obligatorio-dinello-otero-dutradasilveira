# Clasificador de Eventos SoccerNet — ML en Producción (Obligatorio)

Sistema de ML end-to-end que clasifica ventanas de video de partidos de fútbol en
tipos de evento (`goal`, `card`, `substitution`, `corner`, `background`), combinando
**frames** extraídos de los videos de SoccerNet con **features tabulares** point-in-time
(minuto, mitad, diferencia de score acumulada, liga, etc.).

Curso: Machine Learning en Producción — Máster, Universidad ORT Uruguay.

> **Estado: Fase 2 (Baseline end-to-end) — completa.** Modelo v0 (XGBoost sobre late
> fusion `[tabular point-in-time ⊕ embedding ResNet pooled]`), trazabilidad con MLflow
> (experimentos + Model Registry), API FastAPI con `/predict` y `/predict/batch`, y el
> stack completo en Docker Compose. Esto satisface el mínimo del obligatorio. El plan de
> fases completo está en [CLAUDE.md](CLAUDE.md).

## Política de datos (NDA)

Los videos de SoccerNet están protegidos por copyright y se obtuvieron bajo un NDA con
KAUST. **Nunca** se versionan videos `.mkv`, frames/imágenes extraídas, ni la contraseña
del NDA. Las carpetas `data/` y `models/` están en `.gitignore`. La contraseña se lee de
la variable de entorno `SOCCERNET_PASSWORD` (en `.env`, no versionado). El repo contiene
solo **código, configs y manifests** que permiten regenerar el dataset a quien tenga su
propia contraseña del NDA. Detalle completo en [CLAUDE.md](CLAUDE.md).

## Cómo correr

### 0. Construir el dataset (requiere contraseña del NDA para los videos; el camino liviano usa features pre-extraídas)

```bash
cd backend
uv run --group data python -m src.data.download --config ../configs/dataset.yaml
uv run --group data python -m src.data.build_dataset --config ../configs/dataset.yaml
# genera data/processed/{manifest.parquet, resnet_pooled.npy} (gitignored)
```

### 1. Entrenar el baseline v0 (loguea a MLflow, exporta el modelo)

```bash
cd backend
uv sync --group ml
# Local (sqlite, sin servidor): runs + registry en backend/mlflow.db
uv run python -m src.models.train --config ../configs/train.yaml
# ...o apuntando al MLflow del compose (ver abajo): MLFLOW_TRACKING_URI=http://localhost:5500
```

Entrena LogReg y XGBoost, reporta métricas por clase (F1, PR-AUC) y exporta el mejor a
`models/v0/model.joblib` (el bundle que carga la API: modelo + preprocesador fiteado).

### 2. Levantar todo el stack con Docker

```bash
docker compose up --build
```

- **Frontend:** http://localhost:8080 — estado del backend.
- **API:** http://localhost:8000 — Swagger en [`/docs`](http://localhost:8000/docs).
- **MLflow UI:** http://localhost:5500 — experimentos + Model Registry (puerto 5500 porque
  en macOS el 5000 lo toma AirPlay).

La API carga el modelo de `models/v0` (montado read-only). Si todavía no entrenaste,
`/predict` responde **503** y `/model-info` reporta `model_loaded: false`. Para que el
training registre en el MLflow del compose: `docker compose up -d`, después entrenar con
`MLFLOW_TRACKING_URI=http://localhost:5500`, y `docker compose restart api` para recargar
el modelo recién exportado.

### Endpoints de inferencia

`POST /predict` recibe las features tabulares point-in-time + el embedding ResNet
**pre-extraído** (`resnet_features`, 512-d). En v0 el contrato es el embedding, no la
imagen cruda: el modelo se entrena con las features ResNet+PCA de SoccerNet, que no se
pueden reproducir desde pixeles en serving — consumir el mismo embedding evita el
*training-serving skew*. La ingesta de imagen real llega en v1 (CNN propia).

```bash
curl -s http://localhost:8000/predict -H 'content-type: application/json' -d '{
  "half": 2, "minute": 44, "score_diff": 1, "league": "england_epl",
  "team_is_home": 1, "visible": 1, "events_so_far": 27, "secs_since_last_event": 18.0,
  "resnet_features": [0.0, 0.0, "...512 floats..."]
}'
# → {"predicted_label": "...", "probabilities": {...}, "model_version": "v0-xgboost-..."}
```

`POST /predict/batch` recibe `{"items": [<PredictRequest>, ...]}` y devuelve una lista
alineada de resultados (síncrono en v0). Los mismos schemas pydantic respaldan ambos.

### Desarrollo local (sin Docker)

```bash
# Backend (Python 3.12, gestionado por uv)
cd backend && uv sync && uv run fastapi dev src/api/main.py    # http://localhost:8000

# Frontend (en otra terminal)
cd frontend && npm install && npm run dev                      # http://localhost:5173
```

En dev, Vite proxea `/api` al backend (mismo patrón que nginx en producción), por lo
que el código del frontend usa rutas relativas `/api/...` idénticas en ambos entornos.

## Estructura

```
backend/   API FastAPI + datos, features, modelos (train/evaluate/export)
frontend/  React (Vite) + Tailwind
mlflow/    imagen del servidor de tracking de MLflow (Electivo 1)
configs/   YAML de configuración (dataset, train, api) — pydantic-settings
docs/      consigna y documentación
report/    bitácora pedagógica + métricas + (Fase 4) informe
```
