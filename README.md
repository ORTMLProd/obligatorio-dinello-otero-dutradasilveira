# Clasificador de Eventos SoccerNet — ML en Producción (Obligatorio)

Sistema de ML end-to-end que clasifica **clips de video** de partidos de fútbol en tipos de
evento (`goal`, `card`, `substitution`, `corner`, `background`). El modelo de producción es
una **CNN multi-frame** (ResNet18 fine-tuneada + pooling temporal + cabeza MLP) que recibe un
clip subido y devuelve la clase, las probabilidades por clase y un **overlay de Grad-CAM** por
frame.

Curso: Machine Learning en Producción — Máster, Universidad ORT Uruguay.

> **Estado.** Modelo de video `clips-v1` (CNN visual-only fine-tuneada, **test macro-F1 0.757**
> sobre 44 partidos de 6 ligas), trazabilidad con MLflow (experimentos + Model Registry),
> API FastAPI con `/predict/clip` (+ `/predict` y `/predict/batch` del baseline tabular),
> frontend React de upload de video con Grad-CAM, y el stack completo en Docker Compose
> (api + frontend + mlflow + prometheus + grafana). Un baseline tabular `v0` (XGBoost sobre
> `[tabular point-in-time ⊕ embedding ResNet pooled]`) cerró el ciclo end-to-end inicial y se
> conserva como referencia.

## Política de datos (NDA)

Los videos de SoccerNet están protegidos por copyright y se obtuvieron bajo un NDA con
KAUST. **Nunca** se versionan videos `.mkv`, frames/imágenes extraídas, ni la contraseña
del NDA. Las carpetas `data/` y `models/` están en `.gitignore`. La contraseña se lee de
la variable de entorno `SOCCERNET_PASSWORD` (en `.env`, no versionado). El repo contiene
solo **código, configs y manifests** que permiten regenerar el dataset a quien tenga su
propia contraseña del NDA.

## Cómo correr

### 0. Construir el dataset

El dataset de partidos está definido por `configs/dataset.yaml` (`game_ids` explícito: 32 de
la Premier League + 12 cross-liga de LaLiga, Champions, Serie A, Bundesliga y Ligue 1).

```bash
cd backend
# Descarga labels + features + videos 224p (los videos requieren SOCCERNET_PASSWORD, NDA)
uv run --group data python -m src.data.download --config ../configs/dataset.yaml
# Splits por game_id (anti data-leakage) + dataset de clips (extrae K frames por ventana)
uv run --group data python -m src.data.splits --config ../configs/dataset.yaml
uv run --group data python -m src.data.build_clips --config ../configs/dataset.yaml
# → data/processed/clips_manifest.parquet + frames (gitignored)

# (Opcional) baseline tabular v0, sin videos ni NDA (usa features ResNet pre-extraídas):
uv run --group data python -m src.data.build_dataset --config ../configs/dataset.yaml
```

### 1. Entrenar el modelo de video (loguea a MLflow, exporta el bundle)

```bash
cd backend && uv sync --group ml
# CNN de clips: entrena la cabeza + fine-tune de layer4, registra en el Model Registry
# (soccernet-events-clips-v1) y exporta models/clips-v1/clip_model.pt
uv run python -m src.models.train_clips --config ../configs/train_clips.yaml

# (Opcional) baseline tabular v0 (LogReg/XGBoost + tuning Optuna):
uv run python -m src.models.train --config ../configs/train.yaml
```

Optimización medida (electivo): además de la data augmentation (medida dentro de
`train_clips`), la **quantization int8** del backbone se evalúa con:

```bash
uv run python -m src.models.quantize --config ../configs/train_clips.yaml
# reporta latencia, tamaño y macro-F1 FP32 vs int8, y loguea a MLflow
```

### 2. Levantar todo el stack con Docker

```bash
docker compose up --build
```

- **Frontend:** http://localhost:8080 — subir un clip de video → clase + barras de probabilidad
  + overlay de Grad-CAM.
- **API:** http://localhost:8000 — Swagger en [`/docs`](http://localhost:8000/docs).
- **MLflow UI:** http://localhost:5500 — experimentos + Model Registry (puerto 5500 porque en
  macOS el 5000 lo toma AirPlay).
- **Grafana:** http://localhost:3000 · **Prometheus:** http://localhost:9090 (monitoreo).

La API carga los modelos de `models/` (montado read-only): `models/clips-v1` para `/predict/clip`
y `models/v0` para `/predict`. Si un modelo no está entrenado, su endpoint responde **503**.
Para que el training registre en el MLflow del compose: `docker compose up -d`, entrenar con
`MLFLOW_TRACKING_URI=http://localhost:5500`, y `docker compose restart api` para recargar el
modelo recién exportado.

### Endpoints de inferencia

**`POST /predict/clip`** (producto) — recibe un archivo de video (multipart, campo `video`),
muestrea K frames y devuelve la clase, las probabilidades y un overlay de Grad-CAM por frame:

```bash
curl -s -F "video=@un_clip.mp4" http://localhost:8000/predict/clip
# → {"predicted_label": "corner", "probabilities": {...},
#    "model_version": "clips-v1-clips-aug-ft", "gradcam": [{"frame_index": 0, ...}, ...]}
```

**`POST /predict`** y **`POST /predict/batch`** (baseline tabular v0) — reciben las features
tabulares point-in-time + el embedding ResNet **pre-extraído** (`resnet_features`, 512-d);
`/predict/batch` recibe `{"items": [...]}` y devuelve una lista alineada. Consumir el mismo
embedding pre-extraído (en vez de la imagen cruda) evita el *training-serving skew* en v0.
Los mismos schemas pydantic (`extra="forbid"`) respaldan online y batch.

### Desarrollo local (sin Docker)

```bash
# Backend (Python 3.12, gestionado por uv)
cd backend && uv sync && uv run fastapi dev src/api/main.py    # http://localhost:8000

# Frontend (en otra terminal)
cd frontend && npm install && npm run dev                      # http://localhost:5173
```

En dev, Vite proxea `/api` al backend (mismo patrón que nginx en producción), por lo
que el código del frontend usa rutas relativas `/api/...` idénticas en ambos entornos.

## Deploy a AWS Elastic Beanstalk

El stack completo (api, frontend, mlflow, prometheus, grafana) se despliega en un único
EC2 vía la plataforma **Docker** de Elastic Beanstalk (no ECS-managed), que ejecuta
`docker compose up --build` en la instancia igual que en local. `api` y `frontend` se
buildean localmente y se pushean a ECR; `mlflow` se buildea en la propia instancia
(no tiene ECR repo propio); `prometheus`/`grafana` pullean sus imágenes públicas.

Requisitos: credenciales AWS válidas (`aws sts get-caller-identity`), EB CLI (`eb
--version`), Docker con soporte `buildx`, y los modelos entrenados en `models/v0` y
`models/clips-v1` — el `Dockerfile.deploy.api` los **hornea en la imagen** (ambos, incluido
el modelo de video para `/predict/clip`) porque `models/` está gitignored (NDA) y en EB no
hay host para bind-mountearlo.

**Nota — cuenta AWS Academy Learner Lab:** esta cuenta (`625067806263`) es un Learner
Lab (rol asumido `voclabs`), que **no permite crear ni modificar roles IAM**
(`iam:AttachRolePolicy`/`CreateRole` dan `AccessDenied`). El lab ya provee `LabRole` /
`LabInstanceProfile` con `AmazonEC2ContainerRegistryReadOnly` adjunto y con
`elasticbeanstalk.amazonaws.com` habilitado en su trust policy, así que sirven como
instance profile y service role de EB sin tocar IAM. Además las credenciales son
temporales (`ASIA...` + session token) y expiran cada pocas horas: si un comando
`aws`/`eb` empieza a fallar con `ExpiredToken`, hay que pedir credenciales nuevas del
panel del lab y actualizar `~/.aws/credentials`.

```bash
# 1. Login + build + push de api/frontend a ECR (forzar linux/amd64: las instancias
#    EC2 son x86_64, builds locales en Apple Silicon son arm64 por default)
aws ecr get-login-password --region us-east-1 \
  | docker login --username AWS --password-stdin 625067806263.dkr.ecr.us-east-1.amazonaws.com

docker buildx build --platform linux/amd64 -f Dockerfile.deploy.api \
  -t 625067806263.dkr.ecr.us-east-1.amazonaws.com/soccer-net/api:latest --push .

docker buildx build --platform linux/amd64 \
  -t 625067806263.dkr.ecr.us-east-1.amazonaws.com/soccer-net/frontend:latest --push ./frontend

# 2. Setup de EB (una sola vez): reusa LabRole/LabInstanceProfile (no crear IAM nuevo)
eb init -p docker soccer-net --region us-east-1
eb create soccer-net-prod --single --instance-type t3.medium \
  --instance-profile LabInstanceProfile --service-role LabRole

# 3. Deploy (usa docker-compose.prod.yml, ver scripts/deploy_eb.sh). Si el entorno ya
#    existe, este paso solo redeploya las imágenes nuevas (incluido el modelo de video).
scripts/deploy_eb.sh soccer-net-prod
```

`docker-compose.prod.yml` es la variante de deploy de `docker-compose.yml`: `api`/
`frontend` usan `image:` (ECR) en vez de `build:`, `frontend` publica `80:80` (el único
puerto que EB abre por default; `.ebextensions/security-group.config` abre además 5500 y
3000 para MLflow/Grafana), y `api` no monta `./models` porque ya viene horneado. El
script `scripts/deploy_eb.sh` hace el swap temporal `docker-compose.prod.yml` →
`docker-compose.yml` (staged en git, nunca commiteado) y lo restaura al terminar, para no
pisar el archivo de dev.

**El entorno EB genera costo mientras esté prendido** (~US$30/mes con `t3.medium`
24/7). Terminarlo cuando no se esté usando: `eb terminate soccer-net-prod`.

## Estructura

```
backend/   API FastAPI + datos, features, modelos (train/evaluate/export/quantize)
frontend/  React (Vite) + Tailwind — upload de video → clase + Grad-CAM
mlflow/    imagen del servidor de tracking de MLflow (Electivo 1)
configs/   YAML de configuración (dataset, train, train_clips, api) — pydantic-settings
report/    bitácora pedagógica + métricas + ejemplos de inferencia + (Fase 4) informe
```
