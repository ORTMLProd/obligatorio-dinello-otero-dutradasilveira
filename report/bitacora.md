# Bitácora pedagógica — Clasificador de Eventos SoccerNet

Registro de decisiones técnicas del obligatorio de ML en Producción (ORT). Cada
entrada vincula la decisión con (a) el concepto del curso, (b) el requerimiento de la
consigna que cubre, y (c) las alternativas consideradas y descartadas. Es el insumo
principal del informe final y de la declaración de uso de IA generativa.

---

## 2026-06-11 — Fase 0: Setup del monorepo (esqueleto end-to-end)

### Qué se hizo
Scaffolding del repositorio y un ciclo mínimo verificable de punta a punta:

- Repo git inicializado (rama `main`) + `.gitignore` con política NDA.
- **Backend**: FastAPI mínimo (`/health`, `/model-info` stub, `/`) con configuración
  vía `pydantic-settings`, Python 3.12 gestionado por uv, dependencias separadas
  prod/dev, y tests (`pytest`).
- **Frontend**: Vite + React + TypeScript + Tailwind v4; una página que consulta
  `/api/health` y muestra el estado del backend.
- **Docker**: Dockerfiles (backend uv→`python:3.12-slim`; frontend node→nginx) y
  `docker-compose.yml` con servicios `api` + `frontend` conectados por reverse-proxy.
- Docs (`docs/consigna.md`), esta bitácora y persistencia de convenciones de trabajo.

**DoD alcanzado:** `docker compose up` sirve la API y el frontend; el frontend muestra
"Backend conectado ✓".

### Por qué se hizo así
- **End-to-end primero, alcance recortado.** La Fase 0 no toca datos ni modelos: cierra
  el ciclo más chico que funcione. La consigna prioriza el desarrollo end-to-end por
  sobre el rendimiento, y conviene una base verificable antes de sumar complejidad.
- **Una única fuente de preprocesamiento.** `backend/src/features/` ya existe (vacío en
  Fase 0) y será importado por training Y API → previene training-serving skew desde el
  diseño.
- **nginx reverse-proxy en vez de CORS.** El navegador usa un solo origen (`/api/...`
  relativo), el backend queda interno y no hace falta CORS. En dev se replica con el
  `server.proxy` de Vite, así el código del frontend es idéntico en dev y prod.
- **Config antes que constantes.** Host/puerto/log-level salen de `configs/api.yaml` vía
  `pydantic-settings`; en contenedores pisan las env `API_*` (12-factor).
- **Python 3.12 gestionado por uv** (no el de conda del sistema), para reproducibilidad.

### Concepto del curso relacionado
- Semana 1 (scoping): cerrar el ciclo end-to-end antes de optimizar.
- Semana 5 + invariante anti training-serving skew: `src/features/` como única verdad.
- Semana 8 (serving / contratos): schemas pydantic estrictos (`extra="forbid"`).
- Semana 11 (MLOps): Docker + deps separadas prod/dev + reproducibilidad (lockfiles
  `uv.lock` y `package-lock.json` versionados).

### Requerimiento de la consigna que cubre
- Mínimo **"Ambiente: dependencias dev/prod + Docker"** (pyproject con grupos;
  Dockerfiles; docker-compose).
- Mínimo **"Versionado de código: Git + repo en GitHub"** (repo + conventional commits).
- Encamina el mínimo **"API online + batch + documentación"** (Swagger en `/docs`;
  `/predict` y `/predict/batch` llegan en Fase 2).
- Sienta las bases anti **data leakage** (`.gitignore` + estructura) y anti
  **training-serving skew** (`src/features/` compartido, schemas pydantic).

### Alternativas consideradas y descartadas
- **Kubeflow** para trazabilidad → descartado: plataforma sobre Kubernetes,
  desproporcionada para el alcance. Elegimos **MLflow** (liviano, un contenedor).
- **CORS** en vez de reverse-proxy → descartado: acopla el frontend a la URL/puerto del
  backend y expone la API al navegador.
- **shadcn/ui en Fase 0** → diferido a Fase 3 (no aporta al DoD; agrega ruido).
- **Deps de ML (torch/xgboost/mlflow) en Fase 0** → diferidas a Fase 1/2 para mantener
  build e instalación rápidos.
- **Versionar placeholders en `data/`/`models/`** → descartado por seguridad NDA: esas
  carpetas quedan 100% ignoradas; se crean en runtime.
- **Python 3.12 de conda** → descartado a favor del intérprete gestionado por uv.
- **PCAI (HPE Private Cloud AI)** → acelerador opcional y desacoplado para Fase 3 (GPU
  para v1), nunca dependencia del core (reproducibilidad + NDA).

### Referencias al código
- Bootstrap: `.gitignore`, `README.md`.
- Backend: `backend/src/api/`, `backend/src/config.py`, `backend/tests/`,
  `backend/pyproject.toml`.
- Frontend: `frontend/src/App.tsx`, `frontend/vite.config.ts`.
- Docker: `backend/Dockerfile`, `frontend/Dockerfile`, `frontend/nginx.conf`,
  `docker-compose.yml`.

### Uso de IA generativa
Fase desarrollada con asistencia de Claude Code (planificación, generación de código y
redacción de esta bitácora). Todo el contenido fue revisado por el estudiante,
responsable de su corrección.

### Detalle técnico a recordar (Fase 3)
El `.gitignore` ignora imágenes (`*.png`, `*.jpg`, …) por la política NDA. Cuando el
frontend necesite assets propios en Fase 3, acotar los patrones (p. ej. `/data/**/*.png`)
o agregar excepciones (`!frontend/src/assets/**`).

---

## 2026-06-12 — Fase 1: Datos + EDA (camino liviano)

### Qué se hizo
Pipeline de datos config-driven que construye un dataset de **ventanas** etiquetadas y
su EDA, sin descargar videos:

- **Descarga** (`src/data/download.py`): baja `Labels-v2.json` + features ResNet
  pre-extraídas (`*_ResNET_TF2_PCA512.npy`, 512-dim, 2 fps, una array por mitad) de 8
  partidos. **Sin password NDA** (los videos no se tocan). Idempotente.
- **Splits** (`src/data/splits.py`): asigna `game_id → {train,val,test}` (5/2/1)
  determinísticamente y los versiona en `configs/splits.yaml`.
- **Features compartidas** (`src/features/tabular.py`, `visual.py`): features tabulares
  **point-in-time** y pooling de la ventana ±2s sobre las features ResNet. Única fuente
  de preprocesamiento (la importará la API en Fase 2).
- **Constructor** (`src/data/build_dataset.py`): genera ventanas de evento (4 clases) +
  background (sampleado a ratio 2:1, ≥30s de toda anotación) → `manifest.parquet` (615
  ventanas) + `resnet_pooled.npy` (615×512, alineado por fila) + `report/dataset_summary.json`
  (conteos + hash de contenido).
- **EDA** (`notebooks/eda.ipynb`): consume **solo** desde `src/`; reporta distribución de
  clases, desbalance (17.8x; background 66.7%), eventos por minuto/mitad, cobertura por
  split y sanity de las features.
- **Tests** (todos en verde, 23): regresión de **leakage** (`test_leakage.py`),
  **point-in-time** (`test_pointintime.py`), unidad de features y de sampling de background.

**DoD alcanzado:** manifest parquet + splits versionados + notebook de EDA con hallazgos.

### Por qué se hizo así
- **Camino liviano = end-to-end primero.** Construir el manifest desde features
  pre-extraídas (no frames) evita GB de video y la password NDA, y ya da un dataset
  entrenable para el baseline v0 (Fase 2: LogReg/XGBoost sobre ResNet pooled ⊕ tabular).
  La extracción de frames se difiere a Fase 3, cuando la CNN v1 los necesite.
- **Splits por `game_id`, no por ventana.** Si dos ventanas del mismo partido cayeran en
  train y test, el modelo "vería" el partido de test al entrenar → leakage. La asignación
  se versiona en git (contrato chico y auditable), no en el parquet regenerable.
- **Point-in-time en lo tabular.** `score_diff`, `events_so_far`, etc. usan solo
  anotaciones **estrictamente anteriores** a `t` (un evento exactamente en `t` no cuenta
  todavía). Si usáramos el score final, el modelo tendría información del futuro.
- **No se fitea nada en Fase 1.** El manifest guarda features crudas; el scaler/encoder se
  fitea en Fase 2 **solo sobre train** para no filtrar estadísticas de val/test.
- **Background submuestreado (2:1), no natural.** El background real es muchísimo mayor;
  fijar el ratio por config evita un dataset 99% background y deja el desbalance manejable.

### Concepto del curso relacionado
- Preparación de datos y EDA (criterio de evaluación explícito de la consigna).
- **Data leakage** (desafío del curso): splits por grupo + corrección point-in-time.
- **Training-serving skew**: `src/features/` como única verdad, importable por la API.
- Reproducibilidad: todo seedeado y config-driven; hash de contenido del dataset.
- Desbalance de clases: se mide y se explicita (invariante 5); guía las métricas de Fase 2.

### Requerimiento de la consigna que cubre
- Mínimo **"Dataset propio + EDA"** (dataset de ventanas imágenes-derivadas ⊕ tabular).
- Mínimo **"Clasificación con target definido por los estudiantes"** (multiclase: goal,
  card, substitution, corner, background).
- Avanza los desafíos mínimos **data leakage** y **training-serving skew**.
- Encamina el electivo **Trazabilidad de ML** (manifest hasheado + splits versionados).

### Alternativas consideradas y descartadas
- **Descargar videos + extraer frames ahora** → diferido a Fase 3: pesado, requiere
  password NDA y no aporta al baseline v0.
- **Guardar el vector ResNet inline en el parquet** (columna lista de 512 floats) →
  descartado: `npy` separado alineado por fila es más liviano de leer y mantiene el
  manifest legible.
- **Comitear el `manifest.parquet`/`.npy`** → descartado por política NDA/tamaño; se
  versiona la config + splits + summary (hash), que permiten regenerarlo.
- **Split aleatorio por fila** → descartado: viola la invariante 1 (leakage).
- **Score relativo al equipo anotado** → se usó `home − away` (bien definido y
  point-in-time); el relativo se puede revisar si aporta en Fase 2.

### Limitaciones asumidas
- 8 partidos de **una sola liga** (`england_epl`): la dimensión "eventos por liga" del EDA
  queda trivial y val/test tienen pocas muestras por clase minoritaria. Escalable subiendo
  `num_games` en `configs/dataset.yaml` (y, a futuro, diversificando ligas).

### Referencias al código
- Datos: `backend/src/data/{config,download,splits,build_dataset,windows,labels,dataset}.py`.
- Features: `backend/src/features/{tabular,visual}.py`.
- Config/artefactos: `configs/dataset.yaml`, `configs/splits.yaml`, `report/dataset_summary.json`.
- EDA: `backend/notebooks/eda.ipynb`.
- Tests: `backend/tests/test_{leakage,pointintime,features_tabular,features_visual,background_sampling}.py`.

### Uso de IA generativa
Fase desarrollada con asistencia de Claude Code (planificación del alcance, generación de
código, del notebook y de esta bitácora). Todo el contenido fue revisado por el estudiante,
responsable de su corrección.
