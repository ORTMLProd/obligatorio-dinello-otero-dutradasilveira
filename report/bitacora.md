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

---

## 2026-06-12 — Fase 2: Baseline end-to-end (modelo v0 + MLflow + API + Docker)

### Qué se hizo
El ciclo completo de ML, de los datos de Fase 1 a predicciones servidas en Docker. Se
estructuró en PRs chicos y apilados (cada rama sale de la anterior):

- **Preprocesador compartido** (`src/features/preprocess.py`): `build_preprocessor`
  (ColumnTransformer: OneHot para `league`, StandardScaler en numéricas, passthrough en
  binarias) y `assemble_matrix`, el **único punto** donde se arma la matriz
  `[tabular ⊕ embedding]`, usado idéntico por training y API.
- **Entrenamiento** (`src/models/train.py`): late fusion `[tabular point-in-time ⊕ ResNet
  pooled]`. Splitea por la columna `split` del manifest, fitea el preprocesador **solo en
  train**, entrena **LogReg y XGBoost**, selecciona el mejor por **macro-F1 de validación**
  (el test se reporta una vez, sin seleccionar sobre él), loguea a MLflow y registra el
  mejor en el Model Registry. Seedeado.
- **Evaluación** (`src/models/evaluate.py`): precision/recall/F1 por clase, macro-F1,
  PR-AUC one-vs-rest y matriz de confusión. Nunca accuracy a secas.
- **Export/inferencia** (`src/models/export.py`): bundle joblib (modelo + preprocesador
  fiteado + clases + `embedding_dim` + hashes de dataset/config + métricas) y
  `predict_frame`, el path de inferencia que comparte la API.
- **API** (`src/api/`): `POST /predict` y `POST /predict/batch` con schemas pydantic
  estrictos (`extra="forbid"`); el `lifespan` carga el bundle en `app.state` (nunca
  re-fitea); `/model-info` reporta el modelo real.
- **Docker** (`docker-compose.yml`, `mlflow/Dockerfile`): servicio MLflow (UI + Registry
  sobre sqlite, artefactos proxeados) en el puerto 5500; la API monta `./models` y carga
  el bundle.
- **Tests** (33 en verde): anti-skew del preprocesador, smoke de train (fit→export→reload→
  predict) y contrato de los endpoints (probas suman 1, `extra="forbid"`→422, embedding de
  largo inválido→422, 503 sin modelo).

**Resultado v0** (8 partidos, dataset chico): XGBoost gana con **val macro-F1 = 0.885 /
test = 0.800**, sobre LogReg (0.744 / 0.518). **DoD alcanzado:** clone fresco → `train` →
`docker compose up` sirve predicciones, con runs y modelo registrados en MLflow.

### Por qué se hizo así
- **Contrato visual = embedding precomputado (no imagen cruda) en v0.** El modelo se entrena
  con las features ResNet+PCA de SoccerNet, un extractor (TF2+PCA512) que no poseemos y no
  podemos reproducir desde pixeles en serving. Para evitar **training-serving skew**, la API
  consume el mismo embedding. La imagen cruda real llega en v1 (CNN propia → extractor
  idéntico en train y serving). Es la decisión más importante de la fase y queda explícita
  para el informe.
- **Preprocesador fiteado solo en train y serializado con el modelo.** Si fiteáramos el
  scaler/encoder sobre todo el dataset, filtraríamos estadísticas de val/test (leakage); si
  lo re-fiteáramos en la API, divergiría del de training (skew). Se fitea una vez, viaja en
  el bundle y la API lo carga.
- **Dos modelos comparados, selección en validación.** LogReg como piso interpretable y
  XGBoost como v0 real (maneja desbalance y no-linealidades del fusion). Elegir por val (no
  test) evita sesgar la estimación de generalización. Correr ambos es barato y da la
  discusión de alternativas que pide la consigna.
- **Desbalance manejado explícitamente.** `class_weight='balanced'` (LogReg) y
  `sample_weight` balanceado (XGBoost); métricas por clase + PR-AUC, nunca accuracy.
- **MLflow desacoplado del serving.** La API carga el bundle exportado de `models/v0`, no
  depende de MLflow en runtime: el tracking sirve trazabilidad, no es parte del path crítico.
- **El bundle no se comitea** (gitignored); se regenera con `train`. Trazabilidad vía hashes
  de dataset y config dentro del bundle, y vía MLflow.

### Concepto del curso relacionado
- **Training-serving skew** (desafío): única fuente de preprocesamiento + transformador
  serializado + contrato visual coherente train/serving.
- **Contrato de la API**: schemas pydantic estrictos, mismos tipos para online y batch.
- **Trazabilidad / versionado de modelos**: MLflow (experimentos + Model Registry).
- **Desbalance de clases** (invariante 5): pesos + métricas por clase + PR-AUC.
- **Reproducibilidad / determinismo**: seed logueado, config completa registrada, hashes.
- **Selección de modelo**: validación para elegir, test para reportar una sola vez.

### Requerimiento de la consigna que cubre
- Mínimo **"API online + batch + documentación"** (`/predict`, `/predict/batch`, Swagger).
- Mínimo **"Ambiente: dependencias dev/prod + Docker"** (stack completo en compose).
- Cierra los desafíos mínimos **data leakage** y **training-serving skew** en el path de
  entrenamiento + serving.
- Electivo 1 **Trazabilidad de ML** (MLflow: experimentos + registro de modelos).
- **Esto satisface el mínimo del obligatorio** (Fase 2 del plan).

### Alternativas consideradas y descartadas
- **`/predict` con imagen cruda + extractor propio en v0** → descartado: para no tener skew
  el extractor debe ser el mismo en train y serving, lo que obliga a extraer frames →
  videos → password NDA (camino pesado). Se difiere a v1.
- **Red profunda como baseline v0** → descartada: over-engineering para 615 ventanas y
  contradice "end-to-end antes que rendimiento". El gradient boosting sobre el fusion es un
  baseline honesto y fuerte.
- **MLflow con file-store** → descartado: MLflow 3 lo deprecó y no soporta Model Registry.
  Se usa sqlite (local) / servidor con sqlite (Docker).
- **Escalar lo tabular junto al embedding** → no se hizo: el embedding PCA ya está en rango
  razonable y XGBoost es invariante a escala; mantener `assemble_matrix` simple evita un
  scaler extra que también habría que serializar.
- **Batch asíncrono** → diferido: v0 es síncrono (suficiente para el volumen actual).
- **Puerto 5000 para MLflow** → cambiado a 5500: en macOS lo toma AirPlay/Control Center.

### Limitaciones asumidas
- Dataset chico (8 partidos, una liga): las clases minoritarias (goal n=3, card n=4 en test)
  dan métricas ruidosas. El macro-F1 de test (0.800) debe leerse con cautela; escalable
  subiendo `num_games`.
- La API debe reiniciarse si se entrena **después** de levantar el stack (el bundle se carga
  en el `lifespan`). Documentado en el README.

### Referencias al código
- Features: `backend/src/features/preprocess.py`.
- Modelos: `backend/src/models/{config,train,evaluate,export}.py`.
- API: `backend/src/api/{main.py,schemas.py,routers/predict.py,routers/model_info.py}`,
  `backend/src/config.py`.
- Config: `configs/train.yaml`. Métricas: `report/metrics/metrics_v0.json`.
- Docker: `docker-compose.yml`, `mlflow/Dockerfile`.
- Tests: `backend/tests/test_{preprocess,train_smoke,predict}.py`.

### Uso de IA generativa
Fase desarrollada con asistencia de Claude Code (planificación del alcance y de las
decisiones de diseño, generación de código y tests, y redacción de esta bitácora). Todo el
contenido fue revisado por el estudiante, responsable de su corrección y de poder defender
cada decisión.

---

## 2026-06-12 — Fase 3.1: Optimización de modelos (Optuna + feature selection)

### Qué se hizo
- Se implementó el electivo de **optimización de modelos** con **dos sub-técnicas**, ambas
  con impacto medido en métricas de ML **y** latencia:
  1. **Tuning de hiperparámetros con Optuna** (`backend/src/models/tune.py`): búsqueda TPE
     sobre el XGBoost, search space declarado en `configs/train.yaml` (sin números mágicos).
  2. **Feature selection tabular**: cada trial de Optuna también elige un subconjunto de
     `TABULAR_COLUMNS` (las `always_keep` nunca se descartan).
- Se hizo `build_preprocessor` **selection-aware** (parámetro `selected_columns`): filtra la
  partición cat/num/passthrough por intersección, sin dejar de ser la única fuente de
  preprocesamiento (invariante 3). Default `None` = comportamiento v0 intacto.
- `tune.py` **reutiliza** el pipeline de `train.py` (`split_dataset`, `fit_one`,
  `evaluate_on`, `_flatten_metrics`), sin duplicar lógica.
- Resultado de la corrida en vivo (40 trials, dataset v0), logueado en MLflow
  (experimento `optimization-v1`, modelo registrado `soccernet-events-v1`):

  | | macro-F1 (test) | p50 | p95 | features |
  |---|---|---|---|---|
  | baseline (params default, 8 feats) | 0.800 | 1.15 ms | 1.29 ms | 8 |
  | **tuned** (Optuna + selection) | **0.858** | 1.13 ms | 1.25 ms | 5 |
  | Δ | **+0.058** | −0.02 ms | −0.04 ms | −3 |

  El tuning **mejoró F1 y bajó latencia** a la vez, descartando 3 features tabulares débiles
  (`minute`, `score_diff`, `events_so_far`). El bundle tuneado se exporta a `models/v0` igual
  que el baseline; la API lo sirve sin cambios (el schema sigue aceptando las 8 columnas y el
  preprocesador descarta las 3 no usadas).
- Tests nuevos (TDD): selección de columnas del preprocesador (`test_preprocess.py`) y núcleo
  de tuning (`test_tune.py`), incluido un **test anti-leakage** que prueba que el `objective`
  no lee el split de test. Suite total: 41 tests verdes.

### Por qué se hizo así
- **La búsqueda optimiza solo macro-F1 de validación; el test se mide una vez al final.**
  Seleccionar hiperparámetros o features mirando test sería data leakage — el mismo principio
  que los splits por `game_id` y las features point-in-time.
- **Se mide impacto en ML _y_ latencia** porque la consigna lo exige explícitamente para el
  electivo: no basta "tuneé y mejoró", hay que cuantificar el trade-off (acá no hubo trade-off
  adverso: mejoró en ambas).
- **Feature selection integrada en el mismo estudio Optuna**: busca conjuntamente
  hiperparámetros + subconjunto de features, en vez de dos pasos separados; es más fiel a cómo
  interactúan y produce un único óptimo comparable.
- **Optuna y MLflow viven solo en el grupo `ml`** (no en la imagen de la API): el serving
  carga el bundle exportado y no depende de ninguno en runtime.

### Concepto del curso relacionado
- **Optimización de modelos** (electivo): tuning de hiperparámetros + selección de features.
- **Data leakage** (desafío): la función objetivo nunca toca test; selección sobre validación.
- **Trade-off rendimiento/latencia**: medición de p50/p95 de inferencia online.
- **Reproducibilidad**: sampler TPE seedeado, search space versionado en config.

### Requerimiento de la consigna que cubre
- Electivo **"Optimización de modelos (≥2 sub-técnicas, midiendo impacto)"**: cubierto con
  tuning (Optuna) + feature selection, impacto medido en métricas y latencia.
- Refuerza el electivo 1 **Trazabilidad** (nuevo experimento y modelo `v1` en MLflow).

### Alternativas consideradas y descartadas
- **Permutation importance como única feature selection (sin retrain)** → descartada: el
  electivo pide _medir impacto_, y un ranking de importancia no es un delta medido de quitar
  features. Se prefirió retrain con subconjunto para cuantificar el efecto real.
- **Search space hardcodeado en el código** → descartado: contradice "config antes que
  constantes". Vive en `train.yaml` con tipos validados (`SearchParamSpec`).
- **Optimizar sobre test** → prohibido (leakage); se optimiza sobre validación.
- **Grid search** → descartado frente a Optuna/TPE: más caro y peor cobertura del espacio
  continuo (`learning_rate` log, `subsample`, etc.).

### Limitaciones asumidas
- Mismo dataset chico (8 partidos): el +0.058 de macro-F1 se mide sobre pocas decenas de
  ejemplos en clases minoritarias → leer con cautela, igual que el baseline.
- El export sobrescribe `models/v0` con el modelo v1 tuneado; es gitignored y regenerable con
  `train`/`tune`. La versión real queda trazada en el bundle (`v1-tuned-...`) y en MLflow.

### Referencias al código
- Tuning: `backend/src/models/tune.py`. Config: `backend/src/models/config.py`
  (`SearchParamSpec`, `TuningConfig`), `configs/train.yaml` (sección `tuning`).
- Preprocesador selection-aware: `backend/src/features/preprocess.py`.
- Tests: `backend/tests/test_tune.py`, `backend/tests/test_preprocess.py`.
- Comando: `uv run python -m src.models.tune --config ../configs/train.yaml`.

### Uso de IA generativa
Bloque desarrollado con asistencia de Claude Code (diseño del enfoque, generación de código y
tests con TDD, ejecución de la búsqueda y redacción de esta entrada). El estudiante revisó y
validó cada decisión y es responsable de poder defenderla.

---

## 2026-06-12 — Fix: contrato de `/predict` vs. feature selection interna

### Qué se hizo
- Se corrigió una **regresión de serving** introducida por la Fase 3.1: con el modelo
  tuneado (5 de 8 features), `POST /predict` rompía con
  `KeyError: ['minute', 'score_diff', 'events_so_far'] not in index`.
- Causa raíz: `tune.py` guardaba en el bundle `tabular_columns = columnas_seleccionadas` (5).
  El router arma el DataFrame del request a partir de `bundle.tabular_columns`, y
  `assemble_matrix` exige las 8 columnas → faltaban 3 → excepción.
- Fix: `tabular_columns` vuelve a ser **el contrato completo del request** (las 8 que el
  cliente siempre envía); la feature selection vive **solo dentro del preprocesador fiteado**.
- Test de regresión que reproduce el camino del router (frame armado desde
  `bundle.tabular_columns` → `predict_frame`); falla antes del fix, pasa después. Verificado
  además end-to-end con `TestClient`: `/predict` → 200 con `v1-tuned-xgboost`.

### Por qué se hizo así / concepto del curso
- **Contrato de la API estable vs. internals del modelo** (contrato de API + training-serving
  skew): lo que el cliente debe enviar no puede cambiar porque internamente el modelo use
  menos features. El contrato es el schema pydantic (8 campos); la selección es un detalle de
  preprocesamiento que el bundle encapsula.
- **El bug se coló porque los tests de 3.1 no ejercían el camino del router con un bundle de
  subconjunto.** Lección: testear en la *frontera de serving*, no solo las piezas. El nuevo
  test cubre ese hueco.

### Requerimiento de la consigna que cubre
- Sostiene el mínimo **"API online + batch"** funcionando con el modelo optimizado, y el
  desafío **training-serving skew** (contrato coherente train/serving).

### Referencias al código
- Fix: `backend/src/models/tune.py` (`_fit_bundle`, reporte de selección en `run_tuning`).
- Test: `backend/tests/test_tune.py::test_tuned_bundle_serves_with_full_request_contract`.

### Uso de IA generativa
Bug detectado y corregido con Claude Code (reproducción, diagnóstico por stack trace, test de
regresión y fix con TDD). Revisado por el estudiante.

---

## 2026-06-12 — Fase 3.2: Explicabilidad con SHAP (tabular + API)

### Qué se hizo
- Se implementó el electivo de **explicabilidad** con SHAP, en una única fuente reutilizable
  (`backend/src/models/explain.py`) importada por **la API y el notebook** (invariante 3).
- **TreeSHAP nativo de XGBoost** (`pred_contribs=True`) en vez de la librería `shap`: da los
  mismos valores exactos pero sin meter `numba`/`llvmlite` en la imagen de producción. La lib
  `shap` quedó **solo en el grupo dev**, para los gráficos del notebook.
- **Tratamiento del embedding**: las 512 dims del ResNet (opacas) se agregan en un único
  bucket `visual_embedding`; las features tabulares se reportan con su nombre (las one-hot de
  `league` se pliegan a `league`). Resultado: una explicación legible visual-vs-contexto.
- **Exposición en la API**: nuevo campo `explanations` en `PredictResponse` (aditivo, no rompe
  el contrato de entrada). `/predict` lo devuelve por defecto; `/predict/batch` con
  `?explain=true`. Se calcula todo el batch en una sola pasada de TreeSHAP. Gateado por tipo de
  modelo (solo árboles); para LogReg el campo es `null`.
- **Notebook** `backend/notebooks/explainability.ipynb` (ejecutado, con plots embebidos):
  importancia global (media |SHAP|), reparto visual/tabular, y desglose de casos individuales.
- Hallazgo (dataset v0, 72 ventanas de test): el **contexto tabular domina (~89% de la masa
  |SHAP|)**, sobre todo `team_is_home` (separa `background`=-1 de los eventos); el embedding
  visual aporta ~11%. Coherencia con 3.1: solo aparecen las 5 features que el modelo tuneado
  conserva tras feature selection.

### Por qué se hizo así / concepto del curso
- **Explicabilidad / interpretabilidad** (electivo): SHAP reparte la predicción entre features
  (valores de Shapley) — permite *defender* por qué el modelo decidió.
- **No duplicar lógica** (invariante 3): la misma función explica en training-analysis y en
  serving; si la API reimplementara SHAP, habría training-serving skew también en la explicación.
- **Peso de dependencias en serving**: usar el TreeSHAP nativo mantiene la imagen liviana — una
  decisión de MLOps (no todo lo que sirve para analizar debe ir al contenedor de producción).
- **Agregar el embedding** evita 512 barras ilegibles: la explicación útil es visual-vs-tabular
  + ranking de las tabulares interpretables.

### Requerimiento de la consigna que cubre
- Electivo **"Explicabilidad"**: SHAP integrado en notebook **y** en la API.
- Refuerza el contrato de la API (`/predict` ahora explica sus predicciones, con ejemplo en
  Swagger).

### Alternativas consideradas y descartadas
- **Librería `shap` en producción** → descartada: arrastra `numba`/`llvmlite` (imagen pesada).
  El TreeSHAP nativo de XGBoost da los mismos valores sin ese costo.
- **SHAP por cada una de las 512 dims** → descartado: ilegible, domina el ruido sobre la señal.
  Se agregan en un bucket `visual_embedding`.
- **Explicaciones siempre on en batch** → se dejó off por defecto (`?explain=true` para activar):
  controla la latencia del camino de alto volumen.
- **KernelExplainer / explainer agnóstico** → innecesario: el modelo es de árboles y TreeSHAP es
  exacto y rápido.

### Limitaciones asumidas
- En dataset chico, `team_is_home` domina la explicación (separa background de eventos): es un
  artefacto del subconjunto, a releer cuando escale `num_games` y entre la CNN v1.
- TreeSHAP nativo aplica solo a modelos de árbol; si ganara LogReg, la API devuelve
  `explanations: null` y la explicabilidad se haría con `shap` en el notebook.

### Referencias al código
- Lógica: `backend/src/models/explain.py`. API: `backend/src/api/schemas.py`,
  `backend/src/api/routers/predict.py`. Deps: `backend/pyproject.toml` (`shap` en dev).
- Notebook: `backend/notebooks/explainability.ipynb`.
- Tests: `backend/tests/test_explain.py`, `backend/tests/test_predict.py`.

### Uso de IA generativa
Bloque desarrollado con asistencia de Claude Code (diseño, código y tests con TDD, construcción
y ejecución del notebook, redacción de esta entrada). El estudiante revisó y validó cada
decisión y es responsable de poder defenderla.

---

## 2026-06-12 — Fase 3.3: Frontend de visualización (React)

### Qué se hizo
- Se implementó el electivo de **visualización/UI**: un frontend React (Vite + Tailwind) que
  consume la API y muestra la predicción de una ventana.
- **Diseño validado con mockups primero** (convención del proyecto): se produjeron 3 direcciones
  estáticas en `frontend/mockups/` (consola de telemetría, editorial, pizarra táctica) y el
  equipo eligió la **pizarra táctica** antes de codear en React.
- UI: header con `/model-info` (versión, macro-F1, estado), formulario con los 8 campos
  tabulares editables + chips de **ejemplos pre-cargados** (uno por clase, exportados del split
  de test a `frontend/src/examples.json`), y panel de predicción con cancha esquemática (evento
  ubicado por clase), **barras de probabilidad**, **SHAP divergente** (campo `explanations` de la
  API) y espacio reservado para el Grad-CAM.
- **Rediseño por continuidad** (tras revisión): el panel de predicción está **siempre montado**
  con estado idle/cargando/resultado (estructura estable, sin reflow); el resultado se rellena
  con movimiento (caída del marcador, barrido de barras, reveal escalonado), respetando
  `prefers-reduced-motion`.
- Verificado end-to-end: dev (Vite proxy → API) y **stack docker completo** (`docker compose up`
  → nginx :8080 proxea `/api` → predicción real con SHAP). Build de producción OK; lint y tsc
  limpios.

### Por qué se hizo así / concepto del curso
- **Visualización/interacción con el modelo** (electivo): una UI para consultar predicciones y
  ver la explicación, no solo la API cruda.
- **El embedding viene de ejemplos pre-cargados**: la API necesita el ResNet-512 ya extraído,
  que **no se puede reproducir desde píxeles** en serving (sería training-serving skew). Por eso
  no se sube video/frame todavía; eso llega con la CNN v1 (3.5), donde el extractor es el mismo
  en training y serving.
- **NDA / gobernanza de datos**: no se muestran videos ni frames (prohibido commitearlos). Los
  embeddings sí se pueden incluir (se descargan sin password). La cancha esquemática es la
  representación NDA-safe; el slot del frame real queda para 3.5 (local-only).
- **Sin CORS**: el frontend usa rutas relativas `/api/...` que el reverse-proxy (Vite en dev,
  nginx en prod) reenvía al backend — mismo código en ambos entornos.
- **Contrato compartido**: los tipos TS (`lib/types.ts`) reflejan los schemas pydantic; el
  frontend es un consumidor más del contrato de la API.

### Requerimiento de la consigna que cubre
- Electivo **"Visualización / UI"** (cierra el 3.º electivo no-mínimo; con Trazabilidad,
  Optimización y Explicabilidad ya van 4).
- DoD de la fase: `docker compose up` sirve una predicción real desde la UI.

### Alternativas consideradas y descartadas
- **Direcciones estéticas editorial / consola de telemetría** → descartadas a favor de la
  pizarra táctica (más memorable y contextual; la cancha como elemento que en 3.5 alojará el
  frame + Grad-CAM).
- **Endpoint `/examples` en la API** → descartado por ahora: obligaría a que la API cargue el
  dataset (hoy solo carga el modelo). Se prefirió un JSON chico bundleado.
- **Subir frame/video en 3.3** → no es posible sin la CNN propia (skew); diferido a 3.5.
- **Panel de resultado que aparece/desaparece** → descartado tras revisión: rompía la
  continuidad (reflow). Se pasó a un panel siempre presente con estados.
- **shadcn/ui** → no se usó: el set de componentes es chico y a medida; Tailwind + CSS propio
  alcanza y evita dependencia extra.

### Limitaciones asumidas
- La demo predice sobre **ventanas de ejemplo**, no sobre video real (límite de contrato + NDA).
- `examples.json` (~40 KB) bundlea 5 embeddings; aceptable para la demo.

### Referencias al código
- App: `frontend/src/App.tsx`, `frontend/src/components/{WindowForm,PredictionPanel,Pitch}.tsx`.
- Contrato/cliente: `frontend/src/lib/{types,api}.ts`. Estilos: `frontend/src/index.css`.
- Ejemplos: `frontend/src/examples.json`. Mockups: `frontend/mockups/*.html`.

### Uso de IA generativa
Bloque desarrollado con asistencia de Claude Code (mockups, diseño y código React, verificación
end-to-end con Playwright/Docker, redacción de esta entrada). El estudiante revisó la dirección
visual y validó el resultado; es responsable de poder defenderlo.

---

## 2026-06-13 — Fase 3.1 (complemento): visualizaciones de la optimización

### Qué se hizo
- Se agregaron a `src.models.tune` las visualizaciones que faltaban del electivo de
  optimización, logueadas a MLflow (experimento `optimization-v1`):
  - **Matriz de confusión** del modelo baseline y del tuneado sobre test (reusa
    `evaluate.save_confusion_matrix_png`).
  - **Plots de Optuna**: historial de optimización + importancia de hiperparámetros
    (`save_optuna_plots`, vía `optuna.visualization.matplotlib` — sin sumar plotly/kaleido).
- Las imágenes se generan en `report/metrics/` (gitignored por NDA) y se suben como
  artefactos a MLflow; se ven en la UI de MLflow, no en git.

### Hallazgo (insumo para el informe)
- La importancia de hiperparámetros de Optuna muestra que **lo que más mueve el F1 de
  validación es qué features se conservan** (`keep_secs_since_last_event` ≈ 0.62,
  `keep_events_so_far` ≈ 0.13), por encima de los hiperparámetros del XGBoost (todos < 0.02).
  Es decir: en este dataset la **feature selection pesó más que el tuning** — justifica haber
  buscado ambas sub-técnicas en conjunto.
- La matriz de confusión del tuneado tiene diagonal fuerte; los errores residuales son
  `goal→corner` y `substitution→corner` (clases minoritarias, pocas muestras en test).

### Concepto del curso / requerimiento
- Refuerza el electivo **Optimización** (evidencia visual del impacto) y **Trazabilidad**
  (artefactos versionados en MLflow).

### Referencias al código
- `backend/src/models/tune.py` (`save_optuna_plots`, `_confusion_png`, logging en
  `run_tuning`). Test: `backend/tests/test_tune.py::test_save_optuna_plots_produces_files`.

### Uso de IA generativa
Complemento desarrollado con Claude Code (TDD del helper de plots, integración y ejecución).
Revisado por el estudiante.

---

## 2026-06-13 — Fase 3.4: Monitoreo (Prometheus + Grafana)

### Qué se hizo
- Observabilidad de la API con **Prometheus + Grafana** (puntos extra), instrumentada de forma
  explícita (no auto-instrumentación) para que sea defendible.
- **Endpoint `/metrics`** (`src/monitoring/metrics.py`) con métricas propias:
  `soccernet_predictions_total{predicted_label,model_version}` (distribución en vivo),
  `soccernet_prediction_latency_seconds` (histograma con buckets ms),
  `soccernet_requests_total{endpoint,status}` (tráfico/errores) y
  `soccernet_training_class_ratio{class_name}` (**baseline de drift**).
- **Baseline de drift**: la distribución de clases del split de entrenamiento se guarda en el
  bundle (`train_class_ratio`, nuevo campo) en train.py/tune.py, y la API la publica como gauge
  al arrancar. El dashboard compara **vivo vs entrenamiento**.
- **Logging de inferencias** (`src/monitoring/logging.py`): una línea JSON por predicción a
  stdout (12-factor) con features tabulares + clase + probabilidades + versión + latencia,
  **sin el embedding ni imágenes** (política de datos). El logger se configura explícitamente
  a stdout en el arranque (un logger custom sin handler no emitiría).
- **Stack en docker-compose**: servicios `prometheus` (scrapea `api:8000/metrics`) y `grafana`
  con datasource + dashboard **provisionados por archivo** (`monitoring/`), así `docker compose
  up` ya levanta el dashboard sin clicks (anónimo Viewer para la demo).
- Verificado end-to-end: tráfico real → Prometheus target `up`, query OK, y Grafana mostrando
  predicciones totales, requests/s, latencia p50/p95, distribución predicha (pie) y **drift**
  (vivo ~53% corner / 47% background vs baseline 67% background / 16% corner — el sesgo lo
  causan embeddings de prueba irreales, ilustra bien la detección).

### Por qué se hizo así / concepto del curso
- **Monitoreo / observabilidad en producción**: tráfico, latencia y, sobre todo, **qué predice
  el modelo** — la base para detectar *data/concept drift*.
- **Baseline de entrenamiento como referencia de drift**: comparar la distribución servida
  contra la de entrenamiento es la forma operativa de ver si el input se aleja de lo visto.
- **Instrumentación explícita** (Counter/Histogram/Gauge a mano): se ve y se defiende qué se
  mide; preferida a una librería mágica.
- **Política de datos**: el log nunca incluye el embedding ni imágenes; provisioning por
  archivo = reproducibilidad (mismo dashboard en cualquier `docker compose up`).
- **prometheus-client es dep de runtime** de la API (expone `/metrics`); Prometheus/Grafana son
  servicios aparte, no tocan la imagen de la API.

### Requerimiento de la consigna que cubre
- **Técnica avanzada adicional / puntos extra**: monitoreo con Prometheus/Grafana + detección
  de drift. Refuerza la trazabilidad y la operación del modelo servido.

### Alternativas consideradas y descartadas
- **`prometheus-fastapi-instrumentator` (auto-instrumentación)** → descartada: menos código pero
  menos defendible y con métricas genéricas; se prefirió definir las métricas relevantes a mano.
- **Drift con chequeo estadístico (divergencia/chi²)** → diferido: el panel vivo-vs-baseline ya
  da la señal; un score estadístico es más pesado para puntos extra.
- **Persistir logs a archivo** → descartado: stdout (12-factor), sin estado en el contenedor.
- **Exponer el baseline desde el dataset en la API** → descartado: la API no carga el dataset;
  el baseline viaja en el bundle (única fuente versionada con el modelo).

### Limitaciones asumidas
- El "drift" mostrado es ilustrativo (embeddings de prueba); con tráfico real reflejaría el
  uso. Grafana usa login anónimo Viewer (demo local), no apto para producción real.

### Referencias al código
- Métricas/logging: `backend/src/monitoring/{metrics,logging}.py`. Endpoint:
  `backend/src/api/routers/metrics.py`. Instrumentación: `backend/src/api/routers/predict.py`,
  `backend/src/api/main.py`. Baseline: `train_class_ratio` en `export.py`/`train.py`/`tune.py`.
- Stack: `docker-compose.yml`, `monitoring/prometheus/prometheus.yml`,
  `monitoring/grafana/**`. Tests: `backend/tests/test_monitoring.py`, `test_predict.py`.

### Uso de IA generativa
Bloque desarrollado con asistencia de Claude Code (diseño, código y tests con TDD, configuración
del stack y verificación end-to-end con Docker/Playwright, redacción de esta entrada). El
estudiante revisó y validó cada decisión y es responsable de poder defenderla.

---

## 2026-06-13/14 — Fase 3.5, sub-proyecto 1: Datos (clips de video)

### Qué se hizo
- Arranca la Fase 3.5 (integrar video en el flujo). Visión: subir un clip de ~30s → el modelo
  predice una clase. Modelo objetivo: **clip nativo multi-frame** (K frames → CNN compartida →
  pooling → cabeza), solo-visual (un clip subido no tiene contexto tabular del partido). Se
  descompuso en 5 sub-proyectos (Datos, Modelo, Grad-CAM, Serving, Frontend); este es el (1).
- **Proceso disciplinado:** brainstorming → spec (`docs/superpowers/specs/`) → plan
  (`docs/superpowers/plans/`) → implementación TDD con subagentes → review → corrida real.
- **Código (TDD):** `src/data/frames.py` (extracción de K frames equiespaciados con OpenCV,
  *seek* por timestamp), `src/data/build_clips.py` (builder del `clips_manifest.parquet`,
  reusa windows/labels/splits/tabular), extensión de `download.py` (videos 224p con password
  NDA del entorno), `ClipsConfig` en `dataset.yaml`. Tests con video sintético + test de
  leakage por `game_id`. Un fix de review: error claro si no hay clips (videos sin descargar).
- **Corrida real:** descargados labels+features+**videos** de **16 partidos** (~6.2 GB, NDA);
  splits regenerados (10/3/3); **1140 clips** de 8 frames construidos. Escalar a 16 mejoró las
  minoritarias: `goal` 11→45, `card` 22→64. Dataset ResNet de ventana reconstruido para 16
  (consistencia de splits).
- Hallazgo honesto: al retunear el modelo de ventana sobre 16, el tuneado (optimizado en val)
  **no le ganó al baseline en test** (0.780 vs 0.797). Esperable: tunear sobre validación no
  garantiza mejora en test con otro split y clases chicas. Ambas corridas quedan en MLflow.

### Por qué se hizo así / concepto del curso
- **Por qué un extractor propio:** las features ResNet+PCA pre-extraídas no se reproducen desde
  un video arbitrario; para servir clips subidos hace falta una CNN propia sobre frames →
  requiere frames reales → videos (NDA). Evita training-serving skew en lo visual.
- **Splits por `game_id`** (invariante 1, anti-leakage): los clips de un partido nunca cruzan
  train/val/test; cubierto por test.
- **Gobernanza de datos / NDA:** videos y frames nunca al repo (gitignored); password desde
  `SOCCERNET_PASSWORD` (env, vía `.env`), nunca impresa ni logueada. Solo se versionan código,
  config, splits y summaries (con hash de contenido para trazabilidad).
- **Reproducibilidad:** pipeline config-driven, idempotente y seedeado; el dataset se regenera
  con la password propia de cada quien.

### Requerimiento de la consigna que cubre
- Cimiento del electivo **Explicabilidad visual (Grad-CAM)** y de la integración de video
  (técnica avanzada). Habilita los sub-proyectos 2–5.

### Alternativas consideradas y descartadas
- **Clasificar 1 frame representativo / agregación al servir** → se eligió clip nativo
  multi-frame (más fiel a "el video es un gol").
- **ffmpeg de sistema / decord** → OpenCV (`opencv-python`, sin dep de sistema, robusto en arm64).
- **8 partidos** → 16 (el disco sobraba; mejora minoritarias). Escalable por config.
- **Endpoint `/examples` para el baseline de datos** → no aplica acá.

### Limitaciones asumidas
- Desfase temporal train/serve (clips de 8s vs uploads de hasta 30s): se resuelve muestreando K
  frames; a evaluar en el sub-proyecto 2.
- Clases minoritarias siguen chicas en test (goal/card pocas decenas).

### Referencias al código
- `backend/src/data/{frames,build_clips,download,config}.py`, `configs/dataset.yaml`.
- Tests: `backend/tests/test_{frames,build_clips,download,data_config}.py`.
- Spec: `docs/superpowers/specs/2026-06-13-cnn-clips-datos-design.md`.
  Plan: `docs/superpowers/plans/2026-06-13-cnn-clips-datos.md`.
- Comandos: `download` → `splits` → `build_clips` (requieren `SOCCERNET_PASSWORD`).

### Uso de IA generativa
Sub-proyecto desarrollado con Claude Code: brainstorming del alcance, spec y plan, implementación
TDD vía subagentes con review automatizado, y orquestación de la corrida real (descarga NDA →
clips). El estudiante definió la visión (video en el flujo, 16 partidos), aportó la password del
NDA y revisó/validó cada decisión; es responsable de poder defenderla.
