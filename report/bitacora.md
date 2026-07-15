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

---

## 2026-06-14 — Fase 3.5, sub-proyecto 2: Modelo (CNN multi-frame de clips)

### Qué se hizo
- Se entrenó el **clasificador multi-frame visual-only**: ResNet18 ImageNet **congelada** (forward
  en vivo, sin gradientes) → embedding 512-dim por frame → **mean-pool** de los 8 frames → cabeza
  MLP (512→256→5). Solo entrena la cabeza. PyTorch + torchvision en MPS (Apple Silicon).
- **Componentes (TDD, subagentes):** `clip_model.py` (modelo + transforms anti-skew + device),
  `clips_dataset.py` (Dataset que lee los frames del manifest), `clip_export.py` (bundle = head
  state_dict + meta + transforms de eval; `predict_clip` como fuente única de inferencia),
  `train_clips.py` (fit/evaluate + orquestación), `train_clips.yaml` + config tipada. 82 tests.
- **Augmentation con impacto medido:** se entrenó sin y con augmentation (flip/crop/color). Resultado
  sobre test: no-aug macro-F1 **0.596** → aug **0.640** (**Δ = +0.045**). Cierra otra sub-técnica del
  electivo de optimización, medida. El bundle exportado es el augmentado (`clips-v1-clips-aug`).
- Todo a MLflow (experimento `clips-cnn-v1`, runs `clips-noaug`/`clips-aug`): params, métricas por
  clase + macro-F1, matriz de confusión. Bundle `clip_model.pt` (533 KB) gitignored.

### Hallazgo (insumo clave para el informe)
- El macro-F1 visual (~0.64) es **menor que el del modelo de ventana tabular (~0.80)** — y eso es
  **esperado y honesto**: el CNN aprende de los píxeles, sin los **atajos de construcción** del
  dataset (ej. `team_is_home=-1`/`visible=0` en background) que inflaban al tabular (ver SHAP de
  3.2). El modelo visual hace trabajo real; es el que sirve para clips subidos.

### Por qué se hizo así / concepto del curso
- **Transfer learning con backbone congelado:** reusar ResNet18 ImageNet como extractor fijo y
  entrenar solo la cabeza → rápido, robusto al overfitting con dataset chico, entrenable local.
- **Anti training-serving skew (invariante 3):** los transforms de eval se serializan en el bundle;
  `predict_clip` es el único camino de inferencia (lo reusará el serving del sub-proyecto 4).
- **Desbalance (invariante 5):** cross-entropy ponderada por la distribución del train; métricas por
  clase + macro-F1, nunca accuracy.
- **Determinismo (invariante 6):** seed de torch/numpy/random; selección por val, test una vez.
- **Optimización (electivo):** augmentation como sub-técnica, con impacto medido (Δ macro-F1).

### Requerimiento de la consigna que cubre
- Núcleo del modelo que habilita la integración de video (técnica avanzada) y la base del Grad-CAM
  (sub-proyecto 3). Suma una sub-técnica de optimización medida.

### Alternativas consideradas y descartadas
- **Fine-tune del backbone / cachear embeddings** → se eligió congelado + en vivo (permite
  augmentation); fine-tune queda como mejora futura medible.
- **Fusión con tabular** → descartada: el flujo de upload es visual-only y lo tabular tiene artefactos.
- **Colab / GPU en la nube** → descartado por **NDA** (subir frames = divulgar a un tercero,
  cláusula 3.c / Exhibit A "do not distribute to tiers"); se entrena local en MPS.

### Limitaciones asumidas
- Dataset chico (762 train) y clases minoritarias (goal/card) → métricas por clase ruidosas.
- Desfase temporal train(8s)/serve(hasta 30s): se resuelve muestreando K frames; a evaluar con el
  serving real.

### Nota técnica (entorno)
- En Apple Silicon, torch y xgboost traen cada uno su OpenMP (libomp): correr el forward de la
  ResNet en el mismo proceso pytest que xgboost segfaultea. Resuelto con `OMP_NUM_THREADS=1` en
  `conftest.py` (el script de entrenamiento real no se ve afectado y mantiene full threading).

### Referencias al código
- `backend/src/models/{clip_model,clip_export,train_clips,clip_config}.py`,
  `backend/src/data/clips_dataset.py`, `configs/train_clips.yaml`.
- Tests: `backend/tests/test_{clip_model,clips_dataset,clip_export,train_clips,clip_config}.py`.
- Spec: `docs/superpowers/specs/2026-06-14-cnn-clips-modelo-design.md`.
  Plan: `docs/superpowers/plans/2026-06-14-cnn-clips-modelo.md`.
- Comando: `uv run python -m src.models.train_clips --config ../configs/train_clips.yaml`.

### Uso de IA generativa
Sub-proyecto desarrollado con Claude Code: brainstorming (backbone, augmentation, visual-only),
spec y plan, implementación TDD vía subagentes con review, diagnóstico del clash OpenMP, y
orquestación del entrenamiento real local. El estudiante decidió la dirección (visual-only,
augmentation con medición, no-Colab por NDA) y validó cada paso; es responsable de poder defenderlo.

---

## 2026-06-14 — Fase 3.5, sub-proyecto 3: Grad-CAM (explicabilidad visual)

### Qué se hizo
- Se implementó **Grad-CAM** para el clip-model: mapas de calor por frame que muestran *qué región*
  de la imagen sostuvo la predicción. Cierra el lado **visual** del electivo de explicabilidad
  (complementa al SHAP tabular de la Fase 3.2).
- `backend/src/models/clip_gradcam.py` (TDD, 2 tasks vía subagentes):
  - `gradcam_clip(model, clip, class_index=None)`: **forward dedicado con gradientes** (el forward
    de entrenamiento corre el backbone bajo `no_grad`), hook sobre `layer4` para capturar
    activaciones y sus gradientes, Grad-CAM **por cada uno de los 8 frames** → `(K,224,224)` en [0,1].
  - `overlay_heatmap(frame, heatmap)`: mezcla el mapa (colormap jet) sobre el frame (lo usarán el
    serving y el frontend).
- Validado **sobre un clip real** (local, NDA-safe — sin mostrar la imagen): clip `corner` →
  Grad-CAM `corner`, heatmaps de forma/rango correctos, overlay `(224,224,3)`. 88 tests verdes.

### Detalle técnico clave
- Para que el gradiente llegue a las activaciones de `layer4` **a través del backbone congelado**,
  el input (`frames`) se marca `requires_grad_(True)`. Sin eso no se construye el grafo y `backward`
  falla. Los pesos siguen congelados (no se reentrena): el grad fluye hacia las activaciones, no
  hacia los pesos.

### Por qué / concepto del curso
- **Explicabilidad de modelos visuales** (electivo): Grad-CAM es la técnica estándar para CNNs.
- **Reuso sin reentrenar** (single source): usa el mismo `backbone`/`head` del modelo entrenado.
- **NDA / gobernanza:** los overlays van sobre frames reales → contenido NDA. Este sub-proyecto
  entrega solo código + tests con frames sintéticos; la visualización real ocurre local en
  serving/frontend. **Coherencia con la decisión no-Colab:** tampoco se muestran frames reales a
  terceros (incluido el asistente) — la validación fue numérica.

### Requerimiento de la consigna que cubre
- Electivo **Explicabilidad** (lado visual). Base del overlay que mostrará el frontend (sub-proyecto 5).

### Alternativas / handoff
- **Por frame (los 8) vs solo el más saliente** → se devuelven los 8; la selección del frame clave
  queda downstream. Nota: la normalización por-frame deja todos los picos en 1.0, así que el serving
  deberá elegir el frame con otro criterio (ej. máximo del cam crudo antes de normalizar).

### Referencias al código
- `backend/src/models/clip_gradcam.py`. Tests: `backend/tests/test_clip_gradcam.py`.
- Spec: `docs/superpowers/specs/2026-06-14-cnn-clips-gradcam-design.md`.
  Plan: `docs/superpowers/plans/2026-06-14-cnn-clips-gradcam.md`.

### Uso de IA generativa
Desarrollado con Claude Code (brainstorming, spec, plan, implementación TDD vía subagentes,
validación numérica NDA-safe). Revisado por el estudiante.

---

## 2026-06-18 — Fase 3.5, sub-proyecto 4: Serving (upload de video)

### Qué se hizo
- **El que conecta el video con el flujo:** endpoint `POST /predict/clip` que recibe un **video**,
  extrae frames, corre el clip-model + Grad-CAM y devuelve clase + probabilidades + overlays.
- Flujo (`src/serving/clip_inference.py`, TDD): `frames_from_video` (guarda el upload en un temp,
  reusa `extract_clip_frames` para muestrear 8 frames a lo largo del clip, **borra el temp** — no
  se persisten imágenes) → `predict_clip` → `gradcam_clip` → `overlay_heatmap` → JPGs en **base64**.
- Endpoint (`src/api/routers/clip_predict.py`) con schema `ClipPredictResponse` (clase,
  probabilidades, versión, `gradcam: [{frame_index, image_base64}]`); 503 si no hay clip-model.
- El clip-bundle se carga en el `lifespan` (best-effort); `clip_model_dir` por config/env.
- **Deps de prod:** `torch`/`torchvision`/`opencv-python-headless` pasan a `[project]`. Dockerfile:
  pre-cachea los pesos ResNet18 y setea `OMP_NUM_THREADS=1`.
- **Verificado en el stack docker real:** `docker compose up --build` → POST de un video →
  **clase + 8 overlays Grad-CAM**; y `/predict` (xgboost) sigue vivo → **torch y xgboost coexisten
  en el mismo proceso sin crash** (confirma el `OMP_NUM_THREADS=1` en runtime). 94 tests verdes.

### Concepto del curso relacionado
- **Serving de modelos / contrato de API:** endpoint multipart, schema estricto, 503 sin modelo.
- **Anti training-serving skew (invariante 3):** la extracción de frames y el transform de eval en
  serving son **los mismos** que en training (se reusan las funciones).
- **Gobernanza de datos / NDA:** el video lo sube el usuario; los frames se procesan en memoria y el
  temp se borra (no se persisten imágenes); la validación fue con video sintético / sin mostrar
  frames reales a terceros.
- **MLOps / contenedores:** imagen con torch (pesada, asumida para la demo), pesos pre-cacheados en
  el build (serving offline), y el manejo del conflicto de runtimes OpenMP.

### Requerimiento de la consigna que cubre
- Integración de **video en el flujo** (técnica avanzada / objetivo central de la Fase 3.5):
  subir un clip → predicción + explicación visual, servido por la API.

### Alternativas / decisiones
- **Misma imagen API + torch vs servicio aparte** → misma imagen (más simple para la demo), con
  `OMP_NUM_THREADS=1` para la coexistencia con xgboost.
- **Pesos del backbone:** pre-descargados en el Dockerfile (bundle chico) vs empaquetar el modelo
  completo → pre-descarga.
- **Devolver los 8 overlays** vs solo el saliente → los 8 (el frontend elige; evita el problema de
  "frame saliente" con la normalización por-frame).

### Problemas encontrados y resueltos (durante la verificación)
- **Disco de Docker lleno** (`OSError: No space left on device` al cachear pesos) → se liberaron
  ~33 GB de imágenes/caché viejos.
- **`opencv-python` rompe en el contenedor slim** (`ImportError: libxcb.so.1`, deps GUI/X11) →
  se cambió a **`opencv-python-headless`** (misma API `cv2`, sin GUI), estándar para contenedores.

### Referencias al código
- `backend/src/serving/clip_inference.py`, `backend/src/api/routers/clip_predict.py`,
  `backend/src/api/{main.py,schemas.py}`, `backend/src/config.py`, `backend/Dockerfile`,
  `docker-compose.yml`, `backend/pyproject.toml`.
- Tests: `backend/tests/test_{clip_inference,clip_predict,clip_schemas,config_clip}.py`.
- Spec: `docs/superpowers/specs/2026-06-18-cnn-clips-serving-design.md`.
  Plan: `docs/superpowers/plans/2026-06-18-cnn-clips-serving.md`.

### Uso de IA generativa
Desarrollado con Claude Code (brainstorming, spec, plan, implementación TDD vía subagentes,
diagnóstico de los problemas de disco/OpenCV, verificación end-to-end en docker). El estudiante
decidió la arquitectura (misma imagen, pesos pre-cacheados) y validó el resultado.

---

## 2026-06-18 — Fase 3.5, sub-proyecto 5: Frontend (modo Video) · cierre de la Fase 3.5

### Qué se hizo
- Se agregó al frontend React un **modo "Video"** (toggle `Ventana | Video`): subir un clip →
  `POST /predict/clip` → mostrar clase, probabilidades y los **overlays de Grad-CAM** (un frame
  grande + tira de 8 miniaturas clickeables). Reusa la pizarra táctica de la Fase 3.3.
- Componentes nuevos: `VideoForm` (upload), `ClipPrediction` (veredicto + Grad-CAM + barras),
  `GradcamViewer` (frame grande + tira). Cliente `predictClip(file)` (multipart, sin CORS).
- **Se sumó capa de tests al frontend:** setup de **Vitest** + helpers puros en `lib/format.ts`
  (`sortedByValueDesc`, `confidencePct`, `clampIndex`) con sus tests. Decisión: unit-testear la
  **lógica pura** (no el rendering trivial); el render/integración se verifica end-to-end.
- Verificado **end-to-end** (frontend dev + API real con ambos modelos): subir un video → clase +
  barras + Grad-CAM real renderizado; el toggle alterna con la demo de ventana.

### Por qué / concepto del curso
- **Visualización/UI** (electivo) extendida al flujo de video: el usuario interactúa con el modelo
  subiendo un clip y ve la explicación visual.
- **Sin CORS, mismo código dev/prod:** rutas relativas `/api/...` vía reverse-proxy (Vite/nginx).
- **Estrategia de testing por capas:** ML/datos/serving con pytest en el backend; lógica pura del
  front con Vitest; rendering con verificación end-to-end. Criterio de *qué* testear en cada capa.
- **NDA:** el video lo sube el usuario y se procesa en memoria (sub-proyecto 4); los overlays se
  muestran en su navegador; la verificación se hizo con video sintético (no se mostraron frames
  reales a terceros).

### Requerimiento de la consigna que cubre
- Cierra el **electivo de Visualización/UI** del lado del video y completa la integración de video
  en el flujo (objetivo central de la Fase 3.5).

### Decisiones
- **Toggle Ventana | Video** (vs todo en una vista / ruta separada): separación limpia sin sumar
  routing.
- **Grad-CAM: frame grande + tira de 8** (vs grilla / scrubber).
- **Agregar Vitest** (vs dejar el front sin tests): cierra el hueco con costo bajo y mejora el
  código (helpers puros DRY).

### Referencias al código
- `frontend/src/components/{VideoForm,ClipPrediction,GradcamViewer}.tsx`,
  `frontend/src/lib/{api.ts,types.ts,format.ts,format.test.ts}`, `frontend/src/App.tsx`,
  `frontend/src/index.css`, `frontend/package.json` (Vitest). Mockup: `frontend/mockups/predict-video.html`.
- Spec: `docs/superpowers/specs/2026-06-18-cnn-clips-frontend-design.md`.
  Plan: `docs/superpowers/plans/2026-06-18-cnn-clips-frontend.md`.

### Cierre de la Fase 3.5 (CNN de clips de video)
Los 5 sub-proyectos quedaron completos: **(1) Datos** (descarga de videos NDA + extracción de
clips), **(2) Modelo** (CNN multi-frame ResNet18 congelada + augmentation medida), **(3) Grad-CAM**,
**(4) Serving** (`/predict/clip` en docker) y **(5) Frontend** (modo Video). El objetivo —subir un
video y que el modelo prediga el evento con explicación visual— quedó funcionando de punta a punta.

### Uso de IA generativa
Sub-proyecto desarrollado con Claude Code (mockup, spec, plan, implementación TDD/Vitest vía
subagentes, verificación end-to-end con Playwright). El estudiante eligió la dirección visual y la
estrategia de testing, y validó el resultado; es responsable de poder defenderlo.

---

## 2026-07-11 — Deploy a AWS Elastic Beanstalk (infraestructura de despliegue)

### Qué se hizo
Se prepararon los artefactos para desplegar el stack completo (api, frontend, mlflow,
prometheus, grafana) en un entorno de AWS Elastic Beanstalk (EB), sin tocar el flujo de
desarrollo local:

- `Dockerfile.deploy.api` (raíz del repo): variante de `backend/Dockerfile` que además
  hornea el modelo entrenado (`models/v0`, `models/clips-v1`) dentro de la imagen, con su
  propio `Dockerfile.deploy.api.dockerignore` que habilita `/models/` solo para este build
  (el `.dockerignore` genérico de la raíz lo excluye como defensa en profundidad del NDA).
- `docker-compose.prod.yml`: variante de deploy del `docker-compose.yml` de dev — `api` y
  `frontend` usan `image:` apuntando a ECR en vez de `build:`, `frontend` publica `80:80`
  (único puerto que EB abre por default), y `api` no monta `./models` porque ya viene
  horneado en la imagen.
- `.ebextensions/security-group.config`: abre los puertos 5500 (MLflow) y 3000 (Grafana)
  en el security group que EB crea, además del 80.
- `scripts/deploy_eb.sh`: automatiza el swap seguro `docker-compose.prod.yml` →
  `docker-compose.yml` (staged en git, nunca commiteado, restaurado con `trap` aunque el
  deploy falle) porque el EB CLI empaqueta el índice de git, no el working tree.
- Sección "Deploy a AWS Elastic Beanstalk" en el README con el flujo completo
  (login ECR → build/push multi-arch → setup IAM/EB → deploy → validación → `eb
  terminate`).

Todavía **no se ejecutó ningún `eb create`/`eb deploy` real**: las credenciales AWS del
entorno estaban inválidas (`InvalidClientTokenId`) al momento de esta sesión, y no existe
un `models/v0` entrenado localmente para hornear. Los archivos quedan listos para correr
en cuanto se resuelvan esos dos bloqueadores.

**Actualización (misma sesión):** al validar credenciales nuevas se detectó que la cuenta
`625067806263` es un **AWS Academy Learner Lab** (rol asumido `voclabs`), no una cuenta
IAM normal. `iam:AttachRolePolicy`/`CreateRole` devuelven `AccessDenied` — el plan
original de crear/adjuntar la policy `AmazonEC2ContainerRegistryReadOnly` a
`aws-elasticbeanstalk-ec2-role` no es viable ahí. El lab ya provee `LabRole` /
`LabInstanceProfile` con esa policy adjunta y con `elasticbeanstalk.amazonaws.com`
habilitado en su trust policy, así que el README se corrigió para usarlos directo
(`eb create ... --instance-profile LabInstanceProfile --service-role LabRole`) sin tocar
IAM. Además las credenciales del lab son temporales y expiran cada pocas horas.

### Por qué se hizo así
- **Plataforma Docker (no ECS-managed) de EB, vía `docker-compose.yml`.** Investigado
  contra la documentación oficial de AWS: el multicontainer Docker sobre AL1
  (`Dockerrun.aws.json` v2) está retirado desde 2022; su sucesor "ECS on AL2023" exige
  imágenes prebuilds para *todos* los servicios (no permite `build:` en el propio
  despliegue). La plataforma Docker "plana" sí soporta `docker compose up --build`
  directo en la instancia — igual que en local — lo que nos permite mezclar servicios con
  `image:` (api/frontend, desde ECR) y servicios con `build:` (mlflow, sin ECR repo
  propio) sin duplicar infraestructura para 3 servicios que no lo necesitan.
- **Modelo horneado en la imagen, no traído de S3 en runtime.** `models/` está gitignored
  y se bind-mountea en local a propósito (permite reentrenar sin rebuildear la imagen).
  En EB no hay host que lo provea. Se evaluó traerlo de S3 al arrancar el contenedor
  (más "productivo", separa modelo de imagen igual que en local) pero se descartó por
  tiempo/alcance: exige bucket S3, política IAM adicional para la instancia, y tocar el
  entrypoint del backend. Hornear el modelo en una imagen de *deploy* separada
  (`Dockerfile.deploy.api`, no se toca `backend/Dockerfile` de dev) resuelve el problema
  con cero infraestructura nueva; el costo es tener que reconstruir y re-pushear la
  imagen en cada reentrenamiento — aceptable para la cadencia de este obligatorio.
- **`Dockerfile.deploy.api.dockerignore` en vez de editar el `.dockerignore` de la raíz.**
  El `.dockerignore` genérico excluye `/models/` como defensa en profundidad del NDA
  (evitar que builds con contexto raíz filtren accidentalmente contenido gitignored).
  BuildKit permite un ignore-file específico por Dockerfile que pisa al genérico solo
  para ese build — mantiene la protección general intacta y hace explícita, en un único
  lugar, la única excepción deliberada.
- **Single-instance, sin load balancer.** Decisión tomada con el usuario: es una demo
  académica, no un servicio con SLA; evita el costo fijo de un ALB.
- **Swap de compose vía `git add` (staged, sin commit) en vez de mantener un solo
  `docker-compose.yml` con `image:`/`build:` mezclados condicionalmente.** Compose no
  soporta bien "usar `build:` en local, `image:` en EB" desde el mismo archivo sin
  trucos frágiles (variables de entorno para pisar claves). Dos archivos explícitos +
  un script que hace el swap transitorio es más legible y así el archivo que ve un
  lector de dev nunca contiene referencias a ECR.

### Concepto del curso relacionado
- **Reproducibilidad de deploy / training-serving skew:** la misma imagen versionada en
  ECR (`:latest` + tag por commit) que se prueba local es la que corre en AWS — no hay
  un segundo build "de prod" con código distinto.
- **Trazabilidad de ML (Electivo 1):** las imágenes en ECR quedan versionadas junto al
  registro de modelos en MLflow; el tag por commit permite correlacionar una imagen
  desplegada con el commit exacto que la generó.
- **IaC básica:** `.ebextensions/*.config` es infraestructura declarada como código
  (CloudFormation embebido) en vez de clicks manuales en la consola.
- **Gestión de permisos (IAM) en un entorno restringido:** el instance profile de EB
  necesita la policy `AmazonEC2ContainerRegistryReadOnly` para poder autenticar contra
  ECR privado sin embeber credenciales en la imagen. En una cuenta IAM normal esa policy
  se adjuntaría explícitamente (EB ya no crea `aws-elasticbeanstalk-ec2-role` por
  default); en esta cuenta (Learner Lab, permisos IAM de solo lectura) la lección es
  distinta: identificar y reusar el rol preexistente con los permisos correctos
  (`LabRole`) en vez de asumir que uno puede crear infraestructura IAM propia.

### Requerimiento de la consigna que cubre
- Extiende el mínimo **"stack completo en Docker Compose"** (Fase 2) a un despliegue real
  en la nube, más allá de `localhost`.
- Apoya el electivo **"Trazabilidad de ML"** (manifiestos + imágenes versionadas en un
  registro, no solo el Model Registry de MLflow).

### Alternativas consideradas y descartadas
- **ECS managed Docker platform + `Dockerrun.aws.json` v2** → descartado: exige todas las
  imágenes prebuilds (no serviría para `mlflow`, que no tiene ECR repo), y es más
  infraestructura (cluster ECS, task definitions) de la que necesita una demo académica
  de una sola instancia.
- **Fetch del modelo desde S3 al iniciar el contenedor** → descartado por ahora (ver
  arriba); queda anotado como mejora natural si el proyecto necesitara reentrenar sin
  rebuildear la imagen de deploy.
- **Load-balanced + auto-scaling** → descartado: sobra para el caso de uso (demo/entrega),
  agrega costo fijo (ALB) sin beneficio.

### Referencias al código
- `Dockerfile.deploy.api`, `Dockerfile.deploy.api.dockerignore`, `docker-compose.prod.yml`,
  `.ebextensions/security-group.config`, `scripts/deploy_eb.sh`, sección "Deploy a AWS
  Elastic Beanstalk" en `README.md`.

### Uso de IA generativa
Investigación de la documentación oficial de AWS Elastic Beanstalk (estado de las
plataformas Docker, requisitos de `Dockerrun.aws.json` v2 vs. `docker-compose.yml`,
permisos IAM para ECR) y redacción de los artefactos de deploy con Claude Code, en modo
plan con aprobación explícita del usuario en las decisiones de arquitectura (scope del
stack, estrategia de modelo, topología del entorno). Pendiente de ejecución real por el
estudiante una vez resueltos los bloqueadores de credenciales y modelo entrenado.

---

## 2026-07-11 — Primer entrenamiento real del baseline v0 (para desbloquear el deploy)

### Qué se probó
Se corrió por primera vez en esta máquina el pipeline completo de datos + entrenamiento
que hasta ahora solo existía como código (nunca se había ejecutado localmente): descarga
de SoccerNet (16 partidos de `england_epl`, labels + features ResNet pre-extraídas +
videos 224p, `configs/dataset.yaml`), construcción del manifest de ventanas
(`src.data.build_dataset`), y entrenamiento del baseline v0 (`src.models.train`,
LogReg + XGBoost, late fusion tabular ⊕ embedding ResNet pooled).

**Resultado — manifest:** 1140 ventanas de 16 partidos, splits por `game_id`
(train=762, val=192, test=186), clases `{background: 760, corner: 185, substitution: 86,
card: 64, goal: 45}` (confirma el desbalance esperado, invariante 5).

**Resultado — entrenamiento** (experimento MLflow `baseline-v0`, commit `d03dd9c`,
`dataset_hash=44a977f9...`):

| modelo | val macro-F1 | test macro-F1 |
|---|---|---|
| logistic_regression | 0.741 | **0.821** |
| xgboost | **0.877** | 0.797 |

Se exportó **xgboost** (mejor val macro-F1, criterio de selección del script) a
`models/v0/model.joblib`. Por clase en test: `background` F1=1.00, `corner` F1=0.85,
`substitution` F1=0.94, `card` F1=0.70, `goal` F1=0.50 (la clase más golpeada por tener
solo 45 ejemplos en todo el dataset — esperable con 16 partidos).

Es el **primer run del experimento**, no hay un baseline anterior contra el cual
comparar — este entrenamiento *es* el baseline v0 que exige el roadmap de modelos antes
de cualquier mejora.

### Por qué se hizo así / bloqueadores resueltos en el camino
- **`uv` no estaba instalado en la máquina** (proyecto trabajado hasta ahora desde otra
  máquina/sesión) → se instaló vía `brew install uv`.
- **`SOCCERNET_PASSWORD` no estaba seteada.** Se decidió con el usuario pedirle la
  contraseña del NDA para bajar el dataset completo (labels+features+videos) en vez de
  deshabilitar `clips.enabled` temporalmente solo para desbloquear v0 — así de paso queda
  el material para entrenar `clips-v1` (modelo de video) más adelante. El usuario la
  seteó él mismo en `.env` sin pegarla en el chat (mismo criterio de higiene que con las
  credenciales AWS).
- **XGBoost no cargaba (`libxgboost.dylib` no encontraba `libomp.dylib`)**: falta común en
  macOS con Homebrew — se resolvió con `brew install libomp` (xgboost linkea contra esa
  lib para paralelismo OpenMP, no viene embebida en el wheel de macOS).
- Se corrió el training tal cual está en `configs/train.yaml`, sin tocar hiperparámetros:
  el objetivo de esta corrida no es optimizar sino **tener un `models/v0` real para poder
  hornearlo en la imagen de deploy** (ver entrada anterior de esta bitácora).

### Concepto del curso relacionado
- **Invariante 1 (anti data-leakage):** splits por `game_id`, verificado en el summary del
  build (`report/dataset_summary.json`) — ningún partido se repite entre train/val/test.
- **Invariante 5 (desbalance de clases):** se reportan P/R/F1 por clase + PR-AUC en las 5
  clases, nunca accuracy sola; el desbalance real (760 `background` vs. 45 `goal`) se ve
  directo en los números de soporte (`n=`) por clase.
- **Invariante 6 (determinismo):** `seed=42` logueado como param en MLflow junto con el
  commit (`mlflow.source.git.commit`) y el hash del dataset — cualquiera puede reproducir
  exactamente este run.
- **Reproducibilidad de entorno:** el problema de `libomp` es un caso concreto de
  "funciona en mi máquina" — la imagen Docker de la API (que sí corre en Linux, con
  xgboost linkeado distinto) no tiene este problema; es puramente del entorno de
  desarrollo local en macOS.

### Requerimiento de la consigna que cubre
- Cierra el mínimo **"v0 baseline debe existir antes de cualquier mejora"** del roadmap de
  modelos — hasta ahora era código sin ejecutar, ahora hay un `models/v0` real.
- Alimenta directamente el mínimo de **Fase 2 (baseline end-to-end)**: sin esto, el deploy
  a AWS solo podía mostrar `/predict` devolviendo 503.

### Decisión
**Adoptar.** Este es el primer y único baseline v0 disponible; se usa tal cual para
hornear `Dockerfile.deploy.api` y desplegar a Elastic Beanstalk. Iterar sobre
hiperparámetros/features queda para la Fase 3 (fuera del alcance de esta sesión, que
prioriza cerrar el ciclo de deploy).

### Referencias al código
- `configs/dataset.yaml`, `configs/train.yaml`, `backend/src/data/{download,build_dataset}.py`,
  `backend/src/models/train.py`. Artefactos generados (gitignored): `data/processed/manifest.parquet`,
  `data/processed/resnet_pooled.npy`, `models/v0/model.joblib`. Resumen:
  `report/dataset_summary.json`. MLflow: experimento `baseline-v0`, run `xgboost`
  (run_id `138ed32d8d0041248919f583adf605a6`).

### Cierre didáctico
- El baseline v0 no es "el mejor modelo posible": es la evidencia mínima de que el ciclo
  dato→feature→modelo→métrica cierra, tal como pide el criterio de corrección del
  obligatorio. Con 16 partidos y clases con 45-86 ejemplos, las métricas van a ser
  ruidosas — eso es esperable y no invalida el ciclo.
- XGBoost ganó por val macro-F1 pero perdió contra LogReg en test macro-F1 (0.797 vs
  0.821): con datasets chicos y clases minoritarias, la elección "mejor modelo" por una
  sola corrida puede no ser estable — es un punto de discusión válido para el informe
  final (¿convendría selección por CV en vez de un único split val?).
- El error de `libomp` es un buen ejemplo real de por qué el modelo que sirve la API corre
  en Docker (Linux) y no depende del entorno de desarrollo de quien entrena: la imagen no
  hereda estos problemas de librerías nativas del host.

### Uso de IA generativa
Pipeline ejecutado con Claude Code (skill `experimento` del proyecto): decisiones de
alcance (contraseña NDA, camino con clips habilitado) tomadas junto con el usuario;
diagnóstico y resolución de los tres bloqueadores del entorno (uv, password, libomp) y
verificación de los tags/métricas en MLflow hechos por el asistente. El usuario proveyó la
contraseña del NDA de forma segura (directo en `.env`) y es responsable de poder explicar
los resultados y la decisión de adopción.

---

## 2026-07-11 — Deploy a AWS Elastic Beanstalk: primera ejecución real

### Qué se hizo
Con `models/v0` ya entrenado, se ejecutó el deploy planeado en la entrada anterior:

1. Build + push a ECR de `soccer-net/api` (con el modelo horneado, `Dockerfile.deploy.api`)
   y `soccer-net/frontend`, tageadas `:latest` y `:d03dd9c` (commit).
2. `eb init -p docker soccer-net --region us-east-1` + `eb create soccer-net-prod --single
   --instance_type t3.medium --instance_profile LabInstanceProfile --service-role LabRole`.
3. **`eb create` funcionó al primer intento** (`Health: Green` en ~5 minutos) — el swap
   temporal `docker-compose.yml` ↔ `docker-compose.prod.yml` vía dos commits (uno de ida,
   uno de vuelta alrededor del `eb create`) resultó necesario porque `eb create`, a
   diferencia de `eb deploy`, no soporta `--staged`.
4. Validación end-to-end: frontend (200), `/api/health`, `/api/model-info` (modelo XGBoost
   cargado, coincide con el run entrenado), **`/api/predict` con un payload real devolvió
   una predicción + explicaciones SHAP** — el ciclo completo cierra en la nube. Grafana
   accesible. MLflow devolvía 403.
5. MLflow 403 resuelto en dos iteraciones: la protección anti DNS-rebinding de MLflow 3
   (`--allowed-hosts`) necesitaba explícitamente el patrón de dominio de EB, y **además**
   el puerto (`*.elasticbeanstalk.com:5500`) porque el Host header del cliente lo incluye
   y el wildcard no lo cubre solo. Cada fix se validó leyendo los logs reales de la
   instancia (`eb logs --all`) en vez de asumir — el primer intento parecía razonable
   pero los logs mostraron que el middleware quedaba activo con el allowlist "correcto"
   y aun así rechazaba, lo que apuntó al problema real (puerto en el Host header).
6. **Bloqueado a mitad del segundo redeploy**: la sesión del AWS Academy Learner Lab
   terminó (policy de deny explícito `voc-cancel-cred` aplicada por la plataforma del
   lab), cortando todo acceso AWS (`ec2:DescribeInstances`, `s3:CreateBucket`, etc.) justo
   después de que el redeploy con el segundo fix de MLflow completara exitosamente según
   los logs de EB. No se pudo re-verificar el estado final de MLflow tras ese último
   redeploy. Pendiente: retomar con credenciales nuevas del lab y confirmar `eb status` +
   los 4 endpoints.

### Por qué se hizo así
- **Verificar con logs reales, no asumir.** El primer intento de arreglar MLflow (agregar
  `*.elasticbeanstalk.com` a `--allowed-hosts`) era razonable pero insuficiente. En vez de
  suponer que "seguro es algo de la imagen vieja" o iterar a ciegas, se leyeron los logs de
  `docker compose logs` en la instancia (vía `eb logs --all`) y se confirmó que el
  middleware SÍ tenía el allowlist nuevo aplicado — lo que descartó la hipótesis de imagen
  desactualizada y señaló el problema real (el puerto en el Host header).
- **CPU-only torch (bug encontrado durante el build, no planeado).** Ver detalle en el
  commit `fix: resolver torch/torchvision a wheels CPU-only en Linux` — reducir la imagen
  de api de una descarga con timeout (>2GB de CUDA) a ~1GB, sin tocar la resolución local
  en macOS (marker `sys_platform == 'linux'`).
- **Compute local para builds (buildx + QEMU) tuvo fricción real:** dos rondas de "no
  space left on device" en el disco virtual de Docker Desktop (58.6GB fijos), resueltas
  liberando build cache viejo. Vale la pena documentarlo porque es una limitación de
  correr el pipeline de deploy desde una laptop, no del diseño de la infra en sí.

### Concepto del curso relacionado
- **Verificación empírica sobre suposición:** ante un fix que "debería funcionar" y no
  funciona, la respuesta correcta es leer los logs del sistema real (la instancia EB) en
  vez de iterar a ciegas sobre hipótesis — mismo principio que depurar un modelo mirando
  las predicciones reales en vez de asumir por qué falla.
- **Reproducibilidad de entorno (Linux vs. macOS):** el problema de torch CUDA solo
  aparece al cross-buildear para Linux (target real de producción) — otro ejemplo de por
  qué el entorno de serving no puede inferirse del entorno de desarrollo.
- **Infraestructura efímera (AWS Academy Lab):** el corte de sesión a mitad de un deploy
  es justamente el tipo de evento que un pipeline de CI/CD real necesita tolerar
  (reintentable, idempotente) — `scripts/deploy_eb.sh` y el flujo de `eb deploy` ya son
  seguros para reintentar sin dejar el repo en un estado inconsistente (el swap se
  revierte siempre, incluso si el deploy en sí falla).

### Requerimiento de la consigna que cubre
- Extiende el mínimo end-to-end (Fase 2) a un despliegue real verificado con una
  predicción real de punta a punta, no solo "la infra levanta".

### Decisión
**En progreso, no cerrado.** El entorno EB llegó a `Health: Green` con las 4 piezas
funcionando (frontend, api con predicción real, grafana; mlflow con el fix aplicado pero
sin re-verificar tras el corte de sesión). Se retoma en la próxima sesión con credenciales
nuevas del lab.

### Referencias al código
- Commits de esta sesión (deploy real): `f41ef61` (swap temporal), `751663e` (revert),
  `e40b082` y `0e5dc6f` (fixes de MLflow `--allowed-hosts`).
- `mlflow/Dockerfile`, `scripts/deploy_eb.sh`.

### Uso de IA generativa
Ejecución completa (builds, `eb create`/`eb deploy`, diagnóstico de los 403 de MLflow vía
logs de la instancia, diagnóstico del corte de sesión del lab) hecha por Claude Code. El
usuario aprobó explícitamente el paso de `eb create` antes de que se ejecutara (acción
facturable) y proveyó la contraseña NDA y las credenciales AWS de forma directa (sin
pegarlas en el chat cuando fue posible). Pendiente de retomar y cerrar la validación final.

---

## 2026-07-11 — Deploy a AWS EB: cierre (instancia rota diagnosticada y reemplazada)

### Qué se hizo
Al recuperar acceso a AWS tras el reinicio del lab, el entorno seguía en `Health: No
Data` en vez de recuperarse solo. Se diagnosticó **sin usar SSH** (no había keypair
configurado) vía **AWS Systems Manager Run Command** (`aws ssm send-command`,
disponible porque `LabRole` ya tenía `AmazonSSMManagedInstanceCore` adjunta):
`docker.service` estaba `inactive (dead)` y no existía ni un solo archivo de log de
`cfn-init`/`eb-engine` en la instancia — el bootstrap murió en el primer paso (la
descarga del script de arranque de EB desde S3), exactamente en la ventana de tiempo en
que el deny policy `voc-cancel-cred` del lab estaba activo. La instancia nunca llegó a
tener estado ni aplicación corriendo, así que se terminó manualmente
(`aws ec2 terminate-instances`) para que el Auto Scaling Group de EB lanzara una
reemplazante — que bootstrapeó limpio en ~3 minutos con la sesión del lab ya estable.

**Resultado final, verificado con requests reales:** frontend (200), `/api/health`,
`/api/model-info` (modelo cargado), **`/api/predict` con un payload real devuelve
predicción + SHAP**, **MLflow (5500): 200**, **Grafana (3000): 200**. El deploy a AWS
Elastic Beanstalk queda cerrado y funcionando de punta a punta.

### Por qué se hizo así
- **Diagnosticar antes de reintentar a ciegas.** Ante "no se recupera solo", la tentación
  fácil es simplemente reintentar `eb deploy` repetidas veces. En cambio se usó SSM Run
  Command para *ver* el estado real de la instancia (servicios, logs) y confirmar la
  causa raíz exacta antes de actuar — mismo principio que con el diagnóstico de MLflow.
- **Terminar en vez de esperar a que se autorepare.** Como `docker.service` nunca arrancó,
  no había ningún proceso vivo que pudiera reintentar el bootstrap por su cuenta; el único
  mecanismo de recuperación real en una arquitectura de Auto Scaling Group es reemplazar
  la instancia. Se confirmó con el usuario antes de terminarla (acción destructiva) aunque
  el riesgo era bajo (instancia sin estado ni aplicación corriendo).

### Concepto del curso relacionado
- **Infraestructura efímera y auto-healing:** un ASG de una sola instancia igual da
  resiliencia ante un fallo de arranque, siempre que alguien (o un proceso automatizado)
  dispare el reemplazo — acá se hizo manual, pero el patrón es el mismo que un health
  check automatizado en un pipeline real.
- **Observabilidad sin acceso directo:** SSM Run Command permitió diagnosticar sin abrir
  puertos SSH ni generar un keypair nuevo — superficie de ataque mínima, resuelto con los
  permisos que ya traía `LabRole`.

### Decisión
**Cerrado.** El entorno `soccer-net-prod` está en `Health: Ok`, con los 5 servicios del
stack corriendo y verificados con tráfico real. URL:
`soccer-net-prod.eba-cyqdfnnq.us-east-1.elasticbeanstalk.com`.

### Referencias al código
- Sin cambios de código en esta entrada (fue diagnóstico + operación de infraestructura).
  Comandos documentados en el README ("Deploy a AWS Elastic Beanstalk").

### Uso de IA generativa
Diagnóstico vía SSM y decisión de terminar la instancia hechos por Claude Code, con
confirmación explícita del usuario antes de la acción destructiva (`terminate-instances`).
Validación final de los 5 endpoints hecha por el asistente con requests reales.

---

## 2026-07-14 — Electivos y mejora del modelo de imágenes (CNN de clips)

### Qué se hizo
Sesión enfocada en **consolidar los electivos sobre el modelo de imágenes** (el CNN de clips,
que es el producto real) y **mejorar sus resultados**. Rama `feat/cnn-electivos-mejoras`.

1. **Trazabilidad (MLflow Model Registry):** el CNN ahora se **registra** en el registry
   (`soccernet-events-clips-v1`), no solo como bundle en disco (antes el `register_model` vivía
   solo en v0). Así el electivo de trazabilidad —versionar experimentos, datos y modelos— queda
   validado sobre el modelo de imágenes, sin depender de v0.
2. **Mejora del modelo (fine-tune del backbone):** se descongeló el último bloque (`layer4`) de
   la ResNet18 con LR diferencial (backbone más bajo que la cabeza), config-driven. El bundle
   ahora serializa los pesos del backbone cuando hay fine-tune (anti training-serving skew).
   Resultado: **test macro-F1 0.640 → 0.672**. Se probó `layer3+layer4` y dio **peor
   validación** (0.648 vs 0.708) → overfitting; se eligió `layer4` **por validación** (no por
   test).
3. **Más datos + corrección de sesgo:** se escaló 16 → 32 → **44 partidos**. Al ver que los
   primeros 32 eran **todos EPL y Chelsea en 16/32** (sesgo de selección), se sumaron 12
   partidos de **5 ligas más** (LaLiga, Champions, Serie A, Bundesliga, Ligue 1) vía `game_ids`
   explícito reproducible. Resultado: **test 0.672 → 0.757** (val 0.708 → 0.804); `goal`
   F1 0.564 → 0.717. El test ahora incluye partidos de ligas no vistas → mide generalización.
4. **Optimización — 2ª sub-técnica (quantization):** static PTQ int8 del backbone (FX/qnnpack),
   con **impacto medido**: tamaño **4x menor** (44.8→11.2 MB), F1 intacto (0.7567→0.7566), pero
   latencia **3x más lenta en ARM/qnnpack** (sin kernels int8 optimizados). Hallazgo honesto:
   la quantization es hardware-dependiente; en el host x86 de serving (fbgemm) el panorama
   difiere. Se reporta como experimento, no se despliega por defecto.
5. **Explicabilidad — fix de Grad-CAM:** se cambió la normalización de **por-frame a compartida**
   entre los 8 frames del clip (la per-frame estiraba el ruido de frames de baja señal → "manchas
   sin sentido"). Verificado: en un gol resalta el área/arco y a los jugadores festejando.
6. **Ejemplos de inferencia:** set de clips demo desde el **split de test** (partidos no vistos),
   incluidos 2 cross-liga; 7/8 correctos. **Caso destacado:** un clip externo de Champions con el
   **cartel del cuarto árbitro** visible → el modelo predice `substitution` 0.91 y el **Grad-CAM
   se enfoca en los números LED del cartel** (feature interpretable). Ver `report/inference_examples.md`.

### Por qué se hizo así / concepto del curso
- **Selección por validación, nunca por test** (data leakage): tanto en `layer3` vs `layer4`
  como en la elección del mejor epoch. El caso `layer3` (mejor test pero peor val) es el ejemplo
  didáctico de por qué no se mira test para elegir.
- **Diversidad > cantidad:** más datos de la misma liga no arregla la generalización; el sesgo de
  selección (tomar los primeros N de una lista agrupada) se corrige muestreando cross-liga.
- **Medir el impacto de la optimización** (electivo): la quantization se evaluó en métricas ML,
  latencia y tamaño — el resultado "no ayuda en latencia acá pero sí en tamaño" es una evaluación
  válida y muestra que el impacto es hardware-dependiente.
- **Anti training-serving skew:** al fine-tunear, los pesos del backbone cambian y deben
  serializarse en el bundle (test de round-trip agregado), o el serving reconstruiría un backbone
  ImageNet distinto.

### Requerimiento de la consigna que cubre
- **Electivos, todos validados sobre el modelo de imágenes:** Trazabilidad (MLflow registry +
  manifest hasheado), Visualización (frontend + Grad-CAM), **Optimización con ≥2 sub-técnicas
  medidas** (data augmentation + quantization), Explicabilidad (Grad-CAM, con evidencia del
  cartel). AutoML no se usó (electivo no elegido; además mantenerlo afuera deja Visualización
  como electivo que suma).

### Alternativas consideradas y descartadas
- **`layer3+layer4`** → peor validación (overfitting), descartado.
- **Quedarse en 32 EPL** → se diversificó a 44 cross-liga por el sesgo de liga.
- **Dynamic quantization** → descartada frente a static PTQ: solo cuantiza Linear (la cabeza),
  no el backbone Conv que domina el cómputo.
- **Baseline v0** → se deja fuera del relato del informe (no corre sobre un clip subido; su valor
  fue cerrar el ciclo end-to-end, no ser el producto).

### Referencias al código
- `backend/src/models/{train_clips,clip_model,clip_config,clip_export,clip_gradcam,quantize}.py`,
  `configs/{train_clips,dataset,splits}.yaml`, `report/inference_examples.md`.
- Tests: `test_clip_export` (round-trip fine-tune), `test_quantize`, `test_leakage`.
- Modelo final: `clips-v1-clips-aug-ft`, test macro-F1 **0.757**, MLflow `soccernet-events-clips-v1`.

### Uso de IA generativa
Sesión desarrollada con Claude Code: planificación de los electivos, implementación (fine-tune
config-driven, módulo de quantization, fix de Grad-CAM) con TDD y lint, orquestación de las
corridas reales (descargas NDA, reconstrucción del dataset, reentrenamientos) y verificación
visual (Grad-CAM, frames). El estudiante tomó todas las decisiones de dirección (mejorar el
modelo, diversificar cross-liga, aportar el clip del cartel) y validó cada resultado; es
responsable de poder defenderlo.
