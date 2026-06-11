# Clasificador de Eventos SoccerNet — ML en Producción (Obligatorio)

Sistema de ML end-to-end que clasifica ventanas de video de partidos de fútbol en
tipos de evento (`goal`, `card`, `substitution`, `corner`, `background`),
combinando **datos no estructurados** (frames extraídos de los videos de
SoccerNet) con **features tabulares** (minuto de juego, mitad, contexto de score,
liga, etc.).

Curso: Machine Learning en Producción — Máster, Universidad ORT Uruguay.
Entrega: **15/07/2026 21:00** (zip ≤ 40MB en gestion.ort.edu.uy + repo de GitHub
Classroom).

**Criterio de corrección (textual de la consigna): se prioriza el desarrollo
end-to-end por encima del rendimiento del modelo.** Ante la duda, siempre
preferir cerrar el ciclo completo antes que mejorar una métrica.

## Modo docente y trazabilidad pedagógica (OBLIGATORIO en cada sesión)

El usuario es el responsable académico de este trabajo y debe poder explicar y
defender cada decisión ante los docentes. Por lo tanto:

1. **Explicar antes de implementar.** Para toda tarea no trivial, presentar
   primero un plan breve con el *porqué* de cada decisión técnica, y esperar
   confirmación. Usar lenguaje claro, sin asumir que el usuario ya conoce la
   herramienta o técnica.
2. **Vincular cada decisión con el curso y la consigna.** Al explicar, indicar
   explícitamente: (a) qué concepto de ML en producción está en juego (ej. data
   leakage, training-serving skew, reproducibilidad, drift, contrato de API,
   versionado de modelos) y (b) qué requerimiento del obligatorio satisface
   (mínimo o electivo, citando la sección de la consigna en `docs/consigna.md`).
3. **Mantener la bitácora `report/bitacora.md`.** Al completar cada tarea,
   agregar una entrada con: fecha, qué se hizo, por qué se hizo así, concepto del
   curso relacionado, requerimiento de la consigna que cubre, alternativas
   consideradas y descartadas (con motivo), y referencias al código
   (archivos/commits). Esta bitácora es insumo directo del informe final y de la
   declaración de uso de IA generativa que exige la facultad.
4. **Resumen didáctico al cierre.** Al terminar cada sesión o tarea grande,
   cerrar con un resumen de 3-5 puntos de "qué aprendimos / qué deberías poder
   explicar de esto", en español.
5. Si el usuario pide "explicame X", priorizar la explicación conceptual sobre
   el detalle de implementación, conectando con lo visto en el curso.

## Arquitectura

```
SoccerNet (pip) ─► ingesta ─► constructor de dataset por ventanas ─► training ─► registro MLflow
                                      │                                              │
                                      ▼                                              ▼
                                notebooks EDA                            FastAPI (online + batch)
                                                                                     │
                                                       React UI ─────────────────────┤
                                                       Prometheus/Grafana ───────────┘
                                                       (todo vía Docker Compose)
```

## Estructura del repo (monorepo)

```
.
├── CLAUDE.md
├── README.md                  # cómo correr todo; documentación de uso de la API
├── docker-compose.yml         # api + frontend + mlflow + prometheus + grafana
├── docs/
│   └── consigna.md            # resumen estructurado de la letra del obligatorio
├── backend/
│   ├── pyproject.toml         # deps separadas: [project] = prod, [dependency-groups] dev
│   ├── Dockerfile
│   ├── src/
│   │   ├── data/              # descarga SoccerNet, extracción de frames, manifests
│   │   ├── features/          # TODO el preprocesamiento y feature engineering (ver invariantes)
│   │   ├── models/            # entrenamiento, evaluación, tuning, export
│   │   ├── api/               # app FastAPI, routers, schemas pydantic
│   │   └── monitoring/        # logging de inputs, chequeos de drift, métricas prometheus
│   ├── tests/
│   └── notebooks/             # EDA + análisis de experimentos (solo consumen de src/)
├── frontend/                  # React (Vite) + Tailwind + shadcn/ui
│   ├── Dockerfile             # build → estáticos → nginx
│   └── src/
├── configs/                   # yaml: spec del dataset, training, api
├── data/                      # gitignored. raw/ interim/ processed/ (ver política de datos)
├── models/                    # gitignored. artefactos exportados; trackeados vía MLflow
└── report/
    ├── bitacora.md            # bitácora pedagógica (ver Modo docente)
    └── ...                    # informe (español), diagramas de arquitectura
```

## Invariantes duras — nunca violarlas

Existen para prevenir los dos modos de falla que el curso corrige con más peso:
**data leakage** y **training-serving skew**.

1. **Los splits son SIEMPRE por `game_id`.** Nunca por ventana, frame ni fila
   aleatoria. Frames/ventanas del mismo partido jamás pueden aparecer en más de
   uno de train/val/test. La asignación de splits vive en un manifest versionado
   (`configs/splits.yaml` o `data/processed/splits.parquet` generado), nunca se
   recalcula ad hoc.
2. **Corrección point-in-time en features tabulares.** Toda feature tabular de
   una ventana en el instante `t` solo puede usar información disponible hasta
   `t` (ej. score acumulado, no score final; eventos hasta el momento, no eventos
   por partido).
3. **Una única fuente de verdad de preprocesamiento.** Toda la lógica de
   features/transformaciones vive en `backend/src/features/` y la importan TANTO
   el training COMO la API. Nunca duplicar ni reimplementar preprocesamiento en
   `api/` ni en notebooks. Los transformadores fiteados (scalers, encoders,
   config de transforms de imagen) se serializan junto al modelo y la API los
   carga — nunca se re-fitean en serving.
4. **Los schemas pydantic son el contrato de la API.** Validación estricta
   (`model_config = ConfigDict(extra="forbid")`). Los mismos tipos respaldan
   `/predict` y `/predict/batch`.
5. **El desbalance de clases se maneja explícitamente.** `background` domina.
   Reportar precision/recall/F1 por clase y PR-AUC; nunca reportar accuracy a
   secas como métrica única.
6. **Determinismo donde sea posible.** Seedear todo (numpy, torch, random);
   loguear el seed y la config completa a MLflow en cada corrida.

## Política de datos (NDA — legalmente vinculante)

Los videos crudos de SoccerNet están protegidos por copyright y se obtuvieron
bajo un NDA con KAUST.

- **NUNCA comitear a git:** videos `.mkv`, frames/imágenes extraídas, la
  contraseña del NDA, ni archivos comprimidos que los contengan. `data/` y
  `models/` están gitignored.
- La contraseña del NDA se lee de la variable de entorno `SOCCERNET_PASSWORD`
  (en `.env`, gitignored). Nunca hardcodearla, nunca imprimirla en logs.
- El repo contiene solo **código, configs y manifests** que permiten regenerar
  el dataset a cualquiera que tenga su propia contraseña del NDA.
- Las labels (`Labels-v2.json`) y las features pre-extraídas se descargan sin
  contraseña; los videos sí la requieren.

## Especificación del dataset (v0 — mantenerlo chico)

- Fuente: SoccerNet v2 vía `pip install SoccerNet`, `SoccerNetDownloader`.
- Subconjunto: ~30–50 partidos (definido por config en `configs/dataset.yaml`),
  videos 224p.
- Construcción de ventanas: por cada evento anotado de las 4 clases objetivo,
  extraer frame(s) en el timestamp anotado (±2s). Las ventanas `background` se
  muestrean de timestamps a ≥30s de cualquier anotación, submuestreadas a un
  ratio configurable (ej. 2:1 background:evento) — NO conservar el ratio natural.
- Cada fila del manifest del dataset (parquet): `game_id, half, position_ms,
  label, frame_path(s), features tabulares..., split`.
- Features tabulares (point-in-time, ver invariante 2): minuto, mitad,
  diferencia de score acumulada, liga, local/visitante del equipo anotado, flag
  de visibilidad, cantidad de eventos hasta el momento, tiempo desde el último
  evento.

## Roadmap de modelos

1. **v0 baseline (debe existir antes de cualquier mejora):** XGBoost o LogReg
   sobre [features ResNET pre-extraídas de SoccerNet (pooled) ⊕ tabular]. Solo
   CPU, rápido.
2. **v1:** ResNet18/MobileNetV3 fine-tuneado sobre nuestros frames; late fusion
   con tabular (concatenar embedding de la CNN + tabular en una cabeza MLP
   chica).
3. **v2 (solo si hay tiempo):** ventanas multi-frame, tuning con Optuna,
   quantization (medir impacto en latencia — la consigna exige evaluar el
   impacto de las optimizaciones).

## API

- FastAPI. Endpoints:
  - `POST /predict` — online: una imagen (base64 o multipart) + campos tabulares
    → clase + probabilidades por clase + versión del modelo.
  - `POST /predict/batch` — lista de items o upload de CSV → resultados (sync en
    v0, async si da el tiempo).
  - `GET /health`, `GET /metrics` (formato Prometheus), `GET /model-info`.
- La documentación Swagger/OpenAPI es parte de la entrega: escribir buenas
  descripciones y ejemplos en los schemas pydantic.
- Loguear cada input/output de inferencia (anonimizado, sin persistir imágenes
  por defecto) para monitoreo de drift.

## Requerimientos electivos que implementamos (se exigen ≥3)

1. **Trazabilidad de ML:** MLflow (experimentos + registro de modelos) +
   manifests del dataset hasheados y versionados en git (DVC opcional si da el
   tiempo).
2. **Visualización:** frontend React (upload de clip/frame → predicción + barras
   de probabilidad + overlay de Grad-CAM + vista de timeline del partido).
3. **Optimización (≥2 sub-técnicas):** selección de features tabulares + data
   augmentation de imágenes + tuning de hiperparámetros con Optuna. Siempre
   medir el impacto (métricas de ML Y latencia) y registrarlo para el informe.
4. **Explicabilidad:** SHAP (tabular) + Grad-CAM (frames).

## Comandos

```bash
# Entorno (uv)
cd backend && uv sync

# Pipeline de datos (config-driven, idempotente)
uv run python -m src.data.download --config ../configs/dataset.yaml
uv run python -m src.data.build_dataset --config ../configs/dataset.yaml

# Entrenamiento (loguea a MLflow)
uv run python -m src.models.train --config ../configs/train.yaml

# Tests + lint (correr antes de dar por terminada cualquier tarea)
uv run pytest
uv run ruff check . && uv run ruff format --check .

# Stack completo
docker compose up --build
```

## Convenciones

- Python 3.12, type hints en todo, `ruff` para lint+formato.
- Código, identificadores, docstrings y mensajes de commit: **inglés**.
  Explicaciones al usuario, bitácora, informe y textos del frontend: **español
  rioplatense**.
- Config antes que constantes: todo lo tuneable vive en `configs/*.yaml`,
  cargado con pydantic-settings. Sin números mágicos en el código.
- Commits chicos y revisables por tarea; conventional commits (`feat:`, `fix:`,
  `data:`, `exp:`).
- Los notebooks nunca contienen lógica de la que dependa código productivo —
  importan desde `src/`.
- Tests obligatorios para: constructores de features (incluyendo un test de
  regresión de leakage que verifique que ningún `game_id` se cruza entre
  splits), schemas/endpoints de la API, y la corrección point-in-time de las
  features tabulares.

## Plan de fases y definición de terminado (DoD)

- **Fase 0 — Setup:** scaffolding del repo, tooling, esqueleto de docker, este
  archivo, `docs/consigna.md`.
  DoD: `docker compose up` sirve una API + frontend hello-world.
- **Fase 1 — Datos + EDA:** descarga del subconjunto, construcción del dataset
  por ventanas, notebook de EDA (distribución de clases, desbalance, eventos por
  liga/minuto).
  DoD: manifest parquet versionado + splits + notebook de EDA con hallazgos.
- **Fase 2 — Baseline end-to-end:** modelo v0 + MLflow + ambos endpoints de la
  API + docker compose completo. **Esto satisface el mínimo del obligatorio.**
  DoD: un clone fresco + `docker compose up` sirve predicciones; métricas
  logueadas.
- **Fase 3 — Iteración:** CNN v1, augmentation, Optuna, UX del frontend,
  Grad-CAM, SHAP, dashboards de Prometheus/Grafana.
- **Fase 4 — Informe y entrega:** informe en español con diagramas de
  arquitectura, discusión de trade-offs y alternativas (la bitácora es el
  insumo principal); declarar el uso de herramientas de IA (Claude Code) según
  las reglas del curso; zip ≤40MB; subir a Gestión antes del 15/07/2026 21:00.

Ante la duda sobre el alcance: recortar alcance, mantener el ciclo cerrado.
