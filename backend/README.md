# Backend — SoccerNet event classifier

API FastAPI y, desde la Fase 1, el código de datos, features y modelos.

## Desarrollo local

```bash
uv sync                                  # crea el entorno con Python 3.12 (gestionado por uv)
uv run fastapi dev src/api/main.py       # API en http://localhost:8000 (con reload + /docs)
uv run pytest                            # tests
uv run ruff check . && uv run ruff format --check .
```

## Estructura

```
src/
├── api/         # app FastAPI, routers, schemas (contrato pydantic)
├── config.py    # configuración con pydantic-settings (sin números mágicos)
├── data/        # ingesta SoccerNet, extracción de frames, manifests (Fase 1)
├── features/    # ÚNICA fuente de preprocesamiento — la importan training y API (Fase 1+)
├── models/      # entrenamiento, evaluación, export (Fase 2/3)
└── monitoring/  # logging de inferencia, drift, métricas (Fase 3)
```

**Invariante clave:** todo el preprocesamiento vive en `src/features/` y lo importan
tanto el entrenamiento como la API. Los transformadores fiteados se serializan con el
modelo y la API los carga — nunca se re-fitean en serving (anti training-serving skew).
