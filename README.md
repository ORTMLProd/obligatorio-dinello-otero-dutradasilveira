# Clasificador de Eventos SoccerNet — ML en Producción (Obligatorio)

Sistema de ML end-to-end que clasifica ventanas de video de partidos de fútbol en
tipos de evento (`goal`, `card`, `substitution`, `corner`, `background`), combinando
**frames** extraídos de los videos de SoccerNet con **features tabulares** point-in-time
(minuto, mitad, diferencia de score acumulada, liga, etc.).

Curso: Machine Learning en Producción — Máster, Universidad ORT Uruguay.

> **Estado: Fase 0 (Setup) — completa.** Esqueleto del monorepo (backend FastAPI +
> frontend React + Docker Compose) con un ciclo end-to-end "hello-world". Sin datos ni
> modelo todavía (eso llega en Fase 1+). El plan de fases completo está en
> [CLAUDE.md](CLAUDE.md).

## Política de datos (NDA)

Los videos de SoccerNet están protegidos por copyright y se obtuvieron bajo un NDA con
KAUST. **Nunca** se versionan videos `.mkv`, frames/imágenes extraídas, ni la contraseña
del NDA. Las carpetas `data/` y `models/` están en `.gitignore`. La contraseña se lee de
la variable de entorno `SOCCERNET_PASSWORD` (en `.env`, no versionado). El repo contiene
solo **código, configs y manifests** que permiten regenerar el dataset a quien tenga su
propia contraseña del NDA. Detalle completo en [CLAUDE.md](CLAUDE.md).

## Cómo correr

### Todo el stack con Docker (recomendado)

```bash
docker compose up --build
```

Luego abrir **http://localhost:8080** — el frontend muestra el estado del backend.
La API queda en http://localhost:8000 (con documentación Swagger en `/docs`).

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
backend/   API FastAPI + (Fase 1+) datos, features, modelos
frontend/  React (Vite) + Tailwind
configs/   YAML de configuración (pydantic-settings)
docs/      consigna y documentación
report/    bitácora pedagógica + (Fase 4) informe
```
