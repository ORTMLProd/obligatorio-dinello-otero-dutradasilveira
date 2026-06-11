# Clasificador de Eventos SoccerNet — ML en Producción (Obligatorio)

Sistema de ML end-to-end que clasifica ventanas de video de partidos de fútbol en
tipos de evento (`goal`, `card`, `substitution`, `corner`, `background`), combinando
**frames** extraídos de los videos de SoccerNet con **features tabulares** point-in-time
(minuto, mitad, diferencia de score acumulada, liga, etc.).

Curso: Machine Learning en Producción — Máster, Universidad ORT Uruguay.

> **Estado: Fase 0 (Setup).** El scaffolding del monorepo (backend FastAPI + frontend
> React + Docker Compose) se desarrolla en la rama `feat/scaffold-fase-0`. El plan de
> fases completo está en [CLAUDE.md](CLAUDE.md).

## Política de datos (NDA)

Los videos de SoccerNet están protegidos por copyright y se obtuvieron bajo un NDA con
KAUST. **Nunca** se versionan videos `.mkv`, frames/imágenes extraídas, ni la contraseña
del NDA. Las carpetas `data/` y `models/` están en `.gitignore`. La contraseña se lee de
la variable de entorno `SOCCERNET_PASSWORD` (en `.env`, no versionado). El repo contiene
solo **código, configs y manifests** que permiten regenerar el dataset a quien tenga su
propia contraseña del NDA. Detalle completo en [CLAUDE.md](CLAUDE.md).

## Cómo correr

> Se completa al finalizar el scaffolding de la Fase 0: `docker compose up --build`
> levantará la API y el frontend.
