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
