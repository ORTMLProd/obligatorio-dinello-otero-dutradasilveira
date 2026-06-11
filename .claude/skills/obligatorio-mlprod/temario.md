# Temario del curso — Machine Learning en Producción

ESTADO: PARCIAL — construido a partir de los títulos de las diapositivas
compartidas por el usuario (el curso aún está en dictado). Los "conceptos clave"
de cada clase están inferidos del título, no del contenido de las diapositivas:
si una vinculación pedagógica depende de un detalle fino, aclararlo y/o
preguntarle al usuario qué se vio exactamente en esa clase.

## Clases dictadas (según diapositivas disponibles)

## Semana 1 — Intro / Scoping
- Ciclo de vida de un sistema de ML en producción.
- Scoping del proyecto: definir el problema, acotar alcance, baseline primero.
- Vínculo con el proyecto: la decisión de v0 acotado (5 clases, ~30-50 partidos)
  y de cerrar el ciclo end-to-end antes de optimizar sale de acá.

## Semana 2 — Definición y recolección de datos
- Definición del problema de ML y del target.
- Fuentes y recolección de datos.
- Vínculo: definición del target por ventanas {goal, card, substitution,
  corner, background} y la ingesta desde SoccerNet (`src/data/`).

## Semana 4 — Muestreo y repaso / Etiquetado
- Estrategias de muestreo; desbalance de clases.
- Etiquetado de datos: calidad y construcción de labels.
- Vínculo: submuestreo de `background` a ratio configurable, manejo explícito
  del desbalance (invariante 5), uso de las anotaciones de SoccerNet como
  labels y el flag de visibilidad como señal de calidad de etiqueta.

## Semana 5 — Feature engineering
- Construcción de features; pipelines de transformación.
- Vínculo: features tabulares point-in-time (invariante 2) y `src/features/`
  como única fuente de verdad (invariante 3, anti training-serving skew).

## Semana 6 — Modelos
- Selección y familias de modelos.
- Vínculo: roadmap v0 (XGBoost/LogReg sobre features) → v1 (CNN fine-tuneada
  con late fusion).

## Semana 7 — Entrenamiento / Escalabilidad
- Entrenamiento: splits correctos, validación, reproducibilidad.
- Escalabilidad del entrenamiento y de los datos.
- Vínculo: splits por `game_id` (invariante 1, anti-leakage), seeds y configs
  logueadas (invariante 6), dataset config-driven para escalar de 30 a más
  partidos sin tocar código.

## Semana 8 — Serving / Calidad
- Serving: predicción online vs batch, APIs, contratos.
- Calidad: testing de sistemas de ML, validación de datos.
- Vínculo: FastAPI `/predict` y `/predict/batch`, schemas pydantic estrictos
  (invariante 4), tests de leakage y de point-in-time correctness.

## Semana 11 — MLOps
- Ciclo MLOps: CI/CD, versionado, automatización, monitoreo.
- Vínculo: MLflow (trazabilidad), Docker Compose, manifests versionados,
  Prometheus/Grafana.

## Temas probables de las semanas restantes (curso en dictado — confirmar)

A confirmar con el usuario a medida que avance el curso; si aparece uno de
estos, pedirle el título de la clase y actualizar este archivo:
- Monitoreo en producción y detección de drift.
- Explicabilidad / interpretabilidad (SHAP, Grad-CAM).
- Optimización de modelos (tuning, quantization, pruning, distillation).
- Despliegue en la nube (AWS) / infraestructura.

Regla: al vincular una decisión con un tema aún no dictado, decirlo
explícitamente ("esto probablemente lo vean más adelante en el curso como X").
