---
name: obligatorio-mlprod
description: >
  Contexto académico del obligatorio de "Machine Learning en Producción"
  (Máster, Universidad ORT Uruguay). Usar esta skill SIEMPRE que haya que:
  (a) decidir si una tarea o feature cubre un requerimiento mínimo o electivo
  de la consigna, (b) responder preguntas sobre la letra del obligatorio, los
  criterios de evaluación o la entrega, (c) escribir o actualizar la bitácora
  pedagógica o el informe final, (d) planificar fases o recortar alcance, o
  (e) vincular una decisión técnica con los temas vistos en el curso.
---

# Obligatorio ML en Producción — contexto y mapeo

## Cómo usar esta skill

1. La letra completa y estructurada de la consigna está en `consigna.md` (misma
   carpeta). Leerla antes de afirmar qué exige o no exige el obligatorio.
2. El temario del curso está en `temario.md`. Al explicar una decisión técnica,
   citar el tema del curso correspondiente. Si el archivo dice "PENDIENTE",
   pedirle al usuario las diapositivas o el índice de clases para completarlo.
3. Toda explicación al usuario sigue el formato del "Modo docente" definido en
   el CLAUDE.md raíz: concepto del curso + requerimiento de la consigna +
   alternativas descartadas.

## Mapeo requerimiento → componente del proyecto

| Requerimiento (consigna) | Tipo | Componente que lo cubre |
|---|---|---|
| Dataset propio + EDA | Mínimo | `src/data/` (descarga SoccerNet, extracción de frames, manifest) + `notebooks/eda.ipynb` |
| Problema de clasificación con imágenes y/o tabular, target definido por estudiantes | Mínimo | Clasificación multiclase por ventanas: {goal, card, substitution, corner, background}; frames + features tabulares point-in-time |
| Dependencias dev/prod + Docker | Mínimo | `pyproject.toml` (grupos) + Dockerfiles + `docker-compose.yml` |
| Git + repo GitHub Classroom | Mínimo | Repo del equipo, conventional commits |
| Prevenir data leakage | Mínimo (desafío) | Splits por `game_id` (invariante 1), features point-in-time (invariante 2), test de regresión de leakage |
| Prevenir training-serving skew | Mínimo (desafío) | `src/features/` como única fuente de preprocesamiento (invariante 3), transformadores serializados, schemas pydantic |
| API online + batch + documentación | Mínimo | FastAPI: `/predict`, `/predict/batch`, Swagger con ejemplos |
| Trazabilidad de ML (experimentos, modelos, datos) | Electivo 1 | MLflow + manifests hasheados |
| Visualización / UI | Electivo 2 | Frontend React (Vite) + Tailwind + shadcn |
| Optimización de modelos (≥2 sub-técnicas, midiendo impacto) | Electivo 3 | Feature selection tabular + data augmentation + Optuna; medir métricas Y latencia antes/después |
| Explicabilidad | Electivo 4 | SHAP (tabular) + Grad-CAM (frames) |
| Scraper de datos | Electivo NO elegido | — (posible extra: scraping de metadata de partidos) |
| AutoML | Electivo NO elegido | — (ojo: si se usara, Visualización pasa a ser mínimo) |

## Reglas de evaluación a tener siempre presentes

- Se prioriza el end-to-end por sobre el rendimiento del modelo: cerrar el ciclo
  antes de optimizar.
- Criterios: claridad expositiva, prolijidad, EDA y preparación de datos,
  profundidad y uso de técnicas vistas en clase, resolución de los desafíos.
- Cuantos más electivos, mejor; hay puntos extra por técnicas avanzadas
  adicionales (ej. monitoreo con Prometheus/Grafana, detección de drift).
- El informe debe discutir ventajas/desventajas de cada solución y alternativas
  posibles → por eso la bitácora registra alternativas descartadas.
- Uso de IA generativa: debe declararse en el informe qué herramientas se usaron
  (Claude Code, Claude) y en qué contexto (planificación, generación de código,
  redacción, etc.). El estudiante debe poder defender todo el contenido.

## Restricciones de entrega

- Fecha límite: 15/07/2026 21:00 en gestion.ort.edu.uy.
- Un único archivo zip o rar ≤ 40MB; documentos de texto en PDF dentro del zip.
- El repo de GitHub Classroom debe contener todo lo entregado.
- El zip NUNCA debe incluir videos, frames ni la contraseña del NDA (política de
  datos del CLAUDE.md).
