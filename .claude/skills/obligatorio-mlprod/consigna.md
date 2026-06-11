# Consigna del obligatorio — Machine Learning en Producción

Resumen estructurado de la letra oficial (Universidad ORT Uruguay, Facultad de
Ingeniería). Fuente: PDF de la consigna. Ante cualquier ambigüedad, consultar el
PDF original en `report/consigna_original.pdf` o preguntar a los docentes.

## Condiciones generales

- Puntaje: 0 a 40 puntos.
- Entrega: 15/07/2026 hasta las 21:00 en gestion.ort.edu.uy.
- Formato: un único archivo zip o rar ≤ 40MB (documentos de texto en PDF, dentro
  del zip). También se debe entregar todo en el repo de GitHub Classroom.
- Equipos de hasta 3 personas del mismo dictado.
- Problemas técnicos con la entrega: escribir a adjuntos_ei@ort.edu.uy antes de
  las 20:00 del día de entrega.

## Objetivo

Construir un sistema de Machine Learning de principio a fin: sistema de
clasificación que combine datos no estructurados (imágenes) y/o datos tabulares
para un problema de clasificación binaria o multiclase. Incluye construcción del
dataset, selección de características, optimización del modelo y exposición de
una API para predicciones online y batch, con foco en prevenir los problemas de
los sistemas de ML llevados a producción.

## Requerimientos mínimos

1. **Dataset**: crear un dataset de datos no estructurados (imágenes) y/o
   tabulares asociados, de la fuente de preferencia. Incluir EDA.
2. **Representación del problema**: clasificación binaria o multiclase donde
   puedan usarse imágenes y datos tabulares. El target lo definen los
   estudiantes.
3. **Ambiente**: dependencias separadas para desarrollo y producción. Docker
   para el despliegue.
4. **Versionado de código**: Git; compartir el repositorio en GitHub como parte
   de la entrega.
5. **Desafíos generales**: prevenir data leakage y prevenir training-serving
   skew.
6. **API**: predicciones online y batch, con documentación de uso (si es
   FastAPI, aprovechar la documentación automática).
7. **Plataforma de despliegue**: se recomienda AWS vía AWS Academy; se aceptan
   otras plataformas si ya se dominan.

## Requerimientos electivos (implementar MÍNIMO 3; cuantos más, mejor)

- **Scraper de datos**: scraper de una o varias webs para obtener el dataset de
  entrenamiento.
- **Trazabilidad de ML**: versionar experimentos, modelos y datos.
- **Explicabilidad**: técnicas para explicar/interpretar los modelos o sus
  predicciones.
- **Visualización**: herramientas como Streamlit o Gradio (u otras) para
  interactuar con el modelo vía UI.
- **AutoML**: permitido incluso no-code. REGLA: si se usa AutoML, Visualización
  deja de ser electivo y pasa a ser requerimiento mínimo.
- **Optimización de modelos** (implementar al menos 2 de): selección de
  características para tabular; data augmentation para imágenes; ajuste de
  hiperparámetros / quantization / pruning / distillation.
- Si se implementan técnicas de optimización, se DEBE evaluar su impacto en el
  rendimiento del modelo y del sistema (métricas de ML y latencia).
- Otras técnicas avanzadas útiles para producción: incluirlas en el informe
  (dan puntos extra o compensatorios).

## Entrega

- Informe + código base. El informe explica cómo se resolvió cada desafío, con
  referencias al código y a las herramientas usadas; puede apoyarse en diagramas
  de arquitectura.
- Se espera discusión de ventajas/desventajas de cada solución y de alternativas
  o mejoras posibles.
- Aceptar la tarea grupal de GitHub Classroom (crea el repo automáticamente).
  Incluir TODO lo entregable en el repo + entrega formal en Gestión.

## Criterios de evaluación

- Claridad expositiva.
- Prolijidad y contenido.
- Análisis exploratorio y preparación de los datos.
- Profundidad adecuada y uso de técnicas vistas en clase.
- Resolución exitosa de los desafíos planteados.

## Notas adicionales de los docentes

- Se prioriza el desarrollo end-to-end por encima del rendimiento del modelo:
  comenzar con un baseline end-to-end y recién después incorporar mejoras.
- Se aceptan propuestas de problemas distintos al sugerido.
- Puntos extra o compensatorios por problemas adicionales o técnicas avanzadas.

## Reglas de uso de IA generativa (de la carátula)

- Seguir las pautas de los docentes sobre uso de IA en el curso.
- Citar las herramientas utilizadas y el contexto de uso (generación de ideas,
  redacción inicial, análisis de datos, generación de código, corrección de
  estilo, etc.).
- Todo contenido producido por IAG debe ser revisado y verificado; los errores
  son responsabilidad del estudiante.
- La IA no sustituye el razonamiento crítico ni la elaboración personal: el
  trabajo debe reflejar la comprensión, análisis y proceso cognitivo propios del
  estudiante. (De acá sale el "Modo docente" del CLAUDE.md y la bitácora.)
