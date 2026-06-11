---
description: Corre un experimento de entrenamiento, lo loguea a MLflow y actualiza la bitácora pedagógica
---

Vas a correr un experimento de entrenamiento. Descripción/hipótesis del usuario: $ARGUMENTS

Seguí estos pasos en orden:

1. **Plan breve (modo docente).** Antes de ejecutar, explicá en 3-5 líneas: qué
   cambia este experimento respecto del anterior, qué hipótesis prueba, y qué
   concepto del curso está en juego. Si la descripción del usuario es ambigua,
   preguntá antes de correr.
2. **Verificá el estado.** Asegurate de que el working tree esté limpio o que
   los cambios relevantes estén comiteados (el run debe ser reproducible desde
   un commit). Registrá el hash del commit.
3. **Ejecutá el entrenamiento** con la config correspondiente:
   `cd backend && uv run python -m src.models.train --config ../configs/train.yaml`
   (ajustá la config si el experimento lo requiere; los cambios de config van a
   git, nunca hardcodeados).
4. **Verificá el log en MLflow**: que el run tenga nombre descriptivo, params
   completos, seed, hash del commit y hash del manifest del dataset como tags, y
   las métricas por clase + PR-AUC (nunca solo accuracy — invariante 5).
5. **Compará contra el mejor run anterior** del experimento y resumí: ¿mejoró,
   empeoró, empató? ¿En qué métricas y a qué costo (latencia, complejidad)?
6. **Actualizá `report/bitacora.md`** con la entrada estándar: fecha, qué se
   probó, por qué, resultado vs. baseline, concepto del curso relacionado,
   requerimiento de la consigna que aporta (ver skill obligatorio-mlprod), y
   decisión (adoptar / descartar / iterar).
7. **Cierre didáctico**: 2-3 puntos de "qué deberías poder explicar de este
   experimento" en español.
