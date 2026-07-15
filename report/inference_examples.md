# Ejemplos de inferencia (demo del frontend)

Set de clips de ejemplo para probar la inferencia en el frontend (modo Video) y documentar
resultados en el informe. **Todos los clips salen de partidos del split de `test`** (que el
modelo nunca vio en entrenamiento), incluyendo partidos **cross-liga** para mostrar
generalización a ligas no vistas.

## Política de datos (NDA)

Los clips de video **no se comitean** (política NDA de SoccerNet): se generan bajo
`data/interim/demo_clips_test/` (gitignored). Este documento versiona solo la **referencia
textual** (partido, minuto, predicción), no video ni frames. Para regenerar:

```bash
# 1) tener los 44 partidos descargados (configs/dataset.yaml, requiere SOCCERNET_PASSWORD)
# 2) reconstruir el dataset de clips (genera el manifest con el split de test)
uv run python -m src.data.build_clips --config ../configs/dataset.yaml
# 3) extraer los clips demo (script de sesión, en scratchpad)
```

## Resultados verificados (modelo `clips-v1-clips-aug-ft`, test macro-F1 0.757)

Verificados con el flujo de serving real (`serve_clip`: video → 8 frames → predicción):

| Clip (partido de test) | Liga | Clase real | Predicho | Confianza |
|---|---|---|---|---|
| Crystal Palace 1-2 Arsenal, 1T 17:46 | EPL | background | background | 1.00 ✓ |
| Crystal Palace 1-2 Arsenal, 1T 18:34 | EPL | card | card | 1.00 ✓ |
| Crystal Palace 1-2 Arsenal, 1T 09:28 | EPL | corner | corner | 0.99 ✓ |
| Everton 3-1 Chelsea, 2T 09:53 (gráfica "MIKEL↓ KENEDY↑") | EPL | substitution | substitution | 1.00 ✓ |
| Everton 3-1 Chelsea, 2T 24:08 (gráfica "PEDRO↓ FALCAO↑") | EPL | substitution | substitution | 1.00 ✓ |
| (Champions League), 1T 04:36 | UCL | corner | corner | 0.96 ✓ |
| (Serie A), 1T 03:13 | Serie A | corner | corner | 0.71 ✓ |
| Crystal Palace 1-2 Arsenal, 1T 07:06 | EPL | goal | corner | 0.49 ✗ |

**7/8 correctos, incluidos los 2 partidos cross-liga** (generalización a ligas no vistas).

## Nota sobre la señal visual de `substitution` (para el informe)

El **cartel físico del cuarto árbitro** (LED pitchside) casi nunca está en pantalla en el
instante anotado — se buscó entre 24 ventanas de `substitution` y no aparece claro en
ninguna. Lo que sí es un indicador fuerte y frecuente es la **gráfica de sustitución del
broadcast** (lower-third con nº de jugadores + flechas ↓↑ + "SUBSTITUTION"): los dos ejemplos
elegidos (Everton-Chelsea) la muestran y el modelo predice `substitution` con confianza 1.00.
Cuando esa gráfica no está, la clase es visualmente ambigua (jugador en el banco / repetición)
— esto explica el recall alto pero precision baja de `substitution`/`card`.

## Caso destacado: clip externo con el cartel del cuarto árbitro (Grad-CAM)

Se probó un clip **externo** aportado por el estudiante (`player_substitution.mov`, un cambio
de la Champions League — Real Madrid, ~4.6s, 1536×1154, mayor resolución que el train 224p) en
el que **sí se ve el cartel LED del cuarto árbitro** (números rojo=sale / verde=entra):

- **Predicción: `substitution`, confianza 0.91** — el modelo generaliza a un partido y una
  competición no vistos, y a una resolución distinta.
- **Grad-CAM:** el foco de activación cae **directamente sobre los números LED del cartel**
  (y sobre el jugador saliendo). El modelo aprendió una feature **interpretable y alineada con
  lo humano** para `substitution`, no un atajo espurio.

Es la mejor evidencia de explicabilidad del trabajo: cuando el cartel está presente, el modelo
lo usa. Refuerza la lectura del recall alto de `substitution`: la clase se resuelve bien cuando
hay un indicador visual claro (cartel o gráfica de broadcast) y es ambigua cuando no lo hay.

## Nota para el informe (limitación honesta)

El único error (`goal` → `corner`, confianza 0.49) ilustra un **desfase temporal
train/serving**: en entrenamiento la ventana son 8s centrados en el evento anotado; al servir
un clip subido, `frames_from_video` muestrea K frames a lo largo de **todo** el clip (acá 10s),
que puede no alinear con el instante del gol. Además `goal` es la clase más difícil (menor
soporte). La baja confianza (0.49) refleja incertidumbre, no un error confiado.
