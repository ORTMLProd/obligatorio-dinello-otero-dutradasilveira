# Diseño — Fase 3.5, sub-proyecto 5: Frontend (modo Video)

Fecha: 2026-06-18 · Estado: aprobado para planificar

## Contexto

Último sub-proyecto de la Fase 3.5. Agrega al frontend React (pizarra táctica, Fase 3.3) un
**modo "Video"**: subir un clip → `POST /predict/clip` → mostrar la clase predicha, las
probabilidades y los **overlays de Grad-CAM**. Con esto el ciclo "video en el flujo" queda
completo de punta a punta (UI → API → modelo → explicación visual).

## Decisiones (acordadas, mockup validado)

- **Toggle `Ventana | Video`** arriba: cambia entre la demo tabular existente (`/predict`) y el
  modo video nuevo (`/predict/clip`). Cada modo con su panel de entrada y de salida.
- **Grad-CAM:** un **frame grande** del overlay seleccionado + una **tira de 8 miniaturas**
  clickeables (cambian el frame grande).
- En modo video **no hay SHAP ni cancha** (el clip-model es visual-only); el panel de salida es
  veredicto + Grad-CAM + barras de probabilidad.
- El badge del header sigue mostrando el modelo de ventana (estado general de la API); la versión
  del clip-model se muestra en el panel de resultado del clip.

## Arquitectura y componentes

```
App.tsx  (estado de modo 'window' | 'video'; renderiza un flujo u otro)
  ├─ modo window (existente): WindowForm + PredictionPanel
  └─ modo video (nuevo):
       VideoForm.tsx         (dropzone/file picker + archivo + botón "Analizar video")
       ClipPrediction.tsx    (veredicto + GradcamViewer + barras de probabilidad; siempre montado)
         └─ GradcamViewer.tsx (frame grande seleccionado + tira de 8 miniaturas clickeables)
lib/api.ts    + predictClip(file) -> POST /predict/clip (multipart)
lib/types.ts  + GradcamFrame, ClipPredictResponse
index.css     + clases del modo video (toggle, dropzone, gc-main/strip/thumb)
```

### Tipos (`lib/types.ts`)
```ts
export interface GradcamFrame { frame_index: number; image_base64: string }
export interface ClipPredictResponse {
  predicted_label: string
  probabilities: Record<string, number>
  model_version: string
  gradcam: GradcamFrame[]
}
```

### Cliente (`lib/api.ts`)
- `predictClip(file: File): Promise<ClipPredictResponse>` — arma un `FormData` con `video` y hace
  `POST /api/predict/clip` (mismo reverse-proxy relativo, sin CORS). Reusa el manejo de error.

### `VideoForm.tsx`
- Input de archivo (`<input type="file" accept="video/*">`) con una **zona clickeable** (y, si es
  barato, drag&drop). Muestra el nombre del archivo elegido y un botón quitar. Nota: el video se
  procesa en el momento y no se guarda; el modelo usa solo la imagen. Botón **"Analizar video"**
  deshabilitado si no hay archivo o si está cargando.

### `ClipPrediction.tsx`
- Panel siempre montado (estados idle/cargando/resultado, como `PredictionPanel`). En idle:
  invita a subir un video. Con resultado: veredicto (clase + confianza), `GradcamViewer`, y las
  barras de probabilidad ordenadas. Muestra `model_version` del clip en el título.

### `GradcamViewer.tsx`
- Estado local `selected` (0..K-1). Frame grande = `gradcam[selected]` (imagen
  `data:image/jpeg;base64,...`). Tira de miniaturas: K imágenes; click cambia `selected`.

### `App.tsx`
- Nuevo estado `mode: 'window' | 'video'`, `videoResult`, `videoLoading`, `videoError`, `runId`.
- Toggle en el header (debajo) que setea `mode`. Render condicional del flujo correspondiente.
- En `onAnalyze` (video): `predictClip(file)` → set result; manejo de error/loading.

## Verificación
- `npm run build` (tsc + vite) y `eslint` limpios.
- End-to-end con la **API real** (clip-model cargado): subir un video → 200 con clase + 8 overlays
  que se renderizan; el toggle alterna ambos modos; la demo de ventana sigue funcionando.
- Se verifica con un **video sintético / propio** (NDA: no se muestran frames reales a terceros;
  la captura de validación se hace local).

## NDA / datos
- El video lo sube el usuario y se procesa en la API (en memoria, no se persiste — ya cubierto en
  el sub-proyecto 4). Los overlays se muestran en el navegador del usuario (su propio dato). Nada
  de videos/frames se commitea; los mockups usan heatmaps simulados.

## Out of scope
- Reproducción del video en sí / scrubbing temporal real (mostramos los 8 frames muestreados).
- Tests unitarios de React (el frontend no tiene infra de tests; se verifica por build + ejecución,
  igual que en la Fase 3.3).
- Timeline del partido (visión futura, fuera de la Fase 3.5).

## Riesgos
- **Tamaño de la respuesta:** 8 JPGs base64 (~0.5 MB). Aceptable; se renderizan directos.
- **Latencia en CPU (contenedor):** el Grad-CAM hace backward en CPU → algo lento; mostrar estado
  "analizando…". Aceptable para la demo.
