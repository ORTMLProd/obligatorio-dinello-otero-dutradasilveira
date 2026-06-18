# CNN de clips — sub-proyecto 5 (Frontend) · Plan de implementación

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Agregar al frontend React un modo "Video": subir un clip → `POST /predict/clip` → mostrar la clase, las probabilidades y los overlays de Grad-CAM (frame grande + tira de 8 miniaturas), con un toggle Ventana | Video.

**Architecture:** Toggle de modo en `App.tsx`; en modo video, `VideoForm` (subir archivo) + `ClipPrediction` (veredicto + `GradcamViewer` + barras de probabilidad). `lib/api.ts` suma `predictClip(file)` (multipart). Reusa la pizarra táctica (clases CSS existentes) + clases nuevas para el video.

**Tech Stack:** React 19, Vite, TypeScript, Tailwind v4 + CSS propio.

**Spec:** `docs/superpowers/specs/2026-06-18-cnn-clips-frontend-design.md` · Mockup: `frontend/mockups/predict-video.html`.

**Branch:** `feat/fase-3.5-frontend` (ya creada; spec + mockup commiteados ahí).

---

## Nota sobre verificación

Cada task se verifica con typecheck + lint:
- `cd frontend && npx tsc -b` (debe pasar sin errores).
- `cd frontend && npm run lint` (eslint limpio).

Se agrega **Vitest** (liviano para Vite) para unit-testear la **lógica pura** del front
(ordenar probabilidades, confianza, clamp del Grad-CAM) — el rendering/integración se verifica
end-to-end (Task 6). Convenciones: código/identificadores en inglés; textos de UI y commits en
**español**; commits conventional, sin firma de Claude. NO commitear videos/frames.

---

## File Structure

- Modify: `frontend/src/lib/types.ts` — `GradcamFrame`, `ClipPredictResponse`.
- Modify: `frontend/src/lib/api.ts` — `predictClip(file)`.
- Create: `frontend/src/lib/format.ts` — helpers puros (`sortedByValueDesc`, `confidencePct`, `clampIndex`).
- Create: `frontend/src/lib/format.test.ts` — tests (Vitest).
- Modify: `frontend/package.json` — devDep `vitest` + script `test`.
- Modify: `frontend/src/index.css` — clases del modo video (`.modes`, `.dropzone`, `.vid-file`, `.vid-note`, `.gc-main`, `.gc-strip`, `.gc-thumb`).
- Create: `frontend/src/components/GradcamViewer.tsx`, `VideoForm.tsx`, `ClipPrediction.tsx`.
- Modify: `frontend/src/App.tsx` — toggle de modo + flujo video.

---

## Task 1: Tipos + cliente `predictClip`

**Files:**
- Modify: `frontend/src/lib/types.ts`
- Modify: `frontend/src/lib/api.ts`

- [ ] **Step 1: Add the types**

Append to `frontend/src/lib/types.ts`:

```ts
/** Un overlay de Grad-CAM: índice de frame + JPG en base64. */
export interface GradcamFrame {
  frame_index: number
  image_base64: string
}

/** Respuesta de POST /predict/clip: clase, probabilidades, versión y overlays Grad-CAM. */
export interface ClipPredictResponse {
  predicted_label: string
  probabilities: Record<string, number>
  model_version: string
  gradcam: GradcamFrame[]
}
```

- [ ] **Step 2: Add the API call**

In `frontend/src/lib/api.ts`, change the type import line to also import `ClipPredictResponse`:
```ts
import type { ClipPredictResponse, ModelInfo, PredictRequest, PredictResponse } from './types'
```
and append this function at the end:
```ts
export function predictClip(file: File): Promise<ClipPredictResponse> {
  const form = new FormData()
  form.append('video', file)
  // No Content-Type header: the browser sets the multipart boundary automatically.
  return fetch('/api/predict/clip', { method: 'POST', body: form }).then((r) =>
    asJson<ClipPredictResponse>(r),
  )
}
```

- [ ] **Step 3: Verify typecheck + lint**

Run: `cd frontend && npx tsc -b && npm run lint`
Expected: sin errores.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/lib/types.ts frontend/src/lib/api.ts
git commit -m "feat: tipos y cliente predictClip (POST /predict/clip)"
```

---

## Task 2: Vitest + helpers puros (`lib/format.ts`)

**Files:**
- Modify: `frontend/package.json`
- Create: `frontend/src/lib/format.ts`
- Create: `frontend/src/lib/format.test.ts`

- [ ] **Step 1: Add Vitest + a `test` script**

In `frontend/package.json`, add `"vitest": "^2.1.0"` to `devDependencies`, and add to `scripts`:
```json
    "test": "vitest run",
```
Then: `cd frontend && npm install`. (Vitest auto-detecta `vite.config.ts`; no hace falta config extra.)

- [ ] **Step 2: Write the failing tests**

Create `frontend/src/lib/format.test.ts`:

```ts
import { describe, expect, it } from 'vitest'

import { clampIndex, confidencePct, sortedByValueDesc } from './format'

describe('sortedByValueDesc', () => {
  it('orders entries by value descending', () => {
    expect(sortedByValueDesc({ a: 0.1, b: 0.7, c: 0.2 })).toEqual([
      ['b', 0.7],
      ['c', 0.2],
      ['a', 0.1],
    ])
  })
})

describe('confidencePct', () => {
  it('rounds the label probability to a percentage', () => {
    expect(confidencePct({ corner: 0.713, goal: 0.287 }, 'corner')).toBe(71)
  })
  it('returns 0 for a missing label', () => {
    expect(confidencePct({ corner: 0.7 }, 'goal')).toBe(0)
  })
})

describe('clampIndex', () => {
  it('clamps to [0, length-1] and handles empty', () => {
    expect(clampIndex(5, 3)).toBe(2)
    expect(clampIndex(-1, 3)).toBe(0)
    expect(clampIndex(1, 3)).toBe(1)
    expect(clampIndex(2, 0)).toBe(0)
  })
})
```

- [ ] **Step 3: Run, confirm FAIL**

Run: `cd frontend && npm test`
Expected: FAIL (no existe `./format`).

- [ ] **Step 4: Implement `frontend/src/lib/format.ts`**

```ts
// Pure UI helpers, unit-tested with Vitest. Kept out of components so they can be
// reasoned about and tested in isolation.

/** Entries of a probability map, sorted by value descending. */
export function sortedByValueDesc(probabilities: Record<string, number>): [string, number][] {
  return Object.entries(probabilities).sort((a, b) => b[1] - a[1])
}

/** The label's probability as a rounded percentage (0 if the label is absent). */
export function confidencePct(probabilities: Record<string, number>, label: string): number {
  return Math.round((probabilities[label] ?? 0) * 100)
}

/** Clamp ``index`` to ``[0, length - 1]`` (0 when ``length`` is 0). */
export function clampIndex(index: number, length: number): number {
  if (length <= 0) return 0
  return Math.min(Math.max(index, 0), length - 1)
}
```

- [ ] **Step 5: Run, confirm PASS + typecheck + lint**

Run: `cd frontend && npm test && npx tsc -b && npm run lint`
Expected: tests verdes, sin errores de tipos ni lint.

- [ ] **Step 6: Commit**

```bash
git add frontend/package.json frontend/package-lock.json frontend/src/lib/format.ts frontend/src/lib/format.test.ts
git commit -m "test: vitest + helpers puros de formato (probabilidades, confianza, clamp)"
```

---

## Task 3: CSS del modo video

**Files:**
- Modify: `frontend/src/index.css`

- [ ] **Step 1: Append the video-mode styles**

Append to `frontend/src/index.css` (las variables `--board`, `--amber`, etc. ya existen):

```css
/* ---------- modo video (Fase 3.5) ---------- */
.modes {
  display: flex;
  gap: 6px;
  margin-top: 22px;
  background: var(--board-2);
  border: 1px solid var(--line);
  border-radius: 11px;
  padding: 5px;
  width: fit-content;
}
.modes button {
  font-family: "Sora", sans-serif;
  font-weight: 600;
  font-size: 0.84rem;
  padding: 8px 20px;
  border: 0;
  border-radius: 8px;
  background: transparent;
  color: var(--chalk-dim);
  cursor: pointer;
}
.modes button.on {
  background: var(--amber);
  color: var(--board);
}

.dropzone {
  width: 100%;
  border: 1.5px dashed var(--line);
  border-radius: 12px;
  padding: 34px 18px;
  text-align: center;
  background: rgba(126, 224, 129, 0.03);
  cursor: pointer;
  transition: border-color 0.15s ease;
}
.dropzone:hover {
  border-color: var(--lime);
}
.dropzone .ic {
  font-size: 2rem;
  color: var(--lime);
}
.dropzone .t {
  font-weight: 600;
  margin-top: 10px;
}
.dropzone .d {
  font-size: 0.78rem;
  color: var(--chalk-dim);
  margin-top: 5px;
}
.vid-file {
  display: flex;
  align-items: center;
  gap: 10px;
  margin: 16px 0;
  padding: 10px 13px;
  border: 1px solid var(--line);
  border-radius: 9px;
  background: var(--panel);
  font-family: "DM Mono", monospace;
  font-size: 0.8rem;
  word-break: break-all;
}
.vid-file .x {
  margin-left: auto;
  color: var(--chalk-dim);
  cursor: pointer;
  flex: none;
}
.vid-note {
  font-size: 0.74rem;
  color: var(--chalk-dim);
  border-left: 3px solid var(--lime);
  padding: 6px 12px;
  margin: 14px 0;
  line-height: 1.5;
}

.gc-main {
  position: relative;
  width: 100%;
  aspect-ratio: 1 / 1;
  border-radius: 10px;
  overflow: hidden;
  border: 1px solid var(--line);
  background: var(--panel);
}
.gc-main img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}
.gc-main .tag {
  position: absolute;
  left: 10px;
  bottom: 10px;
  font-size: 0.7rem;
  color: var(--chalk);
  background: rgba(7, 11, 9, 0.6);
  padding: 3px 8px;
  border-radius: 6px;
}
.gc-strip {
  display: grid;
  grid-template-columns: repeat(8, 1fr);
  gap: 7px;
  margin-top: 12px;
}
.gc-thumb {
  aspect-ratio: 1 / 1;
  border-radius: 7px;
  border: 1px solid var(--line);
  overflow: hidden;
  cursor: pointer;
  padding: 0;
  background: var(--panel);
}
.gc-thumb img {
  width: 100%;
  height: 100%;
  object-fit: cover;
  display: block;
}
.gc-thumb.on {
  border: 2px solid var(--amber);
}
```

- [ ] **Step 2: Verify build picks up CSS**

Run: `cd frontend && npx tsc -b`
Expected: sin errores (el CSS no afecta el typecheck; este paso solo confirma que nada se rompió).

- [ ] **Step 3: Commit**

```bash
git add frontend/src/index.css
git commit -m "style: clases del modo video (toggle, dropzone, grad-cam)"
```

---

## Task 4: Componentes GradcamViewer, VideoForm, ClipPrediction

**Files:**
- Create: `frontend/src/components/GradcamViewer.tsx`
- Create: `frontend/src/components/VideoForm.tsx`
- Create: `frontend/src/components/ClipPrediction.tsx`

- [ ] **Step 1: Create `GradcamViewer.tsx`**

```tsx
import { useState } from 'react'

import { clampIndex } from '../lib/format'
import type { GradcamFrame } from '../lib/types'

const src = (f: GradcamFrame) => `data:image/jpeg;base64,${f.image_base64}`

export function GradcamViewer({ frames, label }: { frames: GradcamFrame[]; label: string }) {
  const [selected, setSelected] = useState(0)
  if (frames.length === 0) return null
  const index = clampIndex(selected, frames.length)
  const sel = frames[index]

  return (
    <div>
      <div className="gc-main">
        <img src={src(sel)} alt={`Grad-CAM frame ${sel.frame_index + 1}`} />
        <div className="tag mono">
          Grad-CAM · frame {index + 1}/{frames.length} · {label}
        </div>
      </div>
      <div className="gc-strip">
        {frames.map((frame, i) => (
          <button
            key={frame.frame_index}
            className={`gc-thumb${i === index ? ' on' : ''}`}
            onClick={() => setSelected(i)}
            aria-label={`frame ${frame.frame_index + 1}`}
          >
            <img src={src(frame)} alt="" />
          </button>
        ))}
      </div>
    </div>
  )
}
```

- [ ] **Step 2: Create `VideoForm.tsx`**

```tsx
import { useRef } from 'react'

interface Props {
  file: File | null
  loading: boolean
  onSelect: (file: File | null) => void
  onAnalyze: () => void
}

export function VideoForm({ file, loading, onSelect, onAnalyze }: Props) {
  const inputRef = useRef<HTMLInputElement>(null)

  return (
    <section className="card">
      <div className="card-t">Subir clip · entrada</div>

      <button type="button" className="dropzone" onClick={() => inputRef.current?.click()}>
        <div className="ic">⬆</div>
        <div className="t">Elegí un video</div>
        <div className="d">mp4 / mkv / mov · hasta ~30s</div>
      </button>
      <input
        ref={inputRef}
        type="file"
        accept="video/*"
        style={{ display: 'none' }}
        onChange={(e) => onSelect(e.target.files?.[0] ?? null)}
      />

      {file && (
        <div className="vid-file">
          <span>🎞</span> {file.name}
          <span className="x" role="button" onClick={() => onSelect(null)}>
            ✕
          </span>
        </div>
      )}

      <div className="vid-note">
        El video se procesa <b>en el momento</b> y no se guarda. El modelo extrae 8 frames y predice
        solo con la imagen (sin contexto del partido).
      </div>

      <button className="btn" onClick={onAnalyze} disabled={!file || loading}>
        {loading ? 'Analizando…' : 'Analizar video ▸'}
      </button>
    </section>
  )
}
```

- [ ] **Step 3: Create `ClipPrediction.tsx`**

```tsx
import { confidencePct, sortedByValueDesc } from '../lib/format'
import type { ClipPredictResponse } from '../lib/types'
import { GradcamViewer } from './GradcamViewer'

const DEFAULT_CLASSES = ['background', 'card', 'corner', 'goal', 'substitution']

export function ClipPrediction({
  result,
  loading,
}: {
  result: ClipPredictResponse | null
  loading: boolean
}) {
  const probEntries = result
    ? sortedByValueDesc(result.probabilities)
    : DEFAULT_CLASSES.map((c) => [c, 0] as [string, number])
  const conf = result ? confidencePct(result.probabilities, result.predicted_label) : 0

  return (
    <div className="stack">
      <section className="card">
        <div className="card-t">
          Lectura del clip
          {result && <span className="hint mono">{result.model_version}</span>}
        </div>

        <div className="verdict" style={{ marginBottom: 16 }}>
          <div className="pre">Clase predicha</div>
          {result ? (
            <>
              <div className="cls">{result.predicted_label}</div>
              <div className="conf">{conf}% confianza</div>
            </>
          ) : (
            <>
              <div className="cls idle">{loading ? 'Analizando…' : '—'}</div>
              <div className="waiting">
                <span className="pulse" />
                {loading ? 'corriendo inferencia' : 'subí un video para empezar'}
              </div>
            </>
          )}
        </div>

        {result ? (
          <GradcamViewer frames={result.gradcam} label={result.predicted_label} />
        ) : (
          <div className="idle-hint">
            El Grad-CAM mostrará, en 8 frames, qué región del clip miró el modelo.
          </div>
        )}
      </section>

      <section className="card">
        <div className="card-t">Probabilidades por clase</div>
        {probEntries.map(([name, p]) => {
          const win = result != null && name === result.predicted_label
          return (
            <div className={`pbar${win ? ' win' : ''}`} key={name}>
              <div className="top">
                <span>{name}</span>
                <span className="vl">{p.toFixed(2)}</span>
              </div>
              <div className="tr">
                <div className={`fl${win ? '' : ' dim'}`} style={{ width: `${p * 100}%` }} />
              </div>
            </div>
          )
        })}
      </section>
    </div>
  )
}
```

- [ ] **Step 4: Verify typecheck + lint + tests**

Run: `cd frontend && npx tsc -b && npm run lint && npm test`
Expected: sin errores; los tests de `format` siguen verdes.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/GradcamViewer.tsx frontend/src/components/VideoForm.tsx frontend/src/components/ClipPrediction.tsx
git commit -m "feat: componentes del modo video (GradcamViewer, VideoForm, ClipPrediction)"
```

---

## Task 5: Toggle de modo + flujo video en `App.tsx`

**Files:**
- Modify: `frontend/src/App.tsx`

- [ ] **Step 1: Rewrite `App.tsx` with the mode toggle and video flow**

Replace the whole `frontend/src/App.tsx` with:

```tsx
import { useEffect, useMemo, useState } from 'react'

import examplesData from './examples.json'
import { ClipPrediction } from './components/ClipPrediction'
import { PredictionPanel } from './components/PredictionPanel'
import { VideoForm } from './components/VideoForm'
import { WindowForm } from './components/WindowForm'
import { getModelInfo, predict, predictClip } from './lib/api'
import type {
  ClipPredictResponse,
  ExampleWindow,
  ModelInfo,
  PredictResponse,
  TabularFeatures,
} from './lib/types'

const examples = examplesData as ExampleWindow[]

function App() {
  const [info, setInfo] = useState<ModelInfo | null>(null)
  const [infoError, setInfoError] = useState(false)
  const [mode, setMode] = useState<'window' | 'video'>('window')

  // window mode
  const [selectedId, setSelectedId] = useState<string | null>(examples[0]?.id ?? null)
  const [form, setForm] = useState<TabularFeatures>(examples[0]?.tabular)
  const [embedding, setEmbedding] = useState<number[]>(examples[0]?.resnet_features ?? [])
  const [result, setResult] = useState<PredictResponse | null>(null)
  const [runId, setRunId] = useState(0)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // video mode
  const [videoFile, setVideoFile] = useState<File | null>(null)
  const [videoResult, setVideoResult] = useState<ClipPredictResponse | null>(null)
  const [videoLoading, setVideoLoading] = useState(false)
  const [videoError, setVideoError] = useState<string | null>(null)

  useEffect(() => {
    getModelInfo()
      .then(setInfo)
      .catch(() => setInfoError(true))
  }, [])

  const selectExample = (ex: ExampleWindow) => {
    setSelectedId(ex.id)
    setForm(ex.tabular)
    setEmbedding(ex.resnet_features)
    setResult(null)
    setError(null)
  }

  const onPredict = async () => {
    setLoading(true)
    setError(null)
    try {
      const res = await predict({ ...form, resnet_features: embedding })
      setResult(res)
      setRunId((n) => n + 1)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Error al predecir')
      setResult(null)
    } finally {
      setLoading(false)
    }
  }

  const onAnalyze = async () => {
    if (!videoFile) return
    setVideoLoading(true)
    setVideoError(null)
    try {
      setVideoResult(await predictClip(videoFile))
    } catch (e) {
      setVideoError(e instanceof Error ? e.message : 'Error al analizar el video')
      setVideoResult(null)
    } finally {
      setVideoLoading(false)
    }
  }

  const f1 = info?.test_macro_f1
  const status = useMemo(() => {
    if (infoError || (info && !info.model_loaded)) return { color: 'var(--red)', text: 'sin modelo' }
    if (info?.model_loaded) return { color: 'var(--lime)', text: 'en línea' }
    return { color: 'var(--amber)', text: 'conectando…' }
  }, [info, infoError])

  return (
    <div className="wrap">
      <header className="hdr">
        <div className="brand">
          <div className="ball" />
          <div>
            <h1>SoccerNet · Eventos</h1>
            <div className="s">Pizarra de clasificación</div>
          </div>
        </div>
        <div className="hdr-meta">
          <div>
            <div className="k">Modelo</div>
            {info?.version ?? '—'}
          </div>
          <div>
            <div className="k">Macro-F1</div>
            {f1 != null ? f1.toFixed(3) : '—'}
          </div>
          <div>
            <div className="k">Estado</div>
            <span style={{ color: status.color }}>● {status.text}</span>
          </div>
        </div>
      </header>

      <div className="modes">
        <button className={mode === 'window' ? 'on' : ''} onClick={() => setMode('window')}>
          Ventana
        </button>
        <button className={mode === 'video' ? 'on' : ''} onClick={() => setMode('video')}>
          Video
        </button>
      </div>

      {mode === 'window' ? (
        <>
          {error && (
            <div className="banner" style={{ marginTop: 18 }}>
              No se pudo predecir — {error}
            </div>
          )}
          <div className="grid">
            <WindowForm
              examples={examples}
              selectedId={selectedId}
              form={form}
              loading={loading}
              onSelectExample={selectExample}
              onChange={(patch) => setForm((prev) => ({ ...prev, ...patch }))}
              onPredict={onPredict}
            />
            <PredictionPanel
              result={result}
              loading={loading}
              classes={info?.classes ?? null}
              runId={runId}
            />
          </div>
        </>
      ) : (
        <>
          {videoError && (
            <div className="banner" style={{ marginTop: 18 }}>
              No se pudo analizar — {videoError}
            </div>
          )}
          <div className="grid">
            <VideoForm
              file={videoFile}
              loading={videoLoading}
              onSelect={(f) => {
                setVideoFile(f)
                setVideoResult(null)
                setVideoError(null)
              }}
              onAnalyze={onAnalyze}
            />
            <ClipPrediction result={videoResult} loading={videoLoading} />
          </div>
        </>
      )}

      <div className="foot">
        ML en Producción · Obligatorio · Fase 3 · modo Ventana (tabular) o Video (clip → Grad-CAM)
      </div>
    </div>
  )
}

export default App
```

- [ ] **Step 2: Verify typecheck + lint + build**

Run: `cd frontend && npx tsc -b && npm run lint && npm run build`
Expected: sin errores; `npm run build` produce `dist/` OK.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/App.tsx
git commit -m "feat: toggle Ventana|Video y flujo de analisis de video en App"
```

---

## Task 6: Verificación end-to-end (frontend + API real)

> Verifica el modo video contra la API real con el clip-model cargado. No es test automatizado.

- [ ] **Step 1: Levantar API (con clip-model) y frontend dev**

```bash
# API con el modelo de ventana y el clip (ajustar rutas absolutas):
cd backend && API_MODEL_DIR="$(cd ../models/v0 && pwd)" API_CLIP_MODEL_DIR="$(cd ../models/clips-v1 && pwd)" \
  uv run uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --log-level warning &
# Frontend dev (otro shell):
cd frontend && npm run dev -- --port 5173 --host 127.0.0.1 &
```
Esperar a que `curl -s http://127.0.0.1:5173/api/health` responda `{"status":"ok",...}`.

- [ ] **Step 2: Probar el modo video en el navegador**

Abrir `http://localhost:5173`, tocar el toggle **Video**, elegir un video corto (propio o sintético),
y tocar **Analizar video**. Verificar: aparece la clase predicha, las barras de probabilidad, el
frame grande de Grad-CAM y la tira de 8 miniaturas clickeables. El toggle **Ventana** sigue
mostrando la demo tabular.

- [ ] **Step 3: Bajar los procesos**

```bash
lsof -ti tcp:5173 tcp:8000 | xargs -r kill
```

---

## Self-Review (hecho)

- **Cobertura del spec:** tipos + `predictClip` (Task 1), Vitest + helpers puros (Task 2), CSS del
  modo video (Task 3), componentes GradcamViewer/VideoForm/ClipPrediction usando los helpers
  (Task 4), toggle + flujo en App (Task 5), verificación end-to-end (Task 6). Sin SHAP/cancha en
  video; `model_version` del clip en el panel; header con el modelo de ventana como estado general. ✓
- **Placeholders:** ninguno; código real (componentes completos). ✓
- **Consistencia de tipos:** `ClipPredictResponse{predicted_label,probabilities,model_version,gradcam:[GradcamFrame{frame_index,image_base64}]}`, `predictClip(file)`, props de `VideoForm`/`ClipPrediction`/`GradcamViewer`. Consistentes entre tasks. ✓
- **NDA:** no se commitean videos; tests/verificación con video propio/sintético. ✓
- **Out of scope:** timeline; tests unitarios de React.
