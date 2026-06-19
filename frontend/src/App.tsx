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
