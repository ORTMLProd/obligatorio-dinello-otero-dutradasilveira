import { useEffect, useMemo, useState } from 'react'

import { ClipPrediction } from './components/ClipPrediction'
import { VideoForm } from './components/VideoForm'
import { getHealth, predictClip } from './lib/api'
import type { ClipPredictResponse } from './lib/types'

function App() {
  const [online, setOnline] = useState<boolean | null>(null)

  const [videoFile, setVideoFile] = useState<File | null>(null)
  const [videoResult, setVideoResult] = useState<ClipPredictResponse | null>(null)
  const [videoLoading, setVideoLoading] = useState(false)
  const [videoError, setVideoError] = useState<string | null>(null)

  useEffect(() => {
    getHealth()
      .then(() => setOnline(true))
      .catch(() => setOnline(false))
  }, [])

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

  const status = useMemo(() => {
    if (online === false) return { color: 'var(--red)', text: 'sin conexión' }
    if (online) return { color: 'var(--lime)', text: 'en línea' }
    return { color: 'var(--amber)', text: 'conectando…' }
  }, [online])

  return (
    <div className="wrap">
      <header className="hdr">
        <div className="brand">
          <div className="ball" />
          <div>
            <h1>SoccerNet · Eventos</h1>
            <div className="s">Clasificación de clips de video</div>
          </div>
        </div>
        <div className="hdr-meta">
          <div>
            <div className="k">Modelo</div>
            {videoResult?.model_version ?? '—'}
          </div>
          <div>
            <div className="k">Estado</div>
            <span style={{ color: status.color }}>● {status.text}</span>
          </div>
        </div>
      </header>

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

      <div className="foot">ML en Producción · Obligatorio · clip de video → clase + Grad-CAM</div>
    </div>
  )
}

export default App
