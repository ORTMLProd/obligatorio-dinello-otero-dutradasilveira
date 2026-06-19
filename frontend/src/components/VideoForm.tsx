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
