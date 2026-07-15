import { useRef } from 'react'

interface Props {
  files: File[]
  loading: boolean
  onSelect: (files: File[]) => void
  onAnalyze: () => void
}

export function BatchForm({ files, loading, onSelect, onAnalyze }: Props) {
  const inputRef = useRef<HTMLInputElement>(null)

  return (
    <section className="card">
      <div className="card-t">Subir clips · batch</div>

      <button type="button" className="dropzone" onClick={() => inputRef.current?.click()}>
        <div className="ic">⬆</div>
        <div className="t">Elegí varios videos</div>
        <div className="d">mp4 / mkv / mov · se clasifican en una sola llamada</div>
      </button>
      <input
        ref={inputRef}
        type="file"
        accept="video/*"
        multiple
        style={{ display: 'none' }}
        onChange={(e) => onSelect(Array.from(e.target.files ?? []))}
      />

      {files.length > 0 && (
        <div className="batch-files">
          {files.map((f, i) => (
            <div className="vid-file" key={`${f.name}-${i}`}>
              <span>🎞</span> {f.name}
            </div>
          ))}
          <span className="x" role="button" onClick={() => onSelect([])}>
            limpiar ✕
          </span>
        </div>
      )}

      <div className="vid-note">
        Todos los clips se clasifican en una sola llamada a <b>/predict/clip/batch</b> (sin
        Grad-CAM). Se procesan en el momento y no se guardan.
      </div>

      <button className="btn" onClick={onAnalyze} disabled={files.length === 0 || loading}>
        {loading ? 'Analizando…' : `Analizar ${files.length || ''} videos ▸`}
      </button>
    </section>
  )
}
