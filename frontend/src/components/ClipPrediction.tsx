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
