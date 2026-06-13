// Panel de predicción SIEMPRE presente (estructura estable, sin reflow).
// Tres estados: idle (en espera, cancha y barras neutras), cargando, y resultado
// (con reveal escalonado + barras que barren desde 0).
import type { PredictResponse } from '../lib/types'
import { Pitch } from './Pitch'

const CLASS_DESC: Record<string, string> = {
  goal: 'Gol detectado. La señal visual del frame y el contexto del marcador empujaron la decisión.',
  corner: 'Tiro de esquina detectado en zona de área.',
  card: 'Amonestación detectada (tarjeta).',
  substitution: 'Sustitución detectada cerca de la línea técnica.',
  background: 'Sin evento relevante: juego en desarrollo (background).',
}

const DEFAULT_CLASSES = ['background', 'card', 'corner', 'goal', 'substitution']

interface Props {
  result: PredictResponse | null
  loading: boolean
  classes: string[] | null
  runId: number
}

function shortName(name: string) {
  return name.length > 18 ? name.slice(0, 17) + '…' : name
}

export function PredictionPanel({ result, loading, classes, runId }: Props) {
  const classList = classes ?? DEFAULT_CLASSES
  const conf = result ? Math.round((result.probabilities[result.predicted_label] ?? 0) * 100) : 0

  const probEntries = result
    ? Object.entries(result.probabilities).sort((a, b) => b[1] - a[1])
    : classList.map((c) => [c, 0] as [string, number])

  const shap = result?.explanations
    ? Object.entries(result.explanations).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
    : []
  const maxAbs = shap.reduce((m, [, v]) => Math.max(m, Math.abs(v)), 0) || 1

  return (
    <div className="stack">
      {/* HERO: cancha + veredicto (siempre) */}
      <section className="card">
        <div className="card-t">
          Lectura de la jugada
          {result && <span className="hint mono">{result.model_version}</span>}
        </div>
        <div className="hero">
          <Pitch label={result?.predicted_label ?? null} />
          <div className="verdict">
            <div className="pre">Clase predicha</div>
            {result ? (
              <div key={runId} className="reveal">
                <div className="cls">{result.predicted_label}</div>
                <div className="conf">{conf}% confianza</div>
                <div className="desc">{CLASS_DESC[result.predicted_label] ?? ''}</div>
              </div>
            ) : (
              <>
                <div className="cls idle">{loading ? 'Analizando…' : '—'}</div>
                <div className="waiting">
                  <span className="pulse" />
                  {loading ? 'corriendo inferencia' : 'en espera de predicción'}
                </div>
              </>
            )}
          </div>
        </div>
      </section>

      {/* PROBABILIDADES (siempre; idle = 0) */}
      <section className="card">
        <div className="card-t">Probabilidades por clase</div>
        {probEntries.map(([name, p], i) => {
          const win = result != null && name === result.predicted_label
          return (
            <div className={`pbar${win ? ' win' : ''}`} key={name}>
              <div className="top">
                <span>{name}</span>
                <span className="vl">{p.toFixed(2)}</span>
              </div>
              <div className="tr">
                <div
                  className={`fl${win ? '' : ' dim'}`}
                  style={{ width: `${p * 100}%`, transitionDelay: `${i * 50}ms` }}
                />
              </div>
            </div>
          )
        })}
      </section>

      {/* SHAP (siempre; idle = pista) */}
      <section className="card">
        <div className="card-t">Explicación SHAP · aporte a la clase</div>
        {!result && (
          <div className="idle-hint">
            Predecí para ver cómo cada feature empuja la decisión: el contexto tabular y un
            bucket <span style={{ color: 'var(--lime)' }}>visual</span> que agrega las 512
            dimensiones del embedding.
          </div>
        )}
        {result && shap.length === 0 && (
          <div className="idle-hint">El modelo servido no expone SHAP (solo modelos de árbol).</div>
        )}
        {shap.map(([name, v], i) => {
          const pct = (Math.abs(v) / maxAbs) * 48
          const pos = v >= 0
          return (
            <div className="shap" key={`${runId}-${name}`}>
              <span className={`nm${name === 'visual_embedding' ? ' visual' : ''}`}>
                {shortName(name)}
              </span>
              <span className="st">
                <span className="ax" />
                <span
                  className={`sb ${pos ? 'pos' : 'neg'}`}
                  style={{ width: `${pct}%`, transitionDelay: `${i * 60}ms` }}
                />
              </span>
              <span className="sv">
                {pos ? '+' : '−'}
                {Math.abs(v).toFixed(2)}
              </span>
            </div>
          )
        })}
      </section>

      {/* GRAD-CAM (siempre) */}
      <div className="gradcam">
        <div className="b">Próximamente · v1</div>
        <div className="t">Overlay Grad-CAM sobre el frame</div>
        <div className="d">
          Se habilita con la CNN fine-tuneada (Fase 3.5). El espacio queda reservado.
        </div>
      </div>
    </div>
  )
}
