import { confidencePct } from '../lib/format'
import type { ClipBatchResponse } from '../lib/types'

export function BatchResults({
  result,
  loading,
}: {
  result: ClipBatchResponse | null
  loading: boolean
}) {
  return (
    <section className="card">
      <div className="card-t">
        Resultados del batch
        {result && <span className="hint mono">{result.predictions.length} clips</span>}
      </div>

      {!result ? (
        <div className="idle-hint">
          {loading
            ? 'Clasificando los clips…'
            : 'Subí varios videos y analizá para ver la tabla de predicciones.'}
        </div>
      ) : (
        <div className="batch-scroll">
          <table className="batch-table">
            <thead>
              <tr>
                <th>Archivo</th>
                <th>Clase</th>
                <th>Confianza</th>
              </tr>
            </thead>
            <tbody>
              {result.predictions.map((p, i) => (
                <tr key={`${p.filename}-${i}`}>
                  <td className="mono fn">{p.filename ?? `clip ${i + 1}`}</td>
                  <td>
                    <span className="tag">{p.predicted_label}</span>
                  </td>
                  <td className="cf">{confidencePct(p.probabilities, p.predicted_label)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  )
}
