// Panel de entrada: ejemplos pre-cargados + campos tabulares editables.
// El embedding ResNet (512-dim) viene del ejemplo elegido; el usuario edita lo tabular.
import type { ExampleWindow, TabularFeatures } from '../lib/types'

interface Props {
  examples: ExampleWindow[]
  selectedId: string | null
  form: TabularFeatures
  loading: boolean
  onSelectExample: (ex: ExampleWindow) => void
  onChange: (patch: Partial<TabularFeatures>) => void
  onPredict: () => void
}

const LEAGUES = ['england_epl', 'spain_laliga', 'germany_bundesliga', 'italy_serie_a', 'france_ligue1']

function Num({
  label,
  value,
  step,
  onChange,
}: {
  label: string
  value: number
  step?: number
  onChange: (v: number) => void
}) {
  return (
    <div className="field">
      <label>{label}</label>
      <input
        className="control"
        type="number"
        step={step ?? 1}
        value={value}
        onChange={(e) => onChange(e.target.value === '' ? 0 : Number(e.target.value))}
      />
    </div>
  )
}

export function WindowForm({
  examples,
  selectedId,
  form,
  loading,
  onSelectExample,
  onChange,
  onPredict,
}: Props) {
  const selected = examples.find((e) => e.id === selectedId)
  return (
    <section className="card">
      <div className="card-t">La ventana · entrada</div>

      <span className="field-label">Ejemplos del test</span>
      <div className="chips">
        {examples.map((ex) => (
          <button
            key={ex.id}
            className={`chip${ex.id === selectedId ? ' active' : ''}`}
            onClick={() => onSelectExample(ex)}
          >
            {ex.true_label}
          </button>
        ))}
      </div>

      {selected && (
        <div className="emb-note">
          ResNet-512 cargado · ejemplo <b>{selected.true_label}</b> · {form.league}
        </div>
      )}

      <div className="row2">
        <div className="field">
          <label>Mitad</label>
          <div className="seg">
            <button className={form.half === 1 ? 'on' : ''} onClick={() => onChange({ half: 1 })}>
              1ª
            </button>
            <button className={form.half === 2 ? 'on' : ''} onClick={() => onChange({ half: 2 })}>
              2ª
            </button>
          </div>
        </div>
        <Num label="Minuto (en la mitad)" value={form.minute} onChange={(v) => onChange({ minute: v })} />
      </div>

      <div className="field">
        <label>Liga</label>
        <select
          className="control"
          value={form.league}
          onChange={(e) => onChange({ league: e.target.value })}
        >
          {[form.league, ...LEAGUES.filter((l) => l !== form.league)].map((l) => (
            <option key={l} value={l}>
              {l}
            </option>
          ))}
        </select>
      </div>

      <div className="row2">
        <Num
          label="Dif. de score (point-in-time)"
          value={form.score_diff}
          onChange={(v) => onChange({ score_diff: v })}
        />
        <div className="field">
          <label>Equipo</label>
          <div className="seg">
            <button
              className={form.team_is_home === 1 ? 'on' : ''}
              onClick={() => onChange({ team_is_home: 1 })}
            >
              Local
            </button>
            <button
              className={form.team_is_home === 0 ? 'on' : ''}
              onClick={() => onChange({ team_is_home: 0 })}
            >
              Visita
            </button>
            <button
              className={form.team_is_home === -1 ? 'on' : ''}
              onClick={() => onChange({ team_is_home: -1 })}
            >
              N/A
            </button>
          </div>
        </div>
      </div>

      <div className="row2">
        <Num
          label="Eventos hasta t"
          value={form.events_so_far}
          onChange={(v) => onChange({ events_so_far: v })}
        />
        <Num
          label="Seg. desde último evento"
          value={form.secs_since_last_event}
          step={0.5}
          onChange={(v) => onChange({ secs_since_last_event: v })}
        />
      </div>

      <div className="field">
        <label>Visible en pantalla</label>
        <div className="seg">
          <button className={form.visible === 1 ? 'on' : ''} onClick={() => onChange({ visible: 1 })}>
            Sí
          </button>
          <button className={form.visible === 0 ? 'on' : ''} onClick={() => onChange({ visible: 0 })}>
            No
          </button>
        </div>
      </div>

      <button className="btn" onClick={onPredict} disabled={loading || !selected}>
        {loading ? 'Prediciendo…' : 'Predecir evento ▸'}
      </button>
    </section>
  )
}
