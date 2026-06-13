// Cancha esquemática (NDA-safe, sin frame real). Siempre visible: en idle se muestra
// la cancha sin marcador; con predicción, el evento "cae" en su zona.
// En la Fase 3.5 este panel mostrará el frame local + overlay Grad-CAM.

// Posición del marcador en el viewBox 360×230, por tipo de evento.
const SPOTS: Record<string, { x: number; y: number }> = {
  goal: { x: 338, y: 115 },
  corner: { x: 346, y: 15 },
  card: { x: 180, y: 95 },
  substitution: { x: 180, y: 222 },
  background: { x: 180, y: 115 },
}

export function Pitch({ label }: { label: string | null }) {
  const spot = (label && SPOTS[label]) || null
  return (
    <svg
      className={`pitch${label ? '' : ' idle'}`}
      viewBox="0 0 360 230"
      role="img"
      aria-label="cancha de fútbol"
    >
      <g fill="none" stroke="#5fae93" strokeWidth={1.4} opacity={0.85}>
        <rect x={8} y={8} width={344} height={214} rx={3} />
        <line x1={180} y1={8} x2={180} y2={222} />
        <circle cx={180} cy={115} r={34} />
        <circle cx={180} cy={115} r={2.5} fill="#5fae93" />
        <rect x={8} y={62} width={52} height={106} />
        <rect x={300} y={62} width={52} height={106} />
        <rect x={8} y={92} width={20} height={46} />
        <rect x={332} y={92} width={20} height={46} />
      </g>
      {spot && (
        // `key` por etiqueta → remonta y reproduce la animación en cada predicción.
        <g key={label} transform={`translate(${spot.x} ${spot.y})`}>
          <animate attributeName="opacity" from="0" to="1" dur="0.35s" fill="freeze" />
          <animateTransform
            attributeName="transform"
            type="translate"
            additive="sum"
            from="0 -18"
            to="0 0"
            dur="0.5s"
            fill="freeze"
            calcMode="spline"
            keyTimes="0;1"
            keySplines="0.2 0.7 0.2 1"
          />
          <circle r={11} fill="#ffb13b" opacity={0.22}>
            <animate attributeName="r" values="9;13;9" dur="2.2s" repeatCount="indefinite" />
          </circle>
          <circle r={4.5} fill="#ffb13b" />
        </g>
      )}
    </svg>
  )
}
