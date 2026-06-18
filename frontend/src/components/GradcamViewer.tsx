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
