// Cliente de la API. Usa rutas relativas /api/... que el reverse-proxy (nginx en
// prod, proxy de Vite en dev) reenvía al backend — sin CORS, mismo código en ambos.

import type { ClipPredictResponse, Health } from './types'

async function asJson<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const detail = await res.text().catch(() => '')
    throw new Error(`HTTP ${res.status}${detail ? `: ${detail}` : ''}`)
  }
  return res.json() as Promise<T>
}

export function getHealth(): Promise<Health> {
  return fetch('/api/health').then((r) => asJson<Health>(r))
}

export function predictClip(file: File): Promise<ClipPredictResponse> {
  const form = new FormData()
  form.append('video', file)
  // No Content-Type header: the browser sets the multipart boundary automatically.
  return fetch('/api/predict/clip', { method: 'POST', body: form }).then((r) =>
    asJson<ClipPredictResponse>(r),
  )
}
