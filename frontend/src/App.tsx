import { useEffect, useState } from 'react'

type BackendStatus = 'loading' | 'ok' | 'error'

function App() {
  const [status, setStatus] = useState<BackendStatus>('loading')
  const [version, setVersion] = useState<string | null>(null)

  useEffect(() => {
    // Relative path: works in dev (Vite proxy) and in prod (nginx reverse-proxy) alike.
    fetch('/api/health')
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`)
        return res.json()
      })
      .then((data: { status: string; version: string }) => {
        setStatus(data.status === 'ok' ? 'ok' : 'error')
        setVersion(data.version)
      })
      .catch(() => setStatus('error'))
  }, [])

  const dotColor =
    status === 'ok'
      ? 'bg-emerald-500'
      : status === 'error'
        ? 'bg-red-500'
        : 'bg-amber-400 animate-pulse'

  return (
    <main className="flex min-h-screen flex-col items-center justify-center gap-6 bg-slate-950 text-slate-100">
      <div className="text-center">
        <h1 className="text-3xl font-bold tracking-tight">Clasificador de Eventos SoccerNet</h1>
        <p className="mt-2 text-slate-400">ML en Producción — Obligatorio · Fase 0</p>
      </div>

      <div className="flex items-center gap-3 rounded-xl border border-slate-800 bg-slate-900 px-5 py-3">
        <span className={`inline-block h-3 w-3 rounded-full ${dotColor}`} aria-hidden />
        <span className="text-sm">
          {status === 'loading' && 'Conectando con el backend…'}
          {status === 'ok' && `Backend conectado ✓${version ? ` (v${version})` : ''}`}
          {status === 'error' && 'Sin conexión con el backend'}
        </span>
      </div>
    </main>
  )
}

export default App
