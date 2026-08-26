import { useEffect, useState } from 'react'
import { API_BASE_URL, fetchHealth } from '../api/client'

type HealthState =
  | { phase: 'loading' }
  | { phase: 'healthy'; status: string }
  | { phase: 'unreachable'; message: string }

/**
 * Whole-stack smoke test: fetches `GET /health` on mount and renders the
 * result. Mirrors BUILD_PLAN.md Phase 0's own verify step ("GET /health
 * returns 200"), now exercised from the browser instead of curl -- proof
 * the Vite dev server, CORS, and the FastAPI process are all wired
 * together correctly before any real feature page is built on top.
 */
export function HealthCheckPage() {
  const [state, setState] = useState<HealthState>({ phase: 'loading' })

  useEffect(() => {
    let cancelled = false

    fetchHealth()
      .then((response) => {
        if (!cancelled) setState({ phase: 'healthy', status: response.status })
      })
      .catch((error: unknown) => {
        if (!cancelled) {
          setState({
            phase: 'unreachable',
            message: error instanceof Error ? error.message : String(error),
          })
        }
      })

    return () => {
      cancelled = true
    }
  }, [])

  return (
    <section className="rounded-lg border border-slate-800 bg-slate-900/40 p-6">
      <h1 className="text-xl font-semibold text-slate-50">Backend connectivity</h1>
      <p className="mt-1 text-sm text-slate-400">
        Checking <code className="rounded bg-slate-800 px-1.5 py-0.5">{API_BASE_URL}/health</code>
      </p>

      <div className="mt-4 flex items-center gap-2">
        <StatusDot state={state} />
        <span className="text-base">
          {state.phase === 'loading' && 'Checking backend...'}
          {state.phase === 'healthy' && `Backend: healthy (status="${state.status}")`}
          {state.phase === 'unreachable' && 'Backend: unreachable'}
        </span>
      </div>

      {state.phase === 'unreachable' && (
        <p className="mt-3 rounded bg-red-950/50 p-3 text-sm text-red-300">
          {state.message}
          <br />
          Is the backend running? <code>uv run uvicorn backend.main:app --reload</code>
        </p>
      )}
    </section>
  )
}

function StatusDot({ state }: { state: HealthState }) {
  const color =
    state.phase === 'healthy'
      ? 'bg-emerald-500'
      : state.phase === 'unreachable'
        ? 'bg-red-500'
        : 'bg-slate-500 animate-pulse'
  return <span className={`h-2.5 w-2.5 rounded-full ${color}`} aria-hidden="true" />
}
