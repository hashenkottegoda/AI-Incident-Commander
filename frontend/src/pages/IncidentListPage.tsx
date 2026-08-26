import { useEffect, useState } from 'react'
import { ApiError, listIncidents } from '../api/client'
import type { IncidentStatus, IncidentSummary } from '../api/types'

const PAGE_SIZE = 20

const STATUS_OPTIONS: IncidentStatus[] = [
  'detected',
  'triaging',
  'investigating',
  'diagnosed',
  'awaiting_approval',
  'executing',
  'verifying',
  'resolved',
  'manual_intervention_required',
]

const SEVERITY_STYLES: Record<IncidentSummary['severity'], string> = {
  P1: 'bg-red-950/60 text-red-300 border-red-800',
  P2: 'bg-orange-950/60 text-orange-300 border-orange-800',
  P3: 'bg-amber-950/60 text-amber-300 border-amber-800',
  P4: 'bg-slate-800/60 text-slate-300 border-slate-700',
}

const dateFormatter = new Intl.DateTimeFormat(undefined, {
  dateStyle: 'medium',
  timeStyle: 'short',
})

function formatDetectedAt(iso: string): string {
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return iso
  return dateFormatter.format(date)
}

/**
 * Landing page: `GET /api/incidents`, filterable by `?status=` and paged via
 * `limit`/`offset`. No row navigation yet -- `GET /{incident_id}` and the
 * detail view are a later step, so rows are visually inert.
 *
 * Status column caveat: `Incident.status` (the DB column this list reads)
 * only advances at action-execution/recovery/approval time -- the
 * triage/investigation/root-cause/response-planner nodes never write it, so
 * an incident that's actively being investigated still reads "detected"
 * here. See backend/api/incidents.py's `list_incidents` docstring. The
 * muted caption below says so instead of the table implying a live phase
 * this endpoint can't provide.
 */
export function IncidentListPage() {
  const [status, setStatus] = useState<IncidentStatus | ''>('')
  const [offset, setOffset] = useState(0)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [incidents, setIncidents] = useState<IncidentSummary[]>([])

  useEffect(() => {
    let cancelled = false

    listIncidents({
      status: status || undefined,
      limit: PAGE_SIZE,
      offset,
    })
      .then((result) => {
        if (cancelled) return
        setIncidents(result)
        setError(null)
        setLoading(false)
      })
      .catch((fetchError: unknown) => {
        if (cancelled) return
        setError(
          fetchError instanceof ApiError
            ? fetchError.message
            : fetchError instanceof Error
              ? fetchError.message
              : String(fetchError),
        )
        setLoading(false)
      })

    return () => {
      cancelled = true
    }
  }, [status, offset])

  function goToStatus(next: IncidentStatus | '') {
    setStatus(next)
    setOffset(0)
    setLoading(true)
  }

  function goToOffset(next: number) {
    setOffset(next)
    setLoading(true)
  }

  const canGoNext = !error && incidents.length === PAGE_SIZE

  return (
    <section className="rounded-lg border border-slate-800 bg-slate-900/40 p-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold text-slate-50">Incidents</h1>
          <p className="mt-1 text-sm text-slate-400">
            Most recently detected first.{' '}
            <span title="Reads the Incident.status DB column, which only advances at approval/execution/recovery time -- it can lag the graph's live investigation phase.">
              Status reflects the DB record, not necessarily the live investigation phase.
            </span>
          </p>
        </div>

        <label className="flex items-center gap-2 text-sm text-slate-300">
          Status
          <select
            value={status}
            onChange={(event) => goToStatus(event.target.value as IncidentStatus | '')}
            className="rounded border border-slate-700 bg-slate-900 px-2 py-1 text-slate-100 focus:border-slate-500 focus:outline-none"
          >
            <option value="">All statuses</option>
            {STATUS_OPTIONS.map((option) => (
              <option key={option} value={option}>
                {option}
              </option>
            ))}
          </select>
        </label>
      </div>

      <div className="mt-6">
        {loading && (
          <div className="flex items-center gap-2 py-8 text-slate-400">
            <span className="h-2.5 w-2.5 animate-pulse rounded-full bg-slate-500" aria-hidden="true" />
            Loading incidents...
          </div>
        )}

        {!loading && error && (
          <p className="rounded bg-red-950/50 p-3 text-sm text-red-300">
            {error}
            <br />
            Is the backend running? <code>uv run uvicorn backend.main:app --reload</code>
          </p>
        )}

        {!loading && !error && incidents.length === 0 && status !== '' && (
          <p className="rounded border border-dashed border-slate-700 p-6 text-center text-sm text-slate-400">
            No incidents match status <code className="rounded bg-slate-800 px-1.5 py-0.5">{status}</code>.
            {' '}Remember most incidents never sit in an in-flight status long enough to match here --
            see the note above.
          </p>
        )}

        {!loading && !error && incidents.length === 0 && status === '' && (
          <p className="rounded border border-dashed border-slate-700 p-6 text-center text-sm text-slate-400">
            No incidents yet -- inject one via{' '}
            <code className="rounded bg-slate-800 px-1.5 py-0.5">POST /api/simulation/failure</code>.
          </p>
        )}

        {!loading && !error && incidents.length > 0 && (
          <div className="overflow-x-auto">
            <table className="w-full min-w-max border-collapse text-left text-sm">
              <thead>
                <tr className="border-b border-slate-800 text-xs uppercase tracking-wide text-slate-500">
                  <th className="py-2 pr-4">ID</th>
                  <th className="py-2 pr-4">Status</th>
                  <th className="py-2 pr-4">Severity</th>
                  <th className="py-2 pr-4">Failure type</th>
                  <th className="py-2 pr-4">Service</th>
                  <th className="py-2 pr-4">Detected</th>
                </tr>
              </thead>
              <tbody>
                {incidents.map((incident) => (
                  <IncidentRow key={incident.id} incident={incident} />
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      <div className="mt-4 flex items-center justify-between text-sm text-slate-400">
        <span>
          Showing {incidents.length > 0 ? offset + 1 : 0}
          {'–'}
          {offset + incidents.length}
        </span>
        <div className="flex gap-2">
          <button
            type="button"
            onClick={() => goToOffset(Math.max(0, offset - PAGE_SIZE))}
            disabled={loading || offset === 0}
            className="rounded border border-slate-700 px-3 py-1 disabled:cursor-not-allowed disabled:opacity-40"
          >
            Previous
          </button>
          <button
            type="button"
            onClick={() => goToOffset(offset + PAGE_SIZE)}
            disabled={loading || !canGoNext}
            className="rounded border border-slate-700 px-3 py-1 disabled:cursor-not-allowed disabled:opacity-40"
          >
            Next
          </button>
        </div>
      </div>
    </section>
  )
}

function IncidentRow({ incident }: { incident: IncidentSummary }) {
  return (
    <tr
      className="border-b border-slate-900 text-slate-200"
      title="Detail view coming soon"
    >
      <td className="py-2 pr-4 font-mono text-slate-400">{incident.id}</td>
      <td className="py-2 pr-4">
        <span className="rounded border border-slate-700 bg-slate-800/60 px-2 py-0.5 text-xs">
          {incident.status}
        </span>
      </td>
      <td className="py-2 pr-4">
        <span
          className={`rounded border px-2 py-0.5 text-xs font-medium ${SEVERITY_STYLES[incident.severity]}`}
        >
          {incident.severity}
        </span>
      </td>
      <td className="py-2 pr-4">{incident.failure_type}</td>
      <td className="py-2 pr-4">{incident.service_name}</td>
      <td className="py-2 pr-4 text-slate-400">{formatDetectedAt(incident.detected_at)}</td>
    </tr>
  )
}
