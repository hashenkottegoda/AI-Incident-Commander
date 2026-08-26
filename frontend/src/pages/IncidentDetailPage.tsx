import { useCallback, useEffect, useRef, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { ApiError, approveIncident, getIncident, rejectIncident } from '../api/client'
import type {
  AuditDecisionStatus,
  AuditEventSummary,
  EvidenceItem,
  Hypothesis,
  IncidentDetail,
  IncidentStatus,
  InvestigationState,
  RiskClassification,
} from '../api/types'
import { formatDetectedAt, SEVERITY_STYLES } from '../lib/incidentDisplay'

type LoadState =
  | { phase: 'loading' }
  | { phase: 'not_found' }
  | { phase: 'error'; message: string }
  | { phase: 'loaded'; detail: IncidentDetail }

const CONFIDENCE_FORMATTER = new Intl.NumberFormat(undefined, {
  style: 'percent',
  maximumFractionDigits: 0,
})

function formatConfidence(value: number | null): string {
  if (value === null) return 'n/a'
  return CONFIDENCE_FORMATTER.format(value)
}

const DECISION_STYLES: Record<AuditDecisionStatus, string> = {
  pending_approval: 'bg-amber-950/60 text-amber-300 border-amber-800',
  approved: 'bg-emerald-950/60 text-emerald-300 border-emerald-800',
  rejected: 'bg-red-950/60 text-red-300 border-red-800',
  auto_executed: 'bg-slate-800/60 text-slate-300 border-slate-700',
  executed: 'bg-sky-950/60 text-sky-300 border-sky-800',
}

const RISK_STYLES: Record<RiskClassification, string> = {
  safe: 'bg-slate-800/60 text-slate-300 border-slate-700',
  high_impact: 'bg-orange-950/60 text-orange-300 border-orange-800',
}

/**
 * `/incidents/:id` -- `GET /{incident_id}`'s three sources
 * (`incident`/`investigation`/`audit_events`) rendered as their own
 * sections, matching `IncidentDetail`'s shape rather than flattening it.
 *
 * The audit-trail table is read-only except for `pending_approval` rows
 * classified `high_impact` -- those get Approve/Reject buttons wired to
 * `POST /approve` and `/reject` (see `AuditEventsSection`). Both endpoints
 * decide *every* currently-pending action for the incident in one call
 * (see `backend/api/approvals.py`'s docstring), so the decision UI is
 * incident-scoped, not per-row, even though it renders on each qualifying
 * row.
 *
 * `investigation` is `null` whenever no LangGraph checkpoint exists yet for
 * this incident (an incident injected but never run through `/investigate`
 * or `/investigate/graph`) -- see `IncidentDetail`'s docstring in
 * backend/api/incidents.py. That's a normal, expected state, not an error,
 * so it gets an honest "investigation not started yet" panel rather than a
 * blank or broken section.
 */
export function IncidentDetailPage() {
  const { id } = useParams<{ id: string }>()
  // Keyed on `id` so navigating between two detail pages (e.g. via a
  // future "related incident" link) remounts this subtree and starts a
  // fresh `{ phase: 'loading' }` state, instead of needing to reset state
  // synchronously inside the effect below.
  return <IncidentDetailPageForId key={id} id={id} />
}

function IncidentDetailPageForId({ id }: { id: string | undefined }) {
  const incidentId = id !== undefined ? Number(id) : NaN
  // Validity is derived at render time, not stored in `state` below -- an
  // invalid id (e.g. a stray non-numeric route segment) never needs a
  // network round trip, so it's not something the fetch effect decides.
  const validId = Number.isFinite(incidentId)
  const [state, setState] = useState<LoadState>({ phase: 'loading' })
  // Guards both the mount effect's fetch and any later manual `reload()`
  // call (e.g. after an approve/reject decision) against setting state
  // after unmount -- a single ref rather than a per-call local so a
  // manual reload triggered late in the component's life still respects
  // an unmount that happened after it started.
  const cancelledRef = useRef(false)

  // Shared by the initial mount fetch below and by the approve/reject flow
  // in AuditEventsSection (via the `onDecided` prop) -- a decision mutates
  // server state, so the honest way to reflect it is re-reading the same
  // `GET /{incident_id}` this page already trusts, not patching local
  // state with an optimistic guess. Returns the underlying promise (rather
  // than being fire-and-forget) so `submitDecision` below can await the
  // refetch actually landing before it clears its own submitting/confirming
  // state -- otherwise a just-decided row would briefly render its live,
  // clickable Approve/Reject buttons again for the duration of this round
  // trip, using the still-stale `event.decision_status` the parent hasn't
  // received the refetched `detail` for yet.
  const reload = useCallback(() => {
    if (!validId) return Promise.resolve()
    return getIncident(incidentId)
      .then((detail) => {
        if (cancelledRef.current) return
        setState({ phase: 'loaded', detail })
      })
      .catch((fetchError: unknown) => {
        if (cancelledRef.current) return
        if (fetchError instanceof ApiError && fetchError.status === 404) {
          setState({ phase: 'not_found' })
          return
        }
        setState({
          phase: 'error',
          message:
            fetchError instanceof ApiError
              ? fetchError.message
              : fetchError instanceof Error
                ? fetchError.message
                : String(fetchError),
        })
      })
  }, [incidentId, validId])

  useEffect(() => {
    cancelledRef.current = false
    reload()
    return () => {
      cancelledRef.current = true
    }
  }, [reload])

  return (
    <div className="flex flex-col gap-6">
      <Link to="/" className="text-sm text-slate-400 hover:text-slate-50">
        &larr; Back to incidents
      </Link>

      {!validId && (
        <section className="rounded-lg border border-slate-800 bg-slate-900/40 p-6">
          <h1 className="text-xl font-semibold text-slate-50">Couldn't load incident</h1>
          <p className="mt-3 rounded bg-red-950/50 p-3 text-sm text-red-300">
            "{id}" is not a valid incident id.
          </p>
        </section>
      )}

      {validId && state.phase === 'loading' && (
        <section className="rounded-lg border border-slate-800 bg-slate-900/40 p-6">
          <div className="flex items-center gap-2 py-8 text-slate-400">
            <span className="h-2.5 w-2.5 animate-pulse rounded-full bg-slate-500" aria-hidden="true" />
            Loading incident...
          </div>
        </section>
      )}

      {state.phase === 'not_found' && (
        <section className="rounded-lg border border-slate-800 bg-slate-900/40 p-6">
          <h1 className="text-xl font-semibold text-slate-50">Incident not found</h1>
          <p className="mt-2 text-sm text-slate-400">
            No incident with id <code className="rounded bg-slate-800 px-1.5 py-0.5">{id}</code> exists.
          </p>
        </section>
      )}

      {state.phase === 'error' && (
        <section className="rounded-lg border border-slate-800 bg-slate-900/40 p-6">
          <h1 className="text-xl font-semibold text-slate-50">Couldn't load incident</h1>
          <p className="mt-3 rounded bg-red-950/50 p-3 text-sm text-red-300">
            {state.message}
            <br />
            Is the backend running? <code>uv run uvicorn backend.main:app --reload</code>
          </p>
        </section>
      )}

      {state.phase === 'loaded' && <LoadedIncident detail={state.detail} onDecided={reload} />}
    </div>
  )
}

function LoadedIncident({
  detail,
  onDecided,
}: {
  detail: IncidentDetail
  onDecided: () => Promise<void>
}) {
  const { incident, investigation, audit_events: auditEvents } = detail

  // `investigation.incident_status` is the graph's own live-checkpoint
  // phase; it can be ahead of `incident.status` (the DB column, which only
  // advances at approval/execution/recovery time). Prefer it when a
  // checkpoint exists -- see InvestigationState's docstring in
  // backend/api/incidents.py -- and fall back to the DB status only when
  // there's no checkpoint at all.
  const currentPhase: IncidentStatus = investigation ? investigation.incident_status : incident.status

  return (
    <>
      <section className="rounded-lg border border-slate-800 bg-slate-900/40 p-6">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <h1 className="text-xl font-semibold text-slate-50">
              Incident <span className="font-mono text-slate-400">#{incident.id}</span>
            </h1>
            <p className="mt-1 text-sm text-slate-400">{incident.service_name}</p>
          </div>
          <span
            className={`rounded border px-2 py-0.5 text-xs font-medium ${SEVERITY_STYLES[incident.severity]}`}
          >
            {incident.severity}
          </span>
        </div>

        <dl className="mt-6 grid grid-cols-2 gap-x-6 gap-y-4 text-sm sm:grid-cols-3">
          <Field label="Current phase">
            <span className="rounded border border-slate-700 bg-slate-800/60 px-2 py-0.5 text-xs">
              {currentPhase}
            </span>
            <p className="mt-1 text-xs text-slate-500">
              {investigation ? (
                'From the graph’s live checkpoint (investigation.incident_status).'
              ) : (
                <>
                  From the <code className="rounded bg-slate-800 px-1 py-0.5">Incident.status</code> DB
                  column -- no investigation checkpoint exists yet, so this can lag the real phase once
                  one starts. See the caveat on the incident list.
                </>
              )}
            </p>
          </Field>
          <Field label="Failure type">{incident.failure_type}</Field>
          <Field label="Detected">{formatDetectedAt(incident.detected_at)}</Field>
          <Field label="Ground-truth root cause">
            <span
              className="rounded border border-purple-800 bg-purple-950/60 px-2 py-0.5 text-xs text-purple-300"
              title="Injected ground truth for evaluation (predicted-vs-actual), not the AI's diagnosis."
            >
              {incident.root_cause_category}
            </span>
            <p className="mt-1 text-xs text-slate-500">
              Injected for evaluation -- not what the AI diagnosed. See Investigation below for that.
            </p>
          </Field>
        </dl>
      </section>

      <InvestigationSection investigation={investigation} />

      <AuditEventsSection events={auditEvents} incidentId={incident.id} onDecided={onDecided} />
    </>
  )
}

function Field({
  label,
  children,
  className,
}: {
  label: string
  children: React.ReactNode
  className?: string
}) {
  return (
    <div className={className}>
      <dt className="text-xs uppercase tracking-wide text-slate-500">{label}</dt>
      <dd className="mt-1 text-slate-200">{children}</dd>
    </div>
  )
}

function InvestigationSection({ investigation }: { investigation: InvestigationState | null }) {
  if (investigation === null) {
    return (
      <section className="rounded-lg border border-slate-800 bg-slate-900/40 p-6">
        <h2 className="text-lg font-semibold text-slate-50">Investigation</h2>
        <p className="mt-4 rounded border border-dashed border-slate-700 p-6 text-center text-sm text-slate-400">
          Investigation not started yet -- no LangGraph checkpoint exists for this incident. Run{' '}
          <code className="rounded bg-slate-800 px-1.5 py-0.5">
            POST /api/incidents/{'{id}'}/investigate/graph
          </code>{' '}
          to start one.
        </p>
      </section>
    )
  }

  return (
    <section className="rounded-lg border border-slate-800 bg-slate-900/40 p-6">
      <h2 className="text-lg font-semibold text-slate-50">Investigation</h2>

      <div className="mt-4 grid gap-6 lg:grid-cols-2">
        <div>
          <h3 className="text-sm font-semibold text-slate-300">
            Evidence ({investigation.evidence.length})
          </h3>
          <EvidenceList evidence={investigation.evidence} />
        </div>

        <div>
          <h3 className="text-sm font-semibold text-slate-300">
            Hypotheses considered ({investigation.hypotheses.length})
          </h3>
          <HypothesisList hypotheses={investigation.hypotheses} />
        </div>
      </div>

      <div className="mt-6 rounded border border-emerald-900 bg-emerald-950/30 p-4">
        <h3 className="text-sm font-semibold text-emerald-300">AI diagnosis</h3>
        {investigation.root_cause === null ? (
          <p className="mt-2 text-sm text-slate-400">No root cause diagnosed yet.</p>
        ) : (
          <div className="mt-2 flex flex-wrap items-center gap-3">
            <span className="rounded border border-emerald-800 bg-emerald-950/60 px-2 py-0.5 text-xs font-medium text-emerald-300">
              {investigation.root_cause}
            </span>
            <span className="text-sm text-slate-300">
              Confidence: {formatConfidence(investigation.diagnostic_confidence)}
            </span>
          </div>
        )}
      </div>

      {investigation.alternative_hypotheses.length > 0 && (
        <div className="mt-6">
          <h3 className="text-sm font-semibold text-slate-300">
            Alternative hypotheses ({investigation.alternative_hypotheses.length})
          </h3>
          <p className="mt-1 text-xs text-slate-500">
            Considered but not selected as the winning root cause.
          </p>
          <HypothesisList hypotheses={investigation.alternative_hypotheses} />
        </div>
      )}

      <div className="mt-6">
        <h3 className="text-sm font-semibold text-slate-300">
          Recommended actions ({investigation.recommended_actions.length})
        </h3>
        {investigation.recommended_actions.length === 0 ? (
          <p className="mt-2 text-sm text-slate-400">None recommended.</p>
        ) : (
          <ul className="mt-2 flex flex-col gap-2">
            {investigation.recommended_actions.map((action, index) => (
              <li
                key={index}
                className="rounded border border-slate-800 bg-slate-900/60 p-3 text-sm text-slate-300"
              >
                <KeyValueList record={action} />
              </li>
            ))}
          </ul>
        )}
      </div>

      <dl className="mt-6 grid grid-cols-2 gap-x-6 gap-y-4 text-sm sm:grid-cols-3">
        <Field label="Approval decision">{investigation.approval_decision ?? 'n/a'}</Field>
        <Field label="Execution result">
          {investigation.execution_result_id === null
            ? 'n/a'
            : Array.isArray(investigation.execution_result_id)
              ? investigation.execution_result_id.join(', ')
              : investigation.execution_result_id}
        </Field>
        <Field label="Recovery result" className="col-span-2 sm:col-span-3">
          {investigation.recovery_result === null ? (
            'n/a'
          ) : (
            // `recovery_result.checked_metrics` nests two levels deep
            // (service:metric -> {baseline_mean, post_action_mean, ...} --
            // see backend.agents.recovery_check_node._compare_to_baseline),
            // and `KeyValueList` recurses to render that legibly rather
            // than as one flat JSON blob. That real width doesn't fit this
            // field's normal grid column, so this field spans the full
            // row (className above) and scrolls horizontally within its
            // own box (below) instead of overflowing the page.
            <div className="max-w-full overflow-x-auto">
              <KeyValueList record={investigation.recovery_result} />
            </div>
          )}
        </Field>
      </dl>
    </section>
  )
}

function EvidenceList({ evidence }: { evidence: EvidenceItem[] }) {
  if (evidence.length === 0) {
    return <p className="mt-2 text-sm text-slate-400">No evidence gathered yet.</p>
  }
  return (
    <ul className="mt-2 flex flex-col gap-2">
      {evidence.map((item, index) => (
        <li key={index} className="rounded border border-slate-800 bg-slate-900/60 p-3 text-sm">
          <p className="text-slate-200">{item.description}</p>
          <p className="mt-1 text-xs text-slate-500">
            Source: <span className="font-mono">{item.source_ref.tool}</span>
            {item.source_ref.record_id !== null && ` (record #${item.source_ref.record_id})`}
            {item.source_ref.query !== null && ` -- query: "${item.source_ref.query}"`}
          </p>
        </li>
      ))}
    </ul>
  )
}

function HypothesisList({ hypotheses }: { hypotheses: Hypothesis[] }) {
  if (hypotheses.length === 0) {
    return <p className="mt-2 text-sm text-slate-400">None recorded.</p>
  }
  return (
    <ul className="mt-2 flex flex-col gap-2">
      {hypotheses.map((hypothesis, index) => (
        <li key={index} className="rounded border border-slate-800 bg-slate-900/60 p-3 text-sm">
          <div className="flex flex-wrap items-center justify-between gap-2">
            <span className="rounded border border-slate-700 bg-slate-800/60 px-2 py-0.5 text-xs font-medium text-slate-200">
              {hypothesis.category}
            </span>
            <span className="text-xs text-slate-400">
              Confidence: {formatConfidence(hypothesis.confidence)}
            </span>
          </div>
          <p className="mt-2 text-slate-300">{hypothesis.rationale}</p>
        </li>
      ))}
    </ul>
  )
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

/**
 * Renders an arbitrary `Record<string, unknown>` -- used for
 * `recommended_actions` entries (flat scalars) and `recovery_result`
 * (which nests one level deeper: `checked_metrics` is a
 * `dict[str, dict[str, Any]]` per-metric comparison blob, see
 * `backend.agents.recovery_check_node._compare_to_baseline`). Recurses one
 * level into plain-object values so that blob renders as its own indented
 * key/value list instead of one unreadable inline `JSON.stringify` blob;
 * arrays and anything deeper than that still fall back to `JSON.stringify`
 * -- good enough for what this backend actually ever nests here.
 */
function KeyValueList({ record }: { record: Record<string, unknown> }) {
  const entries = Object.entries(record)
  if (entries.length === 0) return <span className="text-slate-500">(empty)</span>
  return (
    <dl className="grid grid-cols-[max-content_1fr] gap-x-3 gap-y-1">
      {entries.map(([key, value]) => (
        <div key={key} className="contents">
          <dt className="text-xs text-slate-500">{key}</dt>
          <dd className="break-all text-xs text-slate-300">
            {isPlainObject(value) ? (
              <div className="rounded border border-slate-800 bg-slate-950/60 p-2">
                <KeyValueList record={value} />
              </div>
            ) : typeof value === 'object' && value !== null ? (
              JSON.stringify(value)
            ) : (
              String(value)
            )}
          </dd>
        </div>
      ))}
    </dl>
  )
}

/** localStorage key remembering the last-used approver name across page
 * loads/incidents -- there's no login in this MVP (BUILD_PLAN.md's
 * explicit scope cut), so the approver identity is just a plain
 * user-typed string, not a fake "logged in as" concept. Remembering it
 * locally only saves retyping; it is never sent anywhere until the user
 * actually clicks Approve/Reject. */
const APPROVER_STORAGE_KEY = 'aic.approver'

type PendingAction = 'approve' | 'reject'

/** Incident-scoped decision UI state, not per-row -- `POST /approve` and
 * `/reject` each decide *every* row still `pending_approval` for the
 * incident in one call (backend/api/approvals.py), so all
 * "Awaiting approval" rows share one confirm/submit/error state rather
 * than tracking it per `AuditEvent.id`. */
type DecisionState =
  | { phase: 'idle' }
  | { phase: 'confirming'; action: PendingAction }
  | { phase: 'submitting'; action: PendingAction }
  | { phase: 'error'; action: PendingAction; message: string }

function AuditEventsSection({
  events,
  incidentId,
  onDecided,
}: {
  events: AuditEventSummary[]
  incidentId: number
  onDecided: () => Promise<void>
}) {
  const hasAwaitingApproval = events.some(
    (event) =>
      event.risk_classification === 'high_impact' && event.decision_status === 'pending_approval',
  )

  const [approver, setApprover] = useState<string>(() => {
    try {
      return localStorage.getItem(APPROVER_STORAGE_KEY) ?? ''
    } catch {
      return '' // localStorage unavailable (private browsing etc.) -- not fatal.
    }
  })
  const [decision, setDecision] = useState<DecisionState>({ phase: 'idle' })
  const approverValid = approver.trim().length > 0
  // Re-entrancy guard for `submitDecision`, separate from `decision` state:
  // a plain `useRef` boolean is set/read synchronously in the same tick, so
  // it's immune to React's render/commit timing in a way `decision.phase`
  // is not -- two click events dispatched back-to-back before React commits
  // a re-render would both close over the SAME pre-click `decision` value
  // (since neither has re-rendered yet), so checking `decision.phase` alone
  // cannot distinguish "the legitimate first call, now past the confirming
  // gate" from "a second, re-entrant call using the same stale closure."
  // This ref can, because it's mutated synchronously the instant the first
  // call passes the guard, before any `await` or state update.
  const submittingRef = useRef(false)

  function updateApprover(value: string) {
    setApprover(value)
    try {
      localStorage.setItem(APPROVER_STORAGE_KEY, value)
    } catch {
      // Not fatal -- just won't be remembered next time.
    }
  }

  async function submitDecision(action: PendingAction) {
    if (submittingRef.current) return
    submittingRef.current = true

    try {
      const trimmedApprover = approver.trim()
      if (trimmedApprover.length === 0) {
        setDecision({ phase: 'error', action, message: 'Enter an approver name first.' })
        return
      }
      setDecision({ phase: 'submitting', action })
      try {
        const call = action === 'approve' ? approveIncident : rejectIncident
        const response = await call(incidentId, { approver: trimmedApprover })
        // Await the refetch landing before clearing `submitting` -- otherwise
        // this row would briefly render its live, clickable Approve/Reject
        // buttons again (still reading the parent's now-stale
        // `event.decision_status` prop) for the duration of this round trip.
        await onDecided()
        if (response.decision === 'already_decided') {
          // Idempotent no-op path -- someone else (or an earlier stuck
          // request) already decided this incident's pending action(s)
          // before this call landed. Not an error in the HTTP sense, but
          // it's not what this click asked for either, so it's surfaced
          // rather than silently treated as success.
          setDecision({
            phase: 'error',
            action,
            message: `Already decided (by ${response.approver ?? 'someone else'}) before this request went through -- the page has been refreshed to show the current state.`,
          })
        } else {
          setDecision({ phase: 'idle' })
        }
      } catch (submitError: unknown) {
        setDecision({
          phase: 'error',
          action,
          message:
            submitError instanceof ApiError
              ? submitError.message
              : submitError instanceof Error
                ? submitError.message
                : String(submitError),
        })
      }
    } finally {
      // Reset on every exit path (empty-approver bail-out, success,
      // already_decided, and network/HTTP error alike) so a later decision
      // (e.g. after dismissing an error and retrying) isn't permanently
      // locked out by a stale `true`.
      submittingRef.current = false
    }
  }

  return (
    <section className="rounded-lg border border-slate-800 bg-slate-900/40 p-6">
      <h2 className="text-lg font-semibold text-slate-50">Audit trail</h2>
      <p className="mt-1 text-sm text-slate-400">
        Every recommended action, its risk classification, and its approval/execution outcome.
      </p>

      {hasAwaitingApproval && (
        <div className="mt-4 flex flex-wrap items-end gap-3 rounded border border-amber-800 bg-amber-950/20 p-3">
          <label className="flex flex-col gap-1 text-xs text-slate-300">
            Approver
            <input
              type="text"
              value={approver}
              onChange={(event) => updateApprover(event.target.value)}
              placeholder="your name or handle"
              className="w-56 rounded border border-slate-700 bg-slate-900 px-2 py-1 text-sm text-slate-100 focus:border-slate-500 focus:outline-none"
            />
          </label>
          <p className="max-w-md text-xs text-slate-500">
            No login in this MVP -- this name is recorded verbatim as the approver on the audit
            trail below. Remembered locally so you don't retype it next time.
          </p>
        </div>
      )}

      {decision.phase === 'error' && (
        <p className="mt-3 rounded bg-red-950/50 p-3 text-sm text-red-300">
          Couldn't {decision.action}: {decision.message}
        </p>
      )}

      {events.length === 0 ? (
        <p className="mt-4 rounded border border-dashed border-slate-700 p-6 text-center text-sm text-slate-400">
          No actions recommended yet.
        </p>
      ) : (
        <div className="mt-4 overflow-x-auto">
          <table className="w-full min-w-max border-collapse text-left text-sm">
            <thead>
              <tr className="border-b border-slate-800 text-xs uppercase tracking-wide text-slate-500">
                <th className="py-2 pr-4">Action</th>
                <th className="py-2 pr-4">Risk</th>
                <th className="py-2 pr-4">Decision</th>
                <th className="py-2 pr-4">Approver</th>
                <th className="py-2 pr-4">Outcome</th>
                <th className="py-2 pr-4">Recommended</th>
                <th className="py-2 pr-4">Decided</th>
                <th className="py-2 pr-4">Executed</th>
              </tr>
            </thead>
            <tbody>
              {events.map((event) => (
                <AuditEventRow
                  key={event.id}
                  event={event}
                  decision={decision}
                  approverValid={approverValid}
                  onRequestConfirm={(action) => setDecision({ phase: 'confirming', action })}
                  onCancelConfirm={() => setDecision({ phase: 'idle' })}
                  onConfirm={(action) => void submitDecision(action)}
                />
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  )
}

function AuditEventRow({
  event,
  decision,
  approverValid,
  onRequestConfirm,
  onCancelConfirm,
  onConfirm,
}: {
  event: AuditEventSummary
  decision: DecisionState
  approverValid: boolean
  onRequestConfirm: (action: PendingAction) => void
  onCancelConfirm: () => void
  onConfirm: (action: PendingAction) => void
}) {
  const awaitingApproval =
    event.risk_classification === 'high_impact' && event.decision_status === 'pending_approval'
  const isSubmitting = decision.phase === 'submitting'
  const isConfirming = decision.phase === 'confirming'

  return (
    <tr className="border-b border-slate-900 text-slate-200">
      <td className="py-2 pr-4">{event.action_type}</td>
      <td className="py-2 pr-4">
        <span
          className={`rounded border px-2 py-0.5 text-xs font-medium ${RISK_STYLES[event.risk_classification]}`}
        >
          {event.risk_classification}
        </span>
      </td>
      <td className="py-2 pr-4">
        <div className="flex flex-wrap items-center gap-2">
          <span
            className={`rounded border px-2 py-0.5 text-xs font-medium ${DECISION_STYLES[event.decision_status]}`}
          >
            {event.decision_status}
          </span>
          {awaitingApproval && (
            <span className="rounded border border-amber-700 bg-amber-950/80 px-2 py-0.5 text-xs font-semibold text-amber-200">
              Awaiting approval
            </span>
          )}
        </div>
        {awaitingApproval && (
          <div className="mt-2">
            {isConfirming ? (
              <div className="flex max-w-xs flex-wrap items-center gap-2">
                <span className="text-xs text-amber-200">
                  {decision.action === 'approve'
                    ? 'Approve? Runs simulated remediation (rollback/restart/scale against the simulation layer only, never anything real).'
                    : 'Reject? Sends the incident to manual intervention; the investigation will not resume.'}
                </span>
                <button
                  type="button"
                  onClick={() => onConfirm(decision.action)}
                  className="rounded border border-emerald-700 bg-emerald-950/60 px-2 py-1 text-xs font-medium text-emerald-300 hover:bg-emerald-900/60"
                >
                  Confirm {decision.action}
                </button>
                <button
                  type="button"
                  onClick={onCancelConfirm}
                  className="rounded border border-slate-700 px-2 py-1 text-xs text-slate-300 hover:bg-slate-800"
                >
                  Cancel
                </button>
              </div>
            ) : (
              <div className="flex flex-wrap items-center gap-2">
                <button
                  type="button"
                  disabled={isSubmitting || !approverValid}
                  title={!approverValid ? 'Enter an approver name above first.' : undefined}
                  onClick={() => onRequestConfirm('approve')}
                  className="rounded border border-emerald-700 bg-emerald-950/60 px-2 py-1 text-xs font-medium text-emerald-300 hover:bg-emerald-900/60 disabled:cursor-not-allowed disabled:opacity-40"
                >
                  {isSubmitting && decision.action === 'approve' ? 'Approving…' : 'Approve'}
                </button>
                <button
                  type="button"
                  disabled={isSubmitting || !approverValid}
                  title={!approverValid ? 'Enter an approver name above first.' : undefined}
                  onClick={() => onRequestConfirm('reject')}
                  className="rounded border border-red-800 bg-red-950/40 px-2 py-1 text-xs font-medium text-red-300 hover:bg-red-900/40 disabled:cursor-not-allowed disabled:opacity-40"
                >
                  {isSubmitting && decision.action === 'reject' ? 'Rejecting…' : 'Reject'}
                </button>
              </div>
            )}
          </div>
        )}
      </td>
      <td className="py-2 pr-4 text-slate-400">{event.approver ?? '—'}</td>
      <td className="py-2 pr-4 text-slate-400">{event.execution_outcome ?? '—'}</td>
      <td className="py-2 pr-4 text-slate-400">{formatDetectedAt(event.recommended_at)}</td>
      <td className="py-2 pr-4 text-slate-400">
        {event.decided_at ? formatDetectedAt(event.decided_at) : '—'}
      </td>
      <td className="py-2 pr-4 text-slate-400">
        {event.executed_at ? formatDetectedAt(event.executed_at) : '—'}
      </td>
    </tr>
  )
}
