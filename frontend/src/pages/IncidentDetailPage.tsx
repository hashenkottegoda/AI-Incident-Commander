import { useCallback, useEffect, useRef, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import {
  ApiError,
  approveIncident,
  getIncident,
  getIncidentProgress,
  rejectIncident,
} from '../api/client'
import type {
  AuditDecisionStatus,
  AuditEventSummary,
  EvidenceItem,
  Hypothesis,
  IncidentDetail,
  IncidentStatus,
  InvestigationState,
  NodeProgressEventSummary,
  RiskClassification,
} from '../api/types'
import { formatDetectedAt, formatNodeName, SEVERITY_STYLES } from '../lib/incidentDisplay'

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

// `investigation.incident_status` values the graph checkpoint never leaves
// once reached (backend/models/incident.py's IncidentStatus) -- the signal
// LiveTraceSection uses to stop polling GET /{id}/progress. See that
// component's docstring for the full stop-condition reasoning.
const TERMINAL_INCIDENT_STATUSES: ReadonlySet<IncidentStatus> = new Set([
  'resolved',
  'manual_intervention_required',
])

const PROGRESS_POLL_INTERVAL_MS = 4000

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
  // LiveTraceSection's polling gate -- deliberately checks BOTH status
  // sources, not just `investigation.incident_status` the way `currentPhase`
  // above does. `POST /reject` is the reason why: it sets `incident.status`
  // (the DB column) to `manual_intervention_required` directly and
  // *never* resumes the paused graph thread (see backend/api/approvals.py's
  // "Approve vs. reject: only approve touches the graph" section) -- so a
  // rejected incident's checkpoint stays at `awaiting_approval` forever.
  // Gating polling on the checkpoint status alone would poll forever after
  // a rejection; checking the DB column too catches that case (and
  // `resolved`, which likewise can be set without every consumer having
  // re-read a fresh checkpoint).
  const isInvestigationTerminal =
    investigation !== null &&
    (TERMINAL_INCIDENT_STATUSES.has(investigation.incident_status) ||
      TERMINAL_INCIDENT_STATUSES.has(incident.status))

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

      <LiveTraceSection
        incidentId={incident.id}
        hasCheckpoint={investigation !== null}
        isTerminal={isInvestigationTerminal}
      />

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

type ProgressLoadState =
  | { phase: 'loading' }
  | { phase: 'error'; message: string }
  | { phase: 'loaded'; events: NodeProgressEventSummary[] }

/**
 * `GET /{incident_id}/progress` rendered as a live-updating timeline --
 * "how" the investigation got to its current state, which is why this sits
 * above `InvestigationSection` (the "what it found" section) rather than
 * inside or after it.
 *
 * This has its own fetch/poll cycle, independent of the one-shot
 * `GET /{incident_id}` this whole page loads once (see `IncidentDetailPageForId`)
 * -- per the progress endpoint's own docstring in backend/api/incidents.py,
 * the entire point of it being a separate lightweight route is that a
 * client can poll *it* repeatedly without re-fetching/re-serializing the
 * full `IncidentDetail` payload on every tick. The two cycles only meet at
 * one boundary: `hasCheckpoint`/`isTerminal`, two booleans derived from the
 * parent's existing (non-polling) `investigation` + `incident` state, which
 * this component reads to decide whether to keep polling at all -- see
 * below.
 *
 * Stop condition: poll only while a checkpoint exists (`hasCheckpoint`) AND
 * it's not yet terminal (`!isTerminal`). No checkpoint at all means no
 * graph run has ever started for this incident, so there is nothing that
 * could plausibly add a new progress row without an external trigger this
 * page has no button for -- polling would just be repeatedly confirming the
 * same empty list.
 *
 * `isTerminal` (computed in `LoadedIncident`) deliberately checks BOTH
 * `investigation.incident_status` (the checkpoint) AND `incident.status`
 * (the DB column) reaching `resolved`/`manual_intervention_required`, not
 * just the checkpoint -- `POST /reject` sets only the DB column and
 * *never* resumes the paused graph thread (backend/api/approvals.py), so a
 * rejected incident's checkpoint is stuck at `awaiting_approval` forever.
 * Checking the checkpoint alone would poll a rejected incident forever.
 *
 * Caveat this doesn't fully close: `isTerminal` is computed from the
 * parent's `investigation`/`incident` snapshot, which (like the rest of
 * this page) only refreshes on mount and after an approve/reject decision,
 * not continuously. A SAFE-only plan (no human approval needed at all)
 * reaches its own terminal state -- `incident.status = DIAGNOSED` via
 * `action_executor_node`, graph `END` -- without any approve/reject click
 * ever happening on this page to trigger a parent reload, and `DIAGNOSED`
 * isn't in `TERMINAL_INCIDENT_STATUSES` (it's *also* a legitimate
 * mid-investigation value, set by `root_cause_node` before
 * `response_planner` even runs, so a dashboard can't treat it as terminal
 * on its own). So this panel can keep polling a SAFE-only-resolved incident
 * past its actual completion until something else triggers a parent
 * reload or the user navigates away. A real fix needs a backend signal
 * that unambiguously means "this graph run is done, full stop" (e.g. a
 * dedicated terminal marker distinct from `DIAGNOSED`'s dual meaning) --
 * out of scope for this read-only dashboard step; flagged rather than
 * routed around.
 */
function LiveTraceSection({
  incidentId,
  hasCheckpoint,
  isTerminal,
}: {
  incidentId: number
  hasCheckpoint: boolean
  isTerminal: boolean
}) {
  const [state, setState] = useState<ProgressLoadState>({ phase: 'loading' })
  const isPolling = hasCheckpoint && !isTerminal

  useEffect(() => {
    // `cancelled` is a plain local, deliberately NOT a `useRef` shared
    // across effect re-invocations (an earlier version of this effect used
    // one, and it was a real bug: every time this effect re-runs -- e.g.
    // React StrictMode's dev-only double-invoke, or `isPolling` itself
    // flipping true -> false -- a SHARED ref gets reset back to `false` at
    // the top of the new invocation. That un-cancels any still-in-flight
    // `tick()` call left over from a PRIOR invocation whose `await` hadn't
    // resolved yet: it wakes up, sees the shared flag is `false` again,
    // concludes it's still valid, and schedules another `setTimeout` using
    // its own stale closed-over `isPolling` (which may still be `true`) --
    // a zombie poll chain nothing can ever cancel again, since every future
    // cleanup only ever flips the SAME already-`true`-then-reset flag.
    // Observed exactly this in manual testing: polling kept firing well
    // after a `POST /reject` flipped `isTerminal` true. A `let` local here
    // is scoped to THIS invocation only, so a prior invocation's closure
    // keeps seeing ITS OWN `cancelled = true` from ITS OWN cleanup,
    // regardless of how many newer invocations start afterward.
    if (!hasCheckpoint) return

    let cancelled = false
    let timer: ReturnType<typeof setTimeout> | undefined

    async function tick() {
      try {
        const events = await getIncidentProgress(incidentId)
        if (cancelled) return
        setState({ phase: 'loaded', events })
      } catch (fetchError: unknown) {
        if (cancelled) return
        // Log and keep going -- a failed poll tick shouldn't tear down the
        // rest of the page. If we already have rows on screen, leave them
        // up rather than replacing them with an error; only show the
        // inline "unavailable" note when we have nothing better to show.
        console.error(`Failed to fetch progress for incident ${incidentId}`, fetchError)
        setState((previous) =>
          previous.phase === 'loaded'
            ? previous
            : {
                phase: 'error',
                message:
                  fetchError instanceof ApiError
                    ? fetchError.message
                    : fetchError instanceof Error
                      ? fetchError.message
                      : String(fetchError),
              },
        )
      } finally {
        // Re-checked here (not just captured from the effect closure)
        // doesn't matter for correctness since `isPolling` can't change
        // mid-tick without this effect re-running first (it's a dependency
        // below) -- but scheduling the *next* tick only when still polling,
        // right here, is what makes the chain stop instead of a stray
        // `setInterval` ticking forever after the incident resolves.
        if (!cancelled && isPolling) {
          timer = setTimeout(() => void tick(), PROGRESS_POLL_INTERVAL_MS)
        }
      }
    }

    void tick()

    return () => {
      cancelled = true
      if (timer !== undefined) clearTimeout(timer)
    }
  }, [incidentId, isPolling, hasCheckpoint])

  return (
    <section className="rounded-lg border border-slate-800 bg-slate-900/40 p-6">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <h2 className="text-lg font-semibold text-slate-50">Live investigation trace</h2>
        {isPolling && (
          <span className="flex items-center gap-1.5 text-xs font-medium text-emerald-400">
            <span className="h-2 w-2 animate-pulse rounded-full bg-emerald-500" aria-hidden="true" />
            Live -- polling every {PROGRESS_POLL_INTERVAL_MS / 1000}s
          </span>
        )}
      </div>
      <p className="mt-1 text-sm text-slate-400">
        One row per graph-node invocation, oldest first -- how the investigation reached its current
        state.
      </p>

      {!hasCheckpoint && (
        <p className="mt-4 rounded border border-dashed border-slate-700 p-4 text-center text-sm text-slate-400">
          No checkpoint exists for this incident yet, so there's no trace to show -- see Investigation
          below.
        </p>
      )}

      {hasCheckpoint && state.phase === 'loading' && (
        <div className="mt-4 flex items-center gap-2 py-4 text-sm text-slate-400">
          <span className="h-2 w-2 animate-pulse rounded-full bg-slate-500" aria-hidden="true" />
          Loading trace...
        </div>
      )}

      {hasCheckpoint && state.phase === 'error' && (
        <p className="mt-4 rounded bg-red-950/50 p-3 text-sm text-red-300">
          Trace unavailable: {state.message}
        </p>
      )}

      {hasCheckpoint && state.phase === 'loaded' && (
        <ProgressTimeline events={state.events} isPolling={isPolling} />
      )}
    </section>
  )
}

/** Renders progress rows as a vertical stepper. The most recent row is
 * marked "current" only while `isPolling` is true (i.e. the checkpoint's
 * phase isn't terminal) -- there's no `completed_at` on these rows to say
 * otherwise (see `backend/models/node_progress.py`'s docstring), so "most
 * recent row, investigation not yet terminal" is the same "currently
 * running" inference the backend's own docstring describes. Once terminal
 * (or mid-poll-failure but already resolved), every row renders as
 * completed instead. */
function ProgressTimeline({
  events,
  isPolling,
}: {
  events: NodeProgressEventSummary[]
  isPolling: boolean
}) {
  if (events.length === 0) {
    return (
      <p className="mt-4 rounded border border-dashed border-slate-700 p-4 text-center text-sm text-slate-400">
        Checkpoint exists but no node has recorded progress yet.
      </p>
    )
  }

  const lastIndex = events.length - 1

  return (
    <ol className="mt-4 flex flex-col">
      {events.map((event, index) => {
        const isCurrent = isPolling && index === lastIndex
        return (
          <li key={`${event.node_name}-${event.started_at}-${index}`} className="flex gap-3">
            <div className="flex flex-col items-center">
              <span
                className={`mt-1 h-2.5 w-2.5 shrink-0 rounded-full ${
                  isCurrent ? 'animate-pulse bg-emerald-500' : 'bg-slate-600'
                }`}
                aria-hidden="true"
              />
              {index < lastIndex && <span className="w-px flex-1 bg-slate-800" aria-hidden="true" />}
            </div>
            <div className="pb-4">
              <div className="flex flex-wrap items-center gap-2">
                <span
                  className={`text-sm font-medium ${isCurrent ? 'text-emerald-300' : 'text-slate-200'}`}
                >
                  {formatNodeName(event.node_name)}
                </span>
                {isCurrent ? (
                  <span className="rounded border border-emerald-800 bg-emerald-950/60 px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-emerald-300">
                    Current
                  </span>
                ) : (
                  <span className="text-xs text-emerald-600" aria-hidden="true">
                    ✓
                  </span>
                )}
              </div>
              <p className="text-xs text-slate-500">{formatDetectedAt(event.started_at)}</p>
            </div>
          </li>
        )
      })}
    </ol>
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
