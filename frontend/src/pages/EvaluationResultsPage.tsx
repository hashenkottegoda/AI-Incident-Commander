import { useEffect, useState } from 'react'
import { ApiError, getEvaluationResults } from '../api/client'
import type {
  DiagnosticAggregate,
  EvaluationResultsResponse,
  OperationalAggregate,
} from '../api/types'

type LoadState =
  | { phase: 'loading' }
  | { phase: 'not_found' }
  | { phase: 'error'; message: string }
  | { phase: 'loaded'; results: EvaluationResultsResponse }

const ARCHITECTURES = ['A', 'B', 'C', 'D'] as const

const ARCH_LABELS: Record<(typeof ARCHITECTURES)[number], string> = {
  A: 'A -- context-stuffing',
  B: 'B -- +tools',
  C: 'C -- +RAG',
  D: 'D -- full graph',
}

const PCT_FORMATTER = new Intl.NumberFormat(undefined, {
  style: 'percent',
  maximumFractionDigits: 1,
})
const NUM_FORMATTER = new Intl.NumberFormat(undefined, { maximumFractionDigits: 2 })
const TOKEN_FORMATTER = new Intl.NumberFormat(undefined, { maximumFractionDigits: 0 })

function formatPct(value: number | null): string {
  if (value === null) return 'n/a'
  return PCT_FORMATTER.format(value)
}

function formatNum(value: number | null, formatter: Intl.NumberFormat = NUM_FORMATTER): string {
  if (value === null) return 'n/a'
  return formatter.format(value)
}

/**
 * `metadata.generated_at` is `run_experiments._write_results`'s own
 * `%Y%m%dT%H%M%SZ` UTC timestamp (e.g. "20260826T153045Z") -- NOT ISO-8601,
 * so unlike every other timestamp this dashboard renders (see
 * `formatDetectedAt` in lib/incidentDisplay.ts), `new Date(...)` cannot
 * parse it directly: it has no `-`/`:` separators. Parsed here by hand with
 * a dedicated regex instead of reusing `formatDetectedAt`. Falls back to
 * the raw string if it ever doesn't match the expected shape -- honest
 * degradation, not a crash or a silently wrong date.
 */
const GENERATED_AT_RE = /^(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})Z$/
const dateFormatter = new Intl.DateTimeFormat(undefined, { dateStyle: 'medium', timeStyle: 'short' })

function formatGeneratedAt(raw: string): string {
  const match = GENERATED_AT_RE.exec(raw)
  if (!match) return raw
  const [, year, month, day, hour, minute, second] = match
  const date = new Date(
    Date.UTC(Number(year), Number(month) - 1, Number(day), Number(hour), Number(minute), Number(second)),
  )
  if (Number.isNaN(date.getTime())) return raw
  return dateFormatter.format(date)
}

/**
 * `/evaluation` -- `GET /api/evaluation/results`, Phase 7's A/B/C/D
 * diagnostic comparison table plus D's separate operational table (see
 * `backend/api/evaluation.py`'s module docstring for the exact contract).
 *
 * 404 ("no results found") is the expected, unremarkable state before
 * `run_experiments` has ever been invoked -- rendered as an honest empty
 * state pointing at the CLI command, not as an error. A run's
 * `operational_aggregate` is `null` whenever that run used
 * `--skip-operational`; that section renders its own "skipped" note rather
 * than an empty/broken table.
 */
export function EvaluationResultsPage() {
  const [state, setState] = useState<LoadState>({ phase: 'loading' })

  useEffect(() => {
    let cancelled = false

    getEvaluationResults()
      .then((results) => {
        if (cancelled) return
        setState({ phase: 'loaded', results })
      })
      .catch((fetchError: unknown) => {
        if (cancelled) return
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

    return () => {
      cancelled = true
    }
  }, [])

  return (
    <div className="flex flex-col gap-6">
      {state.phase === 'loading' && (
        <section className="rounded-lg border border-slate-800 bg-slate-900/40 p-6">
          <div className="flex items-center gap-2 py-8 text-slate-400">
            <span className="h-2.5 w-2.5 animate-pulse rounded-full bg-slate-500" aria-hidden="true" />
            Loading evaluation results...
          </div>
        </section>
      )}

      {state.phase === 'not_found' && (
        <section className="rounded-lg border border-slate-800 bg-slate-900/40 p-6">
          <h1 className="text-xl font-semibold text-slate-50">No evaluation results yet</h1>
          <p className="mt-3 rounded border border-dashed border-slate-700 p-6 text-center text-sm text-slate-400">
            No <code className="rounded bg-slate-800 px-1.5 py-0.5">run_experiments</code> output exists
            yet under <code className="rounded bg-slate-800 px-1.5 py-0.5">evaluation/results/</code>.
            Generate a run with:
            <br />
            <code className="mt-3 inline-block rounded bg-slate-800 px-2 py-1 text-left">
              uv run python -m backend.evaluation.run_experiments --count 5 --seed 42
            </code>
            <br />
            <span className="mt-2 inline-block text-xs text-slate-500">
              This drives real Claude API calls for every incident across all four architectures --
              expect real wall-clock time and real spend, per BUILD_PLAN.md's own cost note.
            </span>
          </p>
        </section>
      )}

      {state.phase === 'error' && (
        <section className="rounded-lg border border-slate-800 bg-slate-900/40 p-6">
          <h1 className="text-xl font-semibold text-slate-50">Couldn't load evaluation results</h1>
          <p className="mt-3 rounded bg-red-950/50 p-3 text-sm text-red-300">
            {state.message}
            <br />
            Is the backend running? <code>uv run uvicorn backend.main:app --reload</code>
          </p>
        </section>
      )}

      {state.phase === 'loaded' && <LoadedResults results={state.results} />}
    </div>
  )
}

function LoadedResults({ results }: { results: EvaluationResultsResponse }) {
  const { metadata, diagnostic_aggregate: diagnosticAggregate, operational_aggregate: operationalAggregate } =
    results

  return (
    <>
      <section className="rounded-lg border border-slate-800 bg-slate-900/40 p-6">
        <h1 className="text-xl font-semibold text-slate-50">Evaluation results</h1>
        <p className="mt-1 text-sm text-slate-400">
          Most recently generated experiment run (by results-file mtime -- see{' '}
          <code className="rounded bg-slate-800 px-1 py-0.5">GET /api/evaluation/results</code>'s
          docstring).
        </p>

        <dl className="mt-6 grid grid-cols-2 gap-x-6 gap-y-4 text-sm sm:grid-cols-4">
          <Field label="Seed">{metadata.seed}</Field>
          <Field label="Incident count">{metadata.count}</Field>
          <Field label="Generated">{formatGeneratedAt(metadata.generated_at)}</Field>
          <Field label="Operational run">{metadata.skip_operational ? 'skipped' : 'included'}</Field>
        </dl>
      </section>

      <DiagnosticTable aggregate={diagnosticAggregate} />

      <OperationalSection aggregate={operationalAggregate} skipped={metadata.skip_operational} />
    </>
  )
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <dt className="text-xs uppercase tracking-wide text-slate-500">{label}</dt>
      <dd className="mt-1 text-slate-200">{children}</dd>
    </div>
  )
}

function DiagnosticTable({ aggregate }: { aggregate: Record<string, DiagnosticAggregate> }) {
  const architectures = ARCHITECTURES.filter((arch) => aggregate[arch] !== undefined)

  return (
    <section className="rounded-lg border border-slate-800 bg-slate-900/40 p-6">
      <h2 className="text-lg font-semibold text-slate-50">Diagnostic comparison (A/B/C/D)</h2>
      <p className="mt-1 text-sm text-slate-400">
        A: context-stuffing baseline, B: +tools, C: +RAG, D: full LangGraph. Means are computed over
        successfully-scored incidents only ("ok") -- an architecture's errored cells are excluded
        entirely, never averaged in as a fabricated zero.
      </p>

      {architectures.length === 0 ? (
        <p className="mt-4 rounded border border-dashed border-slate-700 p-6 text-center text-sm text-slate-400">
          This run's results contain no A/B/C/D rows.
        </p>
      ) : (
        <div className="mt-4 overflow-x-auto">
          <table className="w-full min-w-max border-collapse text-left text-sm">
            <thead>
              <tr className="border-b border-slate-800 text-xs uppercase tracking-wide text-slate-500">
                <th className="py-2 pr-4">Architecture</th>
                <th className="py-2 pr-4">Root-cause accuracy</th>
                <th className="py-2 pr-4">Evidence precision</th>
                <th className="py-2 pr-4">Hallucination rate</th>
                <th className="py-2 pr-4">Tool calls</th>
                <th className="py-2 pr-4">Evidence / call</th>
                <th className="py-2 pr-4">Latency (s)</th>
                <th className="py-2 pr-4">Total tokens</th>
                <th className="py-2 pr-4">Scored (ok / errors)</th>
              </tr>
            </thead>
            <tbody>
              {architectures.map((arch) => {
                const row = aggregate[arch]
                return (
                  <tr key={arch} className="border-b border-slate-900 text-slate-200">
                    <td className="py-2 pr-4 font-medium">{ARCH_LABELS[arch]}</td>
                    <td className="py-2 pr-4">{formatPct(row.root_cause_accuracy_rate)}</td>
                    <td className="py-2 pr-4">{formatPct(row.mean_evidence_precision)}</td>
                    <td className="py-2 pr-4">{formatPct(row.mean_hallucination_rate)}</td>
                    <td className="py-2 pr-4">{formatNum(row.mean_tool_call_count)}</td>
                    <td className="py-2 pr-4">{formatNum(row.mean_evidence_per_tool_call)}</td>
                    <td className="py-2 pr-4">{formatNum(row.mean_latency_seconds)}</td>
                    <td className="py-2 pr-4">{formatNum(row.mean_total_tokens, TOKEN_FORMATTER)}</td>
                    <td className="py-2 pr-4 text-slate-400">
                      {row.n_ok} / {row.n_errors}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </section>
  )
}

function OperationalSection({
  aggregate,
  skipped,
}: {
  aggregate: OperationalAggregate | null
  skipped: boolean
}) {
  return (
    <section className="rounded-lg border border-slate-800 bg-slate-900/40 p-6">
      <h2 className="text-lg font-semibold text-slate-50">Operational evaluation (D only)</h2>
      <p className="mt-1 text-sm text-slate-400">
        D's full closed loop -- Response Planner &rarr; Risk Classifier &rarr; HITL (auto-approved for
        this eval harness) &rarr; Action Executor &rarr; Recovery Check -- scored against each
        incident's injected ground truth.
      </p>

      {aggregate === null ? (
        <p className="mt-4 rounded border border-dashed border-slate-700 p-6 text-center text-sm text-slate-400">
          {skipped
            ? 'Operational metrics were skipped for this run (it was generated with --skip-operational).'
            : "No operational aggregate present in this run's results."}
        </p>
      ) : (
        <>
          <dl className="mt-4 grid grid-cols-2 gap-x-6 gap-y-4 text-sm sm:grid-cols-3">
            <Field label="Remediation success rate">{formatPct(aggregate.remediation_success_rate)}</Field>
            <Field label="Recovery-verification accuracy">
              {formatPct(aggregate.recovery_verification_accuracy)}
            </Field>
            <Field label="Wrong-remediation rate">{formatPct(aggregate.wrong_remediation_rate)}</Field>
            <Field label="In-scope incidents">
              {aggregate.n_in_scope} of {aggregate.n_incidents}
            </Field>
            <Field label="Wrong-remediation attempts">{aggregate.n_wrong_remediation_attempts}</Field>
            <Field label="Scored (ok / errors)">
              {aggregate.n_ok} / {aggregate.n_errors}
            </Field>
          </dl>
          <p className="mt-4 text-xs text-slate-500">
            "In-scope" excludes incidents whose recommended plan was SAFE-only or whose HIGH_IMPACT
            action was rejected -- those never count toward the remediation-success or
            recovery-verification denominators (see
            <code className="mx-1 rounded bg-slate-800 px-1 py-0.5">
              run_experiments._aggregate_operational
            </code>
            's docstring).
          </p>
        </>
      )}
    </section>
  )
}
