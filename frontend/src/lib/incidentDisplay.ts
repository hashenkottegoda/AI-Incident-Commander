/**
 * Small display helpers shared between `IncidentListPage` and
 * `IncidentDetailPage` -- pulled into their own module (rather than
 * exported from a page component file) so oxlint's react-refresh rule
 * doesn't warn about a page file exporting non-component values.
 */

import type { IncidentSummary } from '../api/types'

export const SEVERITY_STYLES: Record<IncidentSummary['severity'], string> = {
  P1: 'bg-red-950/60 text-red-300 border-red-800',
  P2: 'bg-orange-950/60 text-orange-300 border-orange-800',
  P3: 'bg-amber-950/60 text-amber-300 border-amber-800',
  P4: 'bg-slate-800/60 text-slate-300 border-slate-700',
}

const dateFormatter = new Intl.DateTimeFormat(undefined, {
  dateStyle: 'medium',
  timeStyle: 'short',
})

export function formatDetectedAt(iso: string): string {
  const date = new Date(iso)
  if (Number.isNaN(date.getTime())) return iso
  return dateFormatter.format(date)
}

/** Human-readable labels for `backend.graph.build_incident_graph`'s node
 * names (`GET /{incident_id}/progress`'s `node_name` values) -- kept as a
 * plain lookup, not a TS enum, matching that column's own "not constrained
 * to today's set" design (see `backend/models/node_progress.py`'s
 * docstring: the graph's node set is expected to keep growing). Any
 * `node_name` not in this map still renders -- see `formatNodeName` --
 * rather than the panel breaking on an unrecognized value. */
const NODE_NAME_LABELS: Record<string, string> = {
  triage: 'Triage',
  investigation: 'Investigation',
  rag: 'Historical Incident Search',
  root_cause: 'Root Cause Analysis',
  response_planner: 'Response Planning',
  human_approval: 'Human Approval',
  action_executor: 'Action Execution',
  recovery_check: 'Recovery Check',
}

export function formatNodeName(nodeName: string): string {
  return NODE_NAME_LABELS[nodeName] ?? nodeName
}
