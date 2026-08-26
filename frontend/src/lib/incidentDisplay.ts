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
