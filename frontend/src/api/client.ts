/**
 * Typed client for the FastAPI backend (see backend/main.py).
 *
 * Base URL comes from VITE_API_BASE_URL (frontend/.env / .env.local),
 * falling back to the FastAPI default dev port (`uv run uvicorn
 * backend.main:app --reload` binds :8000) so the smoke-test page works
 * out of the box with no env setup.
 *
 * Only `fetchHealth` has a real caller so far (the health-check smoke
 * page). `listIncidents`/`getIncident`/`approveIncident`/`rejectIncident`/
 * `getEvaluationResults` are stubbed out now, against the exact response
 * shapes in `./types.ts`, so the incident-list/detail/evaluation pages
 * that come next don't need to re-derive the API contract.
 */

import type {
  ApprovalRequest,
  ApprovalResponse,
  EvaluationResultsResponse,
  HealthResponse,
  IncidentDetail,
  IncidentSummary,
  ListIncidentsParams,
} from './types'

export const API_BASE_URL: string =
  (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? 'http://localhost:8000'

export class ApiError extends Error {
  readonly status: number
  readonly url: string

  constructor(message: string, status: number, url: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.url = url
  }
}

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const url = `${API_BASE_URL}${path}`
  const response = await fetch(url, {
    ...init,
    headers: {
      Accept: 'application/json',
      ...(init?.body ? { 'Content-Type': 'application/json' } : {}),
      ...init?.headers,
    },
  })

  if (!response.ok) {
    const detail = await response.text().catch(() => '')
    throw new ApiError(
      `${response.status} ${response.statusText}${detail ? `: ${detail}` : ''}`,
      response.status,
      url,
    )
  }

  return (await response.json()) as T
}

/** `GET /health` -- the whole-stack smoke test (mirrors Phase 0's own verify step). */
export function fetchHealth(): Promise<HealthResponse> {
  return apiFetch<HealthResponse>('/health')
}

/** `GET /api/incidents` -- most-recently-detected first. See IncidentSummary's
 * docstring in types.ts for the `?status=` filter's caveat (DB status only,
 * not the graph's live checkpoint phase). */
export function listIncidents(params: ListIncidentsParams = {}): Promise<IncidentSummary[]> {
  const query = new URLSearchParams()
  if (params.status) query.set('status', params.status)
  if (params.limit !== undefined) query.set('limit', String(params.limit))
  if (params.offset !== undefined) query.set('offset', String(params.offset))
  const queryString = query.toString()
  return apiFetch<IncidentSummary[]>(`/api/incidents${queryString ? `?${queryString}` : ''}`)
}

/** `GET /api/incidents/{id}` -- incident + investigation state + audit trail. */
export function getIncident(incidentId: number): Promise<IncidentDetail> {
  return apiFetch<IncidentDetail>(`/api/incidents/${incidentId}`)
}

/** `POST /api/incidents/{id}/approve` -- approve pending high-impact action(s). */
export function approveIncident(
  incidentId: number,
  request: ApprovalRequest,
): Promise<ApprovalResponse> {
  return apiFetch<ApprovalResponse>(`/api/incidents/${incidentId}/approve`, {
    method: 'POST',
    body: JSON.stringify(request),
  })
}

/** `POST /api/incidents/{id}/reject` -- reject pending high-impact action(s). */
export function rejectIncident(
  incidentId: number,
  request: ApprovalRequest,
): Promise<ApprovalResponse> {
  return apiFetch<ApprovalResponse>(`/api/incidents/${incidentId}/reject`, {
    method: 'POST',
    body: JSON.stringify(request),
  })
}

/** `GET /api/evaluation/results` -- most recent A/B/C/D experiment run. 404
 * (surfaced as a thrown ApiError) if `run_experiments` has never been run. */
export function getEvaluationResults(): Promise<EvaluationResultsResponse> {
  return apiFetch<EvaluationResultsResponse>('/api/evaluation/results')
}
