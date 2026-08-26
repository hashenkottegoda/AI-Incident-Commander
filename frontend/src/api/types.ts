/**
 * TypeScript types mirroring the FastAPI backend's Pydantic response
 * models, kept as close to the source-of-truth as possible (same field
 * names, same shape) rather than inventing a client-side representation.
 *
 * Sources (read directly, not guessed):
 *   - backend/api/health.py       -> HealthResponse
 *   - backend/api/incidents.py    -> IncidentSummary, IncidentInfo,
 *                                     InvestigationState, AuditEventSummary,
 *                                     IncidentDetail
 *   - backend/agents/schemas.py   -> RootCauseCategory, SourceRef,
 *                                     EvidenceItem, Hypothesis
 *   - backend/api/approvals.py    -> ApprovalRequest, ApprovalResponse
 *   - backend/api/evaluation.py   -> EvaluationRunMetadata,
 *                                     DiagnosticAggregate,
 *                                     OperationalAggregate,
 *                                     EvaluationResultsResponse
 *   - backend/models/incident.py  -> IncidentStatus, Severity
 *   - backend/models/audit.py     -> RiskClassification, AuditDecisionStatus,
 *                                     ExecutionOutcome
 *
 * If a backend model changes, update the matching type here in the same
 * change -- these are not meant to drift into a separate "client contract."
 */

// --- backend/api/health.py ---------------------------------------------

export interface HealthResponse {
  status: string
}

// --- backend/models/incident.py ------------------------------------------

export type IncidentStatus =
  | 'detected'
  | 'triaging'
  | 'investigating'
  | 'diagnosed'
  | 'awaiting_approval'
  | 'executing'
  | 'verifying'
  | 'resolved'
  | 'manual_intervention_required'

export type Severity = 'P1' | 'P2' | 'P3' | 'P4'

// --- backend/models/audit.py ---------------------------------------------

export type RiskClassification = 'safe' | 'high_impact'

export type AuditDecisionStatus =
  | 'pending_approval'
  | 'approved'
  | 'rejected'
  | 'auto_executed'
  | 'executed'

export type ExecutionOutcome = 'recovered' | 'still_degraded'

// --- backend/agents/schemas.py --------------------------------------------

export type RootCauseCategory =
  | 'database_connection_pool'
  | 'memory_resource_exhaustion'
  | 'application_bug'
  | 'upstream_dependency_failure'
  | 'inefficient_database_query'
  | 'upstream_dependency_timeout'
  | 'unknown'

export interface SourceRef {
  tool: string
  record_id: number | null
  query: string | null
}

export interface EvidenceItem {
  description: string
  source_ref: SourceRef
}

export interface Hypothesis {
  category: RootCauseCategory
  rationale: string
  /** Optional per-hypothesis heuristic confidence (0-1), not always populated. */
  confidence: number | null
}

export interface DiagnosisResult {
  root_cause_category: RootCauseCategory
  hypotheses: Hypothesis[]
  alternative_hypotheses: Hypothesis[]
  evidence: EvidenceItem[]
  diagnostic_confidence: number
}

// --- backend/api/incidents.py ---------------------------------------------

export interface IncidentSummary {
  id: number
  status: IncidentStatus
  severity: Severity
  failure_type: string
  detected_at: string
  service_id: number
  service_name: string
}

export interface IncidentInfo extends IncidentSummary {
  /** Injected ground-truth root cause category (predicted-vs-actual, eval-only). */
  root_cause_category: string
}

export interface InvestigationState {
  /** The graph's own live-checkpoint phase -- may diverge from IncidentSummary.status; see backend/api/incidents.py's docstring. */
  incident_status: IncidentStatus
  evidence: EvidenceItem[]
  hypotheses: Hypothesis[]
  root_cause: RootCauseCategory | null
  diagnostic_confidence: number
  alternative_hypotheses: Hypothesis[]
  recommended_actions: Record<string, unknown>[]
  approval_decision: string | null
  execution_result_id: number | number[] | null
  recovery_result: Record<string, unknown> | null
}

export interface AuditEventSummary {
  id: number
  action_type: string
  risk_classification: RiskClassification
  decision_status: AuditDecisionStatus
  approver: string | null
  execution_outcome: ExecutionOutcome | null
  execution_detail: Record<string, unknown> | null
  recommended_at: string
  decided_at: string | null
  executed_at: string | null
}

export interface IncidentDetail {
  incident: IncidentInfo
  /** null means no LangGraph checkpoint exists yet for this incident (not an error). */
  investigation: InvestigationState | null
  audit_events: AuditEventSummary[]
}

export interface ListIncidentsParams {
  status?: IncidentStatus
  limit?: number
  offset?: number
}

// --- backend/api/approvals.py ----------------------------------------------

export interface ApprovalRequest {
  approver: string
}

export interface ApprovalResponse {
  incident_id: number
  decision: 'approved' | 'rejected' | 'already_decided'
  incident_status: IncidentStatus
  audit_event_ids: number[]
  approver: string | null
  decided_at: string | null
  /** True only when this call actually resumed the paused graph thread. */
  resumed: boolean
}

// --- backend/api/evaluation.py ----------------------------------------------

export interface EvaluationRunMetadata {
  seed: number
  count: number
  generated_at: string
  skip_operational: boolean
}

export interface DiagnosticAggregate {
  n_incidents: number
  n_ok: number
  n_errors: number
  root_cause_accuracy_rate: number | null
  mean_evidence_precision: number | null
  mean_hallucination_rate: number | null
  mean_tool_call_count: number | null
  mean_evidence_per_tool_call: number | null
  mean_latency_seconds: number | null
  mean_total_tokens: number | null
}

export interface OperationalAggregate {
  n_incidents: number
  n_ok: number
  n_errors: number
  n_in_scope: number
  remediation_success_rate: number | null
  recovery_verification_accuracy: number | null
  wrong_remediation_rate: number | null
  n_wrong_remediation_attempts: number
}

export interface EvaluationResultsResponse {
  metadata: EvaluationRunMetadata
  per_incident: Record<string, unknown>[]
  /** Keyed by architecture, e.g. "A" | "B" | "C" | "D". */
  diagnostic_aggregate: Record<string, DiagnosticAggregate>
  operational_aggregate: OperationalAggregate | null
}
