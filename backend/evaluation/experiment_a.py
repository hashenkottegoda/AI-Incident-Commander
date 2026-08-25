"""Experiment A — the context-stuffing baseline (BUILD_PLAN.md Phase 7).

BUILD_PLAN.md's exact framing: *"A — Context-stuffing baseline: all
relevant telemetry for the incident window (logs/metrics/deployments)
dumped into a single prompt -> LLM -> root cause. No tools, no selective
retrieval. (Defining A as 'no data' would make it score ~0 by construction
and turn A->B into a trivial has-data/no-data result -- this framing keeps
it meaningful.)"*

This is the "one shot, everything up front" end of the architecture
spectrum Phase 7 compares -- the opposite of Experiment D's full
multi-agent graph. It exists so the A/B/C/D comparison table can show a
genuine "does tool-calling / selective retrieval / multi-agent
orchestration actually help over just dumping everything into a huge
prompt" result, rather than "does having any data at all help" (which
would be true by construction and uninteresting).

## Window sizing

Unlike Experiments B/C/D, this architecture has no ReAct loop and cannot
iteratively widen its search if the window it queries turns out too
narrow for a given scenario -- it gets exactly one shot at the window,
then exactly one LLM call. Handicapping it with a narrow, one-size window
would attribute a data-access limitation to the architecture itself,
biasing the A/B/C/D comparison unfairly against A rather than measuring
what the comparison is supposed to measure (architecture, holding data
access "fair" across all four).

So the window is deliberately generous and scenario-agnostic:
`incident.detected_at` minus `WINDOW_BEFORE` (8 hours) through
`incident.detected_at` plus `WINDOW_AFTER` (30 minutes).

- 8 hours before comfortably covers every scenario's actual telemetry
  footprint. `backend.simulation.injector.SCENARIO_TIMING_OVERRIDES`
  gives `memory_leak` (the widest of the 6 scenarios, by design -- "a
  slow, organic leak ... that accumulates over hours") a 6-hour
  `pre_incident_window`; every other scenario uses the 45-minute
  `DEFAULT_PRE_INCIDENT_WINDOW`. 8 hours has real headroom over the
  6-hour worst case rather than sitting right at the edge.
- 30 minutes after `detected_at` is slack for the "shortly after" half of
  a realistic incident window. `inject_failure` never actually writes
  telemetry after `incident_start - detection_lag` (all injected
  evidence lands strictly *before* `detected_at`), so this is pure safety
  margin against edge timing, not evidence that would otherwise be
  missed.

## Real record ids, so citation is possible without a tool loop

`investigate_incident`'s docstring establishes the "real ids in context"
principle for the tool-loop architectures (B/C/D): their ReAct loop's tool
results, replayed back to the model at structured-output time, carry each
row's genuine database `id`, which is what lets the model cite a real
`source_ref.record_id` instead of inventing one. Experiment A has no
ReAct loop and therefore no discrete tool results to replay -- so this
module reproduces the same principle a different way: every row in the
dumped telemetry is formatted with its real `id` prefixed inline, e.g.
`[id=123] 2026-07-01T12:00:00Z ERROR checkout-service: ...`, giving the
model real ids to cite exactly as the tool-loop architectures do. Without
this, Phase 7's hallucination-rate metric would be unfairly comparing "A
had no way to cite a real id" against "B/C/D could" -- a prompt-formatting
gap, not a genuine architectural one.

## Model choice

Reuses `get_settings().investigation_model`, the same model Experiment
B/C use for their ReAct-loop-plus-diagnosis call. BUILD_PLAN.md doesn't
name a distinct model role for Experiment A, and giving it a *different*
model would confound Phase 7's comparison: a difference in results would
then be partly "architecture" and partly "which model," when the whole
point of the A/B/C/D table is to isolate architecture as the variable
under test. Same model, same `max_tokens`, no `temperature`/`top_p`
(BUILD_PLAN.md's Tech Stack notes these models reject sampling params).

## Tool functions, not tools

`get_logs`/`get_metrics`/`get_deployments` are called here as plain
Python functions (`backend.tools.logs.get_logs`, etc.), not bound via
`make_get_*_tool`/`ChatAnthropic.bind_tools()` -- Experiment A is defined
as "no tools, no selective retrieval," so there is no LLM-driven tool
loop to bind them into. `get_dependencies` is deliberately not queried
here: BUILD_PLAN.md's Experiment A description names only
"logs/metrics/deployments."

## All canonical services, not just the incident's nominal service

`_assemble_context` queries logs/metrics/deployments for EVERY canonical
service (`backend.simulation.scenario_schema.CANONICAL_SERVICES`), not
only `incident.service.name`. This was a real bug in an earlier version
of this module, caught in code review: `cascading_payment_timeout` and
`dependency_failure` both have `affected_service: checkout-service`, but
their actual root-cause evidence (payment-service's own latency ramp,
timeout/error-response logs, canary-flag-enabled log) lives on
payment-service -- both scenario YAML files document this explicitly as
evidence "only visible by querying payment-service's own logs/metrics."
Scoping the dump to one service structurally blinded Experiment A to that
evidence for reasons unrelated to "no selective retrieval" (the actual
architectural variable Phase 7 measures), which would have biased the
A/B/C/D comparison for exactly the two scenarios BUILD_PLAN calls out as
testing cross-service reasoning. `investigate_incident`'s tool-loop
architectures (B/C/D) are not restricted to one service either -- their
system prompt explicitly tells them to check services the affected
service calls -- so querying all 3 services here keeps data ACCESS fair
across all four experiments, with only the ACCESS METHOD (dump vs.
selective tool calls vs. orchestrated multi-step) varying, which is the
actual point of the comparison.
"""

from __future__ import annotations

from datetime import timedelta

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from sqlalchemy.orm import Session

from backend.agents.schemas import DiagnosisResult
from backend.config import get_settings
from backend.models import Incident
from backend.simulation.baseline import BASELINE_METRICS
from backend.simulation.scenario_schema import CANONICAL_SERVICES
from backend.tools.deployments import get_deployments
from backend.tools.logs import get_logs
from backend.tools.metrics import get_metrics
from backend.tools.schemas import DeploymentRecord, LogRecord, MetricRecord

# See module docstring's "Window sizing" for the reasoning behind both
# constants -- 8h/30m is a fixed, scenario-agnostic choice (Experiment A
# gets exactly one shot at the window, unlike B/C/D's iterative loop).
WINDOW_BEFORE: timedelta = timedelta(hours=8)
WINDOW_AFTER: timedelta = timedelta(minutes=30)

SYSTEM_PROMPT = """You are an incident investigation agent for a production
system. You have been handed ALL of the telemetry (logs, metrics, and
deployments) available for one incident's investigation window, dumped in
full below. There are no tools available to you and no way to query for
more data -- this is everything you get. Read through it and determine
the incident's root_cause_category from a fixed set of categories, using
only the evidence given below.

Every row below is prefixed with its real database record id in the form
"[id=<n>]". When you cite evidence, use these exact ids in
source_ref.record_id -- never invent an id that doesn't appear below.

Two resource-pressure metrics can look identical on the surface (e.g. a
climbing connection count can mean the pool is genuinely exhausted, OR it
can mean queries are individually slow and holding connections longer
than usual). When a metric alone is ambiguous between two plausible
causes, look for an earlier, lower-severity log line (e.g. a "warn"-level
diagnostic message) among the logs below -- it often appears before the
louder error burst that follows it and is the evidence that distinguishes
between two otherwise-similar-looking metric patterns.

Look for a temporally coherent chain of evidence (e.g. a deployment,
followed by a metric anomaly, followed by error logs) rather than relying
on a single isolated signal. An empty DEPLOYMENTS section is itself
meaningful evidence (no recent deploy), not a gap in your data.
"""

STRUCTURED_OUTPUT_INSTRUCTION = """Based on the telemetry dumped above --
including its exact record ids, timestamps, and values -- produce the
final structured diagnosis.

Rules:
- root_cause_category must be one of the fixed categories, or "unknown" if
  the evidence genuinely doesn't clearly support any of them.
- Every evidence item's source_ref must cite a row that actually appears
  above; when the evidence is backed by one specific record, set
  source_ref.record_id to the exact id shown in that row's "[id=<n>]"
  prefix -- never invent a record id that isn't in the dump above.
- hypotheses[0].category should equal root_cause_category.
- diagnostic_confidence is your own heuristic estimate (0.0-1.0), not a
  calibrated probability.
"""


def _format_logs(records: list[LogRecord]) -> str:
    if not records:
        return "(no log lines in this window)"
    lines = []
    for record in records:
        attrs = f" attrs={record.attributes}" if record.attributes else ""
        lines.append(
            f"[id={record.id}] {record.timestamp.isoformat()} "
            f"{record.level.upper()} {record.service}: {record.message}{attrs}"
        )
    return "\n".join(lines)


def _format_metrics(records: list[MetricRecord]) -> str:
    if not records:
        return "(no metric samples in this window)"
    return "\n".join(
        f"[id={record.id}] {record.timestamp.isoformat()} "
        f"{record.service} {record.metric_name}={record.value:.4f}"
        for record in records
    )


def _format_deployments(records: list[DeploymentRecord]) -> str:
    if not records:
        return "(no deployments in this window)"
    return "\n".join(
        f"[id={record.id}] {record.deployed_at.isoformat()} "
        f"{record.service} deployed version={record.version}"
        for record in records
    )


def _known_metric_names(service_name: str) -> list[str]:
    """Every metric name `backend.simulation.baseline` generates for this
    service. `get_metrics` requires a single `metric_name` per call (Phase
    2's tool signature), so Experiment A loops over the known set rather
    than being able to "scan all metrics" in one query -- see
    `BASELINE_METRICS` (also the source table `inject_failure`'s
    `METRIC_RAMP` entries perturb), which is the authoritative list of
    metric names this system ever writes for a given service."""
    return [baseline.name for baseline in BASELINE_METRICS.get(service_name, ())]


def _assemble_context(db: Session, incident: Incident, start: str, end: str) -> str:
    """Gather all telemetry across every canonical service (see module
    docstring's "All canonical services" section for why this must NOT be
    scoped to just `incident.service.name`) across `[start, end)`, via the
    plain tool functions (no LangChain tool binding), and format it into
    one large, citable text block."""
    service_name = incident.service.name
    all_services = sorted(CANONICAL_SERVICES)

    logs: list[LogRecord] = []
    deployments: list[DeploymentRecord] = []
    metrics: list[MetricRecord] = []
    metric_names_by_service: dict[str, list[str]] = {}
    for name in all_services:
        # No `level` filter: Experiment A dumps everything, not a curated
        # subset -- that's the point of this baseline (see module docstring).
        logs.extend(get_logs(db, name, start, end))
        deployments.extend(get_deployments(db, name, start, end))

        metric_names = _known_metric_names(name)
        metric_names_by_service[name] = metric_names
        for metric_name in metric_names:
            metrics.extend(get_metrics(db, name, metric_name, start, end))

    logs.sort(key=lambda record: record.timestamp)
    deployments.sort(key=lambda record: record.deployed_at)
    metrics.sort(key=lambda record: (record.timestamp, record.service, record.metric_name))

    metric_names_desc = ", ".join(
        f"{name}: {', '.join(metric_names_by_service[name]) or 'none known'}"
        for name in all_services
    )

    return (
        f"=== Incident #{incident.id} ===\n"
        f"Affected service (per initial detection): {service_name}\n"
        f"Detected at: {incident.detected_at.isoformat()}\n"
        f"Severity: {incident.severity.value}\n"
        f"Investigation window: {start} to {end}\n"
        f"Services covered by this telemetry dump: {', '.join(all_services)}\n\n"
        f"=== LOGS ({len(logs)} rows, all services) ===\n"
        f"{_format_logs(logs)}\n\n"
        f"=== METRICS ({len(metrics)} rows, all services; known metric names per "
        f"service: {metric_names_desc}) ===\n"
        f"{_format_metrics(metrics)}\n\n"
        f"=== DEPLOYMENTS ({len(deployments)} rows, all services) ===\n"
        f"{_format_deployments(deployments)}\n"
    )


def run_context_stuffing_baseline(db: Session, incident: Incident) -> DiagnosisResult:
    """Run Experiment A: dump all telemetry for `incident`'s investigation
    window into one prompt and make exactly one Claude API call for the
    diagnosis. No tools, no ReAct loop, no selective retrieval -- see
    module docstring for the full reasoning behind window sizing, id
    citability, and model choice.

    Mirrors `backend.agents.investigator.investigate_incident`'s
    `(db, incident) -> DiagnosisResult` shape so Phase 7's experiment
    runner can call all four experiments (A/B/C/D) uniformly.
    """
    settings = get_settings()
    start = (incident.detected_at - WINDOW_BEFORE).isoformat()
    end = (incident.detected_at + WINDOW_AFTER).isoformat()

    context = _assemble_context(db, incident, start, end)

    # No temperature/top_p: these models reject sampling params
    # (BUILD_PLAN.md Tech Stack), same as `investigate_incident`.
    llm = ChatAnthropic(
        model=settings.investigation_model,
        api_key=settings.anthropic_api_key,
        max_tokens=4096,
    )
    # `.with_structured_output(...)` + a single `.invoke(...)` below is the
    # entire interaction with the model -- no tool binding, no loop, one
    # shot in, one `DiagnosisResult` out.
    structured_llm = llm.with_structured_output(DiagnosisResult)

    messages: list[BaseMessage] = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=context + "\n\n" + STRUCTURED_OUTPUT_INSTRUCTION),
    ]
    result = structured_llm.invoke(messages)

    if not isinstance(result, DiagnosisResult):
        # with_structured_output's default "include_raw=False" should always
        # return the parsed model; this is a defensive guard, not an
        # expected path (mirrors investigate_incident's same guard).
        raise TypeError(f"expected DiagnosisResult from structured output, got {type(result)!r}")

    return result
