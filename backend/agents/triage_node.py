"""Phase 5's TRIAGE node: `claude-haiku-4-5`, no tools bound, one cheap
structured-output call.

BUILD_PLAN.md's Agent Architecture section: *"TRIAGE node (`claude-haiku-4-5`
via `get_settings().triage_model`, no tools bound): a cheap, fast first
pass. Given the incident's initial detection context (service, detected_at,
severity as reported by the alerting system ...) classify/confirm
affected_services and set incident_status to triaging then investigating.
Keep this genuinely lightweight -- no tool calls, a single structured-output
call is enough."*

`severity` here is a realistic input, not a ground-truth leak: it's already
on the `Incident` row because a monitoring/alerting system assigned it at
detection time, before any investigation happened -- this node isn't
inferring it from evidence, just carrying forward what the alert already
said (see `backend.graph.initial_state`, which seeds `IncidentState.severity`
straight from the `Incident` row before the graph starts).

## `incident_status` transition

The state this node receives already has `incident_status == TRIAGING` (set
by `backend.graph.initial_state` before the graph starts running -- that's
the "triaging" half of BUILD_PLAN.md's "set incident_status to triaging then
investigating"). This node's job is only the second half: once triage
completes, hand off `INVESTIGATING`.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openrouter import ChatOpenRouter
from pydantic import BaseModel, Field

from backend.agents.state import IncidentState
from backend.config import get_settings
from backend.models import IncidentStatus

TRIAGE_SYSTEM_PROMPT = """You are the triage step of a production incident investigation system.

You are given the initial detection context for one incident -- the
service an alerting system flagged, when it was detected, and the severity
the alerting system assigned. Your only job is to confirm (or, if the
context clearly implies it, expand) the list of affected services. Do not
guess at a root cause and do not invent additional affected services with
no basis in the context you were given -- if you have no reason to add a
service beyond the one reported, confirm the reported service as-is.
"""


class TriageResult(BaseModel):
    """Small structured-output shape for the Triage node.

    Deliberately NOT `DiagnosisResult` -- BUILD_PLAN.md: "Use
    `.with_structured_output(...)` for whatever small triage output shape
    you need (define a small schema for this if needed, doesn't need to be
    `DiagnosisResult`)." Triage classifies/confirms scope; it doesn't
    diagnose anything.
    """

    affected_services: list[str] = Field(
        min_length=1,
        description=(
            "Confirmed list of services affected by this incident, starting "
            "from the reported service and adding any other service the "
            "initial context clearly implicates. Do not invent services "
            "with no basis in the given context."
        ),
    )
    triage_notes: str | None = Field(
        default=None,
        description="Optional one-sentence rationale for the classification.",
    )


def _triage_prompt(state: IncidentState) -> str:
    reported_services = ", ".join(state.affected_services) or "unknown"
    severity = state.severity.value if state.severity is not None else "unknown"
    return (
        f"Incident #{state.incident_id}\n"
        f"Reported affected service(s): {reported_services}\n"
        f"Severity (as assigned by the alerting system that created this "
        f"incident): {severity}\n\n"
        "Confirm the affected service list for this incident."
    )


def make_triage_node():
    """Return a LangGraph node function bound to `get_settings().triage_model`.

    A factory (even though this node has nothing request-scoped to close
    over, unlike the Investigation/RAG nodes) so every node in
    `backend/graph.py` follows the same `make_*_node(...) -> node_fn`
    shape.
    """

    def triage_node(state: IncidentState) -> dict:
        settings = get_settings()
        # No temperature/top_p override -- kept unset for consistent,
        # prompt-driven behavior; explicit max_tokens, matching
        # investigator.py's conventions. max_tokens is small: this is a
        # classification call, not a reasoning-heavy one.
        llm = ChatOpenRouter(
            model=settings.triage_model,
            api_key=settings.openrouter_api_key,
            max_tokens=512,
        )
        structured_llm = llm.with_structured_output(TriageResult)

        messages = [
            SystemMessage(content=TRIAGE_SYSTEM_PROMPT),
            HumanMessage(content=_triage_prompt(state)),
        ]
        result = structured_llm.invoke(messages)

        if not isinstance(result, TriageResult):
            # with_structured_output's default "include_raw=False" should
            # always return the parsed model; defensive guard only.
            raise TypeError(f"expected TriageResult from structured output, got {type(result)!r}")

        return {
            "affected_services": result.affected_services,
            "incident_status": IncidentStatus.INVESTIGATING,
        }

    return triage_node
