"""Phase 3's single-agent investigator — BUILD_PLAN.md's Experiment B baseline.

BUILD_PLAN.md Phase 3: *"One LangGraph node: `ChatAnthropic` + `ToolNode`
ReAct loop, ends in a `DiagnosisResult` ... Validates tool-calling mechanics
end-to-end before the full graph is built."*

## Why a plain function, not a `StateGraph`

Phase 3 is explicitly pre-full-graph ("before the full graph is built").
Wrapping this in a single-node `StateGraph` now would mean standing up
`IncidentState`, a checkpointer, and conditional-edge plumbing for
machinery this phase doesn't use yet (no interrupt, no re-investigation
loop, no multi-node routing — that's Phase 5). A plain function that IS
the node body is the more honest shape: it does real ReAct tool-calling
via `ChatAnthropic.bind_tools()` exactly as the eventual graph node will,
and Phase 5 can lift this logic into a `StateGraph` node without changing
what it does internally. Building the graph shell now would be premature
scaffolding for state this phase doesn't have a use for.

## Model choice: `investigation_model`, not `rca_model`

This node does both evidence-gathering (ReAct loop) *and* final diagnosis
(structured output) in one call, because Phase 3 is scoped to exactly one
node. `investigation_model` is used throughout — for the tool-calling loop
*and* the final structured-output call — for two reasons: (1) this node's
dominant cost and behavior is the ReAct tool loop, which is squarely the
Investigation role; (2) reusing the same `ChatAnthropic` instance for both
calls avoids a mid-run model swap, which per BUILD_PLAN.md's agent-design
guidance would invalidate any prompt cache built up over the tool-calling
turns. `investigation_model` and `rca_model` currently default to the same
model ID anyway (`claude-opus-4-8`) — Phase 5 splits investigation and RCA
into genuinely separate graph nodes, at which point the RCA half moves to
`rca_model` for real.

## Tool-call budget

`MAX_TOOL_CALLS = 10`: enough to touch all five tool types
(logs/metrics/deployments/dependencies/search_historical_incidents)
against the affected service and one hop of dependencies, with room for a
couple of follow-up/narrower queries, while still bounding cost and
blocking the "repeatedly query the same thing" runaway-loop failure mode
the idea doc calls out. This is a real cost/safety control, not a
decorative constant. The budget is
enforced *between* turns, not by truncating a turn's tool calls mid-way —
Claude can request several tool calls in one turn, and the API requires
every `tool_use` block in a turn to get a matching `tool_result` before
the next turn, so a turn is always completed in full even if it pushes the
running total slightly past the budget; the check that stops the loop runs
after each turn completes.

## Preserving real tool-call history for evidence citation

The final structured-output call reuses the *entire* accumulated message
history — including every `ToolMessage` with the tool's actual JSON
result (each record's real `id`) — as context. This is what lets the model
cite genuine `record_id`s in `DiagnosisResult.evidence[].source_ref`
rather than inventing them, which is the plumbing Phase 7's hallucination-
rate metric depends on later.
"""

from __future__ import annotations

import json
import logging

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from sqlalchemy.orm import Session

from backend.agents.schemas import DiagnosisResult
from backend.config import get_settings
from backend.models import Incident
from backend.rag.qdrant_client import get_qdrant_client
from backend.tools import build_rag_tools, build_tools

logger = logging.getLogger(__name__)

MAX_TOOL_CALLS = 10

SYSTEM_PROMPT = """You are an incident investigation agent for a production system.

You have been handed one incident to investigate. Your job is to determine
its root_cause_category from a fixed set of categories by gathering real
evidence with the tools available to you (get_logs, get_metrics,
get_deployments, get_dependencies, search_historical_incidents). Do not
guess without evidence, and do not repeatedly call the same tool with the
same arguments.

Once you have gathered enough evidence from logs/metrics/deployments/
dependencies to describe the incident (its service, symptoms, any recent
change, any dependency involvement, and a short timeline), consider
calling search_historical_incidents to check whether this presentation
matches a past incident -- a real historical match with its similarity
score and how it was resolved is useful corroborating evidence, not a
replacement for the evidence you already gathered.

Investigate the affected service's logs, metrics, recent deployments, and
downstream dependencies around the incident time. Look for a temporally
coherent chain of evidence (e.g. a deployment, followed by a metric
anomaly, followed by error logs) rather than relying on a single isolated
signal. If the affected service calls other services, consider checking
those services too -- the loudest symptom is not always where the root
cause is.

Two resource-pressure metrics can look identical on the surface (e.g. a
climbing connection count can mean the pool is genuinely exhausted, OR it
can mean queries are individually slow and holding connections longer than
usual -- very different root causes with the same metric signature).
When a metric alone is ambiguous between two plausible causes, call
get_logs WITHOUT a level filter (not just level="error") across the full
window you are investigating: an earlier, lower-severity log line (e.g. a
"warn"-level detection/diagnostic message) is often the one piece of
evidence that distinguishes between two otherwise-similar-looking metric
patterns, and it usually appears before the louder error burst that
follows it -- don't stop widening your search once you find *an*
explanation if a metric's shape is still ambiguous between two categories.

You have a limited budget of tool calls, so be deliberate: prefer a
reasonably wide time window and a small number of well-chosen queries over
many narrow, redundant ones.

When you have gathered enough evidence -- or you are told your tool-call
budget is exhausted -- stop calling tools and briefly summarize your
findings in plain text (a structured extraction pass follows, so you do
not need to produce JSON yourself; just state what you found, including
specific record ids/timestamps/values, and your conclusion).
"""

STRUCTURED_OUTPUT_INSTRUCTION = """Based on the entire investigation above --
including the exact tool results returned (their record ids, timestamps,
and values) -- produce the final structured diagnosis.

Rules:
- root_cause_category must be one of the fixed categories, or "unknown" if
  the evidence genuinely doesn't clearly support any of them.
- Every evidence item's source_ref must cite a tool you actually called
  above; when the evidence is backed by one specific record, set
  source_ref.record_id to the exact `id` field from that tool call's JSON
  result -- never invent a record id that wasn't actually returned.
- hypotheses[0].category should equal root_cause_category.
- diagnostic_confidence is your own heuristic estimate (0.0-1.0), not a
  calibrated probability.
"""

_BUDGET_EXHAUSTED_NOTICE = (
    "You have reached your tool-call budget ({budget} calls). Stop calling "
    "tools now and rely on what you have already found."
)


def _incident_context_message(incident: Incident) -> str:
    return (
        f"Incident #{incident.id}\n"
        f"Affected service: {incident.service.name}\n"
        f"Detected at: {incident.detected_at.isoformat()}\n"
        f"Severity: {incident.severity.value}\n\n"
        "Investigate this incident using the available tools and determine "
        "its root_cause_category."
    )


def _tool_by_name(tools: list[BaseTool], name: str) -> BaseTool | None:
    for tool in tools:
        if tool.name == name:
            return tool
    return None


def _run_react_loop(llm_with_tools, tools: list[BaseTool], messages: list[BaseMessage]) -> None:
    """Mutate `messages` in place, running the ReAct tool-calling loop until
    the model stops calling tools or `MAX_TOOL_CALLS` is reached.

    Bounded by construction: each iteration that continues the loop must
    have made at least one tool call, and the loop exits once the running
    total reaches `MAX_TOOL_CALLS` -- so the number of loop iterations can
    never exceed `MAX_TOOL_CALLS`, independent of the explicit check below.
    """
    tool_calls_made = 0

    while True:
        response: AIMessage = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            return  # model concluded the investigation on its own

        tool_results: list[ToolMessage] = []
        for call in response.tool_calls:
            tool_calls_made += 1
            tool = _tool_by_name(tools, call["name"])
            status = "success"
            if tool is None:
                content = f"Error: unknown tool {call['name']!r}"
                status = "error"
            else:
                try:
                    result = tool.invoke(call["args"])
                    content = result if isinstance(result, str) else json.dumps(result)
                except Exception as exc:  # noqa: BLE001 - surfaced to the model as a tool error
                    logger.warning("tool %s raised during investigation: %s", call["name"], exc)
                    content = f"Error: {exc}"
                    status = "error"
            # `status="error"` (not just the "Error: ..." text) is what makes
            # langchain-anthropic mark the corresponding tool_result block as
            # `is_error` for the API -- without it a genuine tool failure
            # looks like a successful call whose result happens to say
            # "Error", losing the model's structured signal to treat it as a
            # failure rather than data.
            tool_results.append(
                ToolMessage(
                    content=content, tool_call_id=call["id"], name=call["name"], status=status
                )
            )
        messages.extend(tool_results)

        if tool_calls_made >= MAX_TOOL_CALLS:
            messages.append(HumanMessage(content=_BUDGET_EXHAUSTED_NOTICE.format(budget=MAX_TOOL_CALLS)))
            return


def investigate_incident(db: Session, incident: Incident) -> DiagnosisResult:
    """Run the Phase 3 baseline investigator against one incident.

    Binds `backend.tools.build_tools(db)` to a `ChatAnthropic` instance
    (`get_settings().investigation_model`, no `temperature`/`top_p` --
    these models reject sampling params), runs a bounded ReAct tool-calling
    loop, then makes one final `.with_structured_output(DiagnosisResult)`
    call against the same model with the full conversation (including real
    tool results) as context. Returns the resulting `DiagnosisResult`.
    """
    settings = get_settings()
    # RAG tool bound separately (backend.tools.build_rag_tools) because it
    # depends on a QdrantClient, not the request-scoped `db: Session` every
    # other Phase 2 tool needs -- see backend/tools/historical_incidents.py's
    # docstring. get_qdrant_client() is itself a cached, connection-less
    # constructor (no network I/O happens until a tool call actually
    # searches), so adding it here doesn't change this function's
    # lazy-until-invoked behavior.
    tools = build_tools(db) + build_rag_tools(get_qdrant_client())

    # No temperature/top_p: these models reject sampling params (BUILD_PLAN.md
    # Tech Stack). max_tokens is set explicitly rather than relying on the
    # SDK default, per the claude-api skill's guidance for non-streaming calls.
    llm = ChatAnthropic(
        model=settings.investigation_model,
        api_key=settings.anthropic_api_key,
        max_tokens=4096,
    )
    llm_with_tools = llm.bind_tools(tools)

    messages: list[BaseMessage] = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=_incident_context_message(incident)),
    ]

    _run_react_loop(llm_with_tools, tools, messages)

    # Final structured-output call: same ChatAnthropic instance (no tools
    # bound this time), through LangChain's structured-output binding --
    # never free-text parsing, never the raw Anthropic SDK's messages.parse().
    structured_llm = llm.with_structured_output(DiagnosisResult)
    messages.append(HumanMessage(content=STRUCTURED_OUTPUT_INSTRUCTION))
    result = structured_llm.invoke(messages)

    if not isinstance(result, DiagnosisResult):
        # with_structured_output's default "include_raw=False" should always
        # return the parsed model; this is a defensive guard, not an
        # expected path.
        raise TypeError(f"expected DiagnosisResult from structured output, got {type(result)!r}")

    return result
