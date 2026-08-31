"""Phase 5's INVESTIGATION node: the graph's bounded ReAct tool-calling step.

BUILD_PLAN.md's Agent Architecture section: *"INVESTIGATION (claude-opus-4-8,
ReAct tool loop: logs/metrics/deployments/dependencies/db-status/config;
traces = optional secondary signal)."* Per Phase 2's tool layer, "db-status"/
"config" have no dedicated tools (see `backend/tools/__init__.py`'s
docstring) -- the four real tools are logs/metrics/deployments/dependencies,
via `backend.tools.build_tools(db)`.

## Why this duplicates (adapts), rather than shares, `investigator.py`'s loop

`backend.agents.investigator._run_react_loop` is Phase 3's Experiment B
baseline and must stay byte-for-byte unchanged (Phase 7's eval harness
needs to keep running it as-is) -- this task's instructions are explicit
that module is frozen. This node's ReAct loop is adapted from that same
bounded-loop pattern (bind tools, loop until the model stops calling tools
or the budget is exhausted) but is NOT a shared helper, because the two
loops actually do different things with each tool result:

1. Only `backend.tools.build_tools(db)` is bound here -- the 4
   Postgres-backed tools. `search_historical_incidents` is deliberately
   NOT offered to this node (BUILD_PLAN.md's graph-flow diagram): in the
   full graph RAG is its own always-run downstream node, not something
   left to the ReAct loop's discretion the way Phase 3's baseline does it.
2. Every tool call this node makes is turned into one structured
   `EvidenceItem` *programmatically*, the moment the tool result comes
   back -- including calls that return zero records ("no deployments in
   this window" is itself a real finding). This is what makes the
   evidence-sufficiency conditional-edge predicate
   (`backend.agents.routing.evidence_sufficiency_check_failed`) a
   deterministic check against real tool coverage, rather than depending
   on whether a *second* LLM extraction pass happened to mention a given
   tool's result as "evidence" (which is how Phase 3's baseline builds
   `DiagnosisResult.evidence`, and is fine there since Phase 3 has no
   coverage predicate riding on it).

Given how much of the surrounding code differs (evidence synthesis,
tool_call_log_ids bookkeeping, only 4 tools, a re-investigation-aware
prompt), factoring a shared helper for just the innermost while-loop would
add an import coupling between a frozen Phase 3 module and live Phase 5
code for very little real duplication saved -- some duplication here is the
lower-risk choice, per this task's own guidance.
"""

from __future__ import annotations

import json
import logging

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_openrouter import ChatOpenRouter
from sqlalchemy.orm import Session

from backend.agents.schemas import EvidenceItem, SourceRef
from backend.agents.state import IncidentState
from backend.config import get_settings
from backend.models import IncidentStatus
from backend.tools import build_tools

logger = logging.getLogger(__name__)

# Same reasoning as investigator.py's MAX_TOOL_CALLS: enough to touch every
# bound tool type against the affected service (and re-investigation passes
# get their own fresh budget of MAX_TOOL_CALLS, on top of whatever the prior
# pass already gathered -- the *iteration* count, not this per-pass budget,
# is what routing.MAX_REINVESTIGATION_LOOPS bounds), while still blocking
# the "repeatedly query the same thing" runaway-loop failure mode.
MAX_TOOL_CALLS = 10

SYSTEM_PROMPT = """You are the investigation step of a production incident investigation system.

Your job is to gather real evidence with the tools available to you
(get_logs, get_metrics, get_deployments, get_dependencies) -- do not guess
without evidence, and do not repeatedly call the same tool with the same
arguments. A separate downstream step handles historical-incident lookup
and the final root-cause diagnosis; your job is only to gather evidence.

Investigate the affected service's logs, metrics, recent deployments, and
downstream dependencies around the incident time. Look for a temporally
coherent chain of evidence (e.g. a deployment, followed by a metric
anomaly, followed by error logs) rather than relying on a single isolated
signal. If the affected service calls other services, consider checking
those services too (get_dependencies tells you which downstream services
it calls) -- the loudest symptom is not always where the root cause is.

You have a limited budget of tool calls, so be deliberate: prefer a
reasonably wide time window and a small number of well-chosen queries over
many narrow, redundant ones. Always check both get_deployments and
get_dependencies at least once -- a recent deployment and downstream
dependency involvement are both explicitly required checks, even when the
answer turns out to be "nothing relevant found."

When you have gathered enough evidence -- or you are told your tool-call
budget is exhausted -- stop calling tools and briefly summarize your
findings in plain text.
"""

REINVESTIGATION_PREFIX = """This is a follow-up investigation pass: the previous pass's \
diagnosis was not accepted (either the top two hypotheses' confidence was \
too close together, or evidence coverage was incomplete). Focus \
specifically on closing the gap -- if evidence so far implicates a \
downstream dependency, trace the chain further (e.g. check that \
dependency's own service, not just the originally reported one) rather \
than re-confirming what you already found.

Evidence gathered so far:
{evidence_summary}
"""

_BUDGET_EXHAUSTED_NOTICE = (
    "You have reached your tool-call budget ({budget} calls). Stop calling "
    "tools now and rely on what you have already found."
)


def _tool_by_name(tools: list[BaseTool], name: str) -> BaseTool | None:
    for tool in tools:
        if tool.name == name:
            return tool
    return None


def _compact_args(args: dict) -> str:
    return ", ".join(f"{key}={value!r}" for key, value in sorted(args.items()))


def _describe_tool_result(
    tool_name: str, args: dict, records: list[dict]
) -> tuple[str, int | None, str | None]:
    """Build a programmatic (non-LLM-authored) `(description, record_id,
    query)` triple for one tool call's result -- see this module's
    docstring for why this is done in code rather than via a second LLM
    extraction pass.

    Every branch returns a genuinely informative description, including
    the zero-record case ("no deployments found" is real evidence, not a
    non-event) -- callers always get one `EvidenceItem` per tool call,
    which is what makes tool coverage checkable via `evidence[]` alone.
    """
    args_desc = _compact_args(args)
    count = len(records)
    if count == 0:
        return (f"{tool_name} found no matching records ({args_desc}).", None, args_desc)

    if tool_name == "get_logs":
        top = records[-1]
        return (
            f"{tool_name} returned {count} log record(s) ({args_desc}); most recent: "
            f"[{top.get('level')}] {top.get('message')!r} at {top.get('timestamp')}.",
            top.get("id"),
            None,
        )
    if tool_name == "get_metrics":
        top = records[-1]
        return (
            f"{tool_name} returned {count} metric point(s) ({args_desc}); most recent "
            f"{top.get('metric_name')}={top.get('value')} at {top.get('timestamp')}.",
            top.get("id"),
            None,
        )
    if tool_name == "get_deployments":
        top = records[-1]
        return (
            f"{tool_name} returned {count} deployment(s) ({args_desc}); most recent "
            f"version {top.get('version')!r} deployed at {top.get('deployed_at')}.",
            top.get("id"),
            None,
        )
    if tool_name == "get_dependencies":
        top = records[-1]
        downstream = top.get("downstream_service") or "none"
        return (
            f"{tool_name} returned {count} dependency span(s) ({args_desc}); most recent "
            f"span {top.get('span_name')!r} -> downstream={downstream}, "
            f"duration_ms={top.get('duration_ms')}.",
            top.get("id"),
            None,
        )
    # Defensive fallback -- should not happen given build_tools(db)'s fixed
    # 4-tool set, but stays generic rather than raising if a tool is added
    # later.
    return (f"{tool_name} returned {count} record(s) ({args_desc}).", None, args_desc)


def _incident_context_message(state: IncidentState) -> str:
    return (
        f"Incident #{state.incident_id}\n"
        f"Affected service(s): {', '.join(state.affected_services) or 'unknown'}\n"
        f"Severity: {state.severity.value if state.severity is not None else 'unknown'}\n\n"
        "Investigate this incident using the available tools."
    )


def make_investigation_node(db: Session):
    """Return a LangGraph node function bound to one request-scoped `db`.

    Factory pattern matches `backend.tools`'s `make_get_*_tool(db)` and
    Phase 3's `investigate_incident(db, ...)` -- one `Session` per
    request/graph run, explicit rather than global.
    """

    def investigation_node(state: IncidentState) -> dict:
        settings = get_settings()
        tools = build_tools(db)

        # No temperature/top_p override -- kept unset for consistent,
        # prompt-driven behavior; explicit max_tokens, matching
        # investigator.py's conventions.
        llm = ChatOpenRouter(
            model=settings.investigation_model,
            api_key=settings.openrouter_api_key,
            max_tokens=4096,
        )
        llm_with_tools = llm.bind_tools(tools)

        messages: list[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]
        if state.evidence:
            evidence_summary = "\n".join(
                f"- [{item.source_ref.tool}] {item.description}" for item in state.evidence
            )
            messages.append(
                HumanMessage(content=REINVESTIGATION_PREFIX.format(evidence_summary=evidence_summary))
            )
        messages.append(HumanMessage(content=_incident_context_message(state)))

        new_evidence: list[EvidenceItem] = []
        new_tool_call_ids: list[int] = []
        next_id = (state.tool_call_log_ids[-1] + 1) if state.tool_call_log_ids else 1

        tool_calls_made = 0
        while True:
            response: AIMessage = llm_with_tools.invoke(messages)
            messages.append(response)

            if not response.tool_calls:
                break  # model concluded this investigation pass on its own

            tool_results: list[ToolMessage] = []
            for call in response.tool_calls:
                tool_calls_made += 1
                tool = _tool_by_name(tools, call["name"])
                status = "success"
                records: list[dict] = []
                if tool is None:
                    content = f"Error: unknown tool {call['name']!r}"
                    status = "error"
                else:
                    try:
                        result = tool.invoke(call["args"])
                        records = result if isinstance(result, list) else []
                        content = result if isinstance(result, str) else json.dumps(result)
                    except Exception as exc:  # noqa: BLE001 - surfaced to the model as a tool error
                        logger.warning("tool %s raised during investigation: %s", call["name"], exc)
                        content = f"Error: {exc}"
                        status = "error"

                tool_results.append(
                    ToolMessage(
                        content=content, tool_call_id=call["id"], name=call["name"], status=status
                    )
                )

                if status == "success":
                    description, record_id, query = _describe_tool_result(
                        call["name"], call["args"], records
                    )
                    new_evidence.append(
                        EvidenceItem(
                            description=description,
                            source_ref=SourceRef(
                                tool=call["name"], record_id=record_id, query=query
                            ),
                        )
                    )
                    new_tool_call_ids.append(next_id)
                    next_id += 1

            messages.extend(tool_results)

            if tool_calls_made >= MAX_TOOL_CALLS:
                messages.append(
                    HumanMessage(content=_BUDGET_EXHAUSTED_NOTICE.format(budget=MAX_TOOL_CALLS))
                )
                break

        return {
            "evidence": list(state.evidence) + new_evidence,
            "tool_call_log_ids": list(state.tool_call_log_ids) + new_tool_call_ids,
            "incident_status": IncidentStatus.INVESTIGATING,
            "investigation_iterations": state.investigation_iterations + 1,
        }

    return investigation_node
