"""Phase 5's ROOT CAUSE / HYPOTHESIS node: `get_settings().rca_model`,
structured output into the shared `DiagnosisResult` schema.

BUILD_PLAN.md's Agent Architecture section: *"ROOT CAUSE / HYPOTHESIS
(claude-opus-4-8, structured output: ranked hypotheses + enum
root_cause_category + structured cited evidence) ... Include the RAG
matches as additional context in the prompt (a real historical match is
corroborating evidence, not the deciding factor)."*

This node reuses `backend.agents.schemas.DiagnosisResult` for its structured
output -- the same schema Phase 3's baseline produces (BUILD_PLAN.md: "All
four experiments emit the same DiagnosisResult schema") -- but maps its
fields onto `IncidentState`'s separate `root_cause`/`hypotheses`/
`alternative_hypotheses`/`diagnostic_confidence` fields rather than storing
a nested `DiagnosisResult` object, matching `IncidentState`'s literal field
list from BUILD_PLAN.md's Agent Architecture section.

## Evidence: read, not replaced

`state.evidence` (Investigation's tool-call-grounded findings + RAG's
historical matches, both tagged with a real `source_ref.tool`) is passed
into this node's prompt as-is and is NOT overwritten by this node's own
`DiagnosisResult.evidence` output -- only genuinely new items (by
description) get merged in. Letting a second, unrelated LLM call freely
rewrite `evidence[]` would risk exactly the kind of ungrounded citation
Phase 7's hallucination-rate metric is designed to catch; keeping
Investigation's programmatically-grounded evidence as the base and only
appending is the safer default.
"""

from __future__ import annotations

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from backend.agents.schemas import DiagnosisResult
from backend.agents.state import IncidentState
from backend.config import get_settings
from backend.models import IncidentStatus

RCA_SYSTEM_PROMPT = """You are the root-cause analysis step of a production incident \
investigation system.

You are given the accumulated evidence gathered by an investigation agent
-- each item is a structured, factual finding citing the real tool call it
came from (tool name in brackets). Some items come from
search_historical_incidents: a semantic search over past incident
writeups. Historical matches are corroborating context, not authoritative
-- weigh them alongside the evidence you were actually given, and do not
treat a historical match's category as the deciding factor if the rest of
the evidence doesn't support it.

Produce a structured diagnosis:
- root_cause_category must be one of the fixed categories, or "unknown" if
  the evidence genuinely doesn't clearly support any of them.
- hypotheses is ranked, most likely first; hypotheses[0].category must
  equal root_cause_category. Populate each hypothesis's confidence
  (0.0-1.0) with your own heuristic estimate for that specific hypothesis
  -- this is compared against the runner-up's confidence to decide whether
  more investigation is needed, so give it real thought rather than a
  placeholder value.
- alternative_hypotheses holds categories you seriously considered but did
  not choose, each with its own rationale and confidence -- put your
  strongest runner-up first.
- evidence you cite must be grounded in the tool/source_ref combinations
  you were actually given above -- never invent a record id.
- diagnostic_confidence is your own overall heuristic (0.0-1.0), not a
  calibrated probability.

If the evidence you were given doesn't cover a recent deployment check or a
downstream dependency check for the affected service, say so explicitly in
your reasoning -- a follow-up investigation pass may be triggered to close
that gap.
"""


def _evidence_lines(state: IncidentState) -> str:
    if not state.evidence:
        return "(no evidence gathered)"
    return "\n".join(f"- [{item.source_ref.tool}] {item.description}" for item in state.evidence)


def _build_rca_prompt(state: IncidentState) -> str:
    severity = state.severity.value if state.severity is not None else "unknown"
    return (
        f"Incident #{state.incident_id}\n"
        f"Affected service(s): {', '.join(state.affected_services) or 'unknown'}\n"
        f"Severity: {severity}\n\n"
        f"Accumulated evidence:\n{_evidence_lines(state)}\n\n"
        "Produce the structured root-cause diagnosis now."
    )


def make_root_cause_node():
    """Return a LangGraph node function bound to `get_settings().rca_model`.

    A factory (no per-request resource to close over besides settings) so
    every node in `backend/graph.py` follows the same `make_*_node(...) ->
    node_fn` shape as the Investigation/RAG nodes.
    """

    def root_cause_node(state: IncidentState) -> dict:
        settings = get_settings()
        # No temperature/top_p (these models reject sampling params);
        # explicit max_tokens, matching investigator.py's conventions.
        llm = ChatAnthropic(
            model=settings.rca_model,
            api_key=settings.anthropic_api_key,
            max_tokens=4096,
        )
        structured_llm = llm.with_structured_output(DiagnosisResult)

        messages = [
            SystemMessage(content=RCA_SYSTEM_PROMPT),
            HumanMessage(content=_build_rca_prompt(state)),
        ]
        result = structured_llm.invoke(messages)

        if not isinstance(result, DiagnosisResult):
            # with_structured_output's default "include_raw=False" should
            # always return the parsed model; defensive guard only.
            raise TypeError(
                f"expected DiagnosisResult from structured output, got {type(result)!r}"
            )

        existing_descriptions = {item.description for item in state.evidence}
        merged_evidence = list(state.evidence) + [
            item for item in result.evidence if item.description not in existing_descriptions
        ]

        return {
            "root_cause": result.root_cause_category,
            "hypotheses": result.hypotheses,
            "alternative_hypotheses": result.alternative_hypotheses,
            "diagnostic_confidence": result.diagnostic_confidence,
            "evidence": merged_evidence,
            "incident_status": IncidentStatus.DIAGNOSED,
        }

    return root_cause_node
