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

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openrouter import ChatOpenRouter

from backend.agents.schemas import DiagnosisResult, Hypothesis
from backend.agents.state import IncidentState
from backend.agents.structured_output import TRANSIENT_OPENROUTER_ERRORS, invoke_structured
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
- hypotheses must contain EXACTLY ONE entry: your single chosen root cause,
  with hypotheses[0].category equal to root_cause_category. Do not list
  other candidates you considered here -- those belong in
  alternative_hypotheses below. Populate its confidence (0.0-1.0) with your
  own heuristic estimate -- this is compared against your strongest
  runner-up's confidence to decide whether more investigation is needed, so
  give it real thought rather than a placeholder value.
- alternative_hypotheses must contain AT LEAST ONE entry whenever you
  seriously considered more than one category (only leave it empty if
  literally no other category was plausible) -- each with its own rationale
  and confidence, strongest runner-up first.
- evidence you cite must be grounded in the tool/source_ref combinations
  you were actually given above -- never invent a record id.
- evidence from search_historical_incidents must cite via source_ref.query
  (the historical incident's string id, e.g. "hist-012"), never via
  source_ref.record_id -- record_id is only for the integer-keyed
  telemetry tools (get_logs/get_metrics/get_deployments/get_dependencies).
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
        # No temperature/top_p override -- kept unset for consistent,
        # prompt-driven behavior; explicit max_tokens, matching
        # investigator.py's conventions. max_tokens must cover BOTH a
        # model's reasoning spend and its real output -- OpenRouter's
        # unified `reasoning` field bills chain-of-thought against this same
        # ceiling. Empirically (live GLM-5.3-flash calls against this
        # node's exact prompt shape, 2026-09) reasoning-token spend is
        # highly variable per call -- observed from a few dozen tokens up to
        # a full 4096-token budget exhausted on an *identical* prompt across
        # repeated calls, with no reliable correlation to prompt content.
        # OpenRouter's `reasoning={"max_tokens": N}` param was tried as an
        # alternative cap and found NOT to be reliably enforced by this
        # model/route (asked for 16, observed 200+ actually spent), so the
        # fix here is real headroom on the shared ceiling, not a reasoning
        # cap. A full budget exhaustion means zero tokens left for the
        # actual structured tool call, which surfaces as `None` out of
        # `invoke_structured` (see structured_output.py) -- consistently
        # across retries, since the underlying cause doesn't change.
        llm = ChatOpenRouter(
            model=settings.rca_model,
            api_key=settings.openrouter_api_key,
            max_tokens=16384,
        )
        # Free-tier OpenRouter models sit behind a shared, fluctuating-capacity
        # pool -- transient 502/429 "upstream overloaded" responses are routine,
        # not exceptional, and the openrouter SDK doesn't retry them internally
        # for this response shape. Retry at the LangChain Runnable level instead,
        # narrowed to the actually-transient error subset (see
        # structured_output.TRANSIENT_OPENROUTER_ERRORS) -- a bad API key or
        # oversized prompt should fail fast, not burn 5 backoff attempts first.
        structured_llm = llm.with_structured_output(DiagnosisResult).with_retry(
            retry_if_exception_type=TRANSIENT_OPENROUTER_ERRORS, stop_after_attempt=5
        )

        messages = [
            SystemMessage(content=RCA_SYSTEM_PROMPT),
            HumanMessage(content=_build_rca_prompt(state)),
        ]
        # Free-tier models occasionally ignore the forced tool_choice and
        # reply with plain text instead -- the parser turns that into a
        # silent `None` rather than an exception, so `.with_retry()` above
        # never fires. Retry that case here too before giving up.
        result = invoke_structured(structured_llm, messages, DiagnosisResult)

        # Graceful degradation for weak/reasoning models. GLM-5.3-Flash (and
        # similar free-tier models) intermittently omit the `hypotheses`
        # wrapper from their structured output entirely -- despite
        # RCA_SYSTEM_PROMPT's explicit "EXACTLY ONE entry" instruction -- while
        # still committing to a `root_cause_category` and `diagnostic_confidence`.
        # `DiagnosisResult.hypotheses` is deliberately NOT `min_length`-
        # constrained (that constraint is shared across the A/B/C/D eval schema,
        # where an empty list is meaningful signal, and it converted this weak-
        # model omission into hard 500s -- confirmed live, incidents 17080/17085,
        # 2026-09-01). Repair rather than reject: the category + evidence the
        # model DID produce is a usable, explainable diagnosis, so synthesize the
        # single hypothesis the model failed to wrap rather than crashing an
        # otherwise-complete incident over a missing field. This runs ONLY here
        # (the graph's diagnosis producer); `investigator.py` stays frozen.
        # `root_cause_category` is a required Literal (never None), so an empty
        # `hypotheses` list is the only condition to guard on here.
        hypotheses = result.hypotheses
        if not hypotheses:
            hypotheses = [
                Hypothesis(
                    category=result.root_cause_category,
                    rationale=(
                        "Synthesized from the model's chosen root_cause_category; "
                        "the model did not return an explicit hypothesis list."
                    ),
                    confidence=result.diagnostic_confidence,
                )
            ]

        existing_descriptions = {item.description for item in state.evidence}
        merged_evidence = list(state.evidence) + [
            item for item in result.evidence if item.description not in existing_descriptions
        ]

        return {
            "root_cause": result.root_cause_category,
            "hypotheses": hypotheses,
            "alternative_hypotheses": result.alternative_hypotheses,
            "diagnostic_confidence": result.diagnostic_confidence,
            "evidence": merged_evidence,
            "incident_status": IncidentStatus.DIAGNOSED,
        }

    return root_cause_node
