"""Unit tests for `backend.agents.root_cause_node`'s synthesize-fallback.

No real OpenRouter call: `ChatOpenRouter` is monkeypatched in the node
module with a fake whose structured output returns a caller-chosen
`DiagnosisResult`, matching the fake shape used across
`tests/test_node_progress.py` / `tests/test_graph_response_planner_e2e.py`.

The behavior under test is the graceful degradation added for weak/reasoning
models (e.g. GLM-5.3-Flash) that intermittently omit the `hypotheses` wrapper
from their structured output while still committing to a
`root_cause_category`: rather than crashing an otherwise-complete diagnosis,
`root_cause_node` synthesizes a single hypothesis from the category the model
did produce. `DiagnosisResult.hypotheses` is deliberately NOT
`min_length`-constrained (that schema is shared with the A/B/C/D eval, where
an empty list is meaningful signal), so this repair lives in the node, not the
schema.
"""

from __future__ import annotations

import backend.agents.root_cause_node as rca_module
from backend.agents.schemas import DiagnosisResult, Hypothesis
from backend.agents.state import IncidentState
from backend.models import IncidentStatus


class _FakeStructuredLLM:
    def __init__(self, result):
        self._result = result

    def invoke(self, messages):  # noqa: ARG002 - signature-compatible stand-in
        return self._result

    def with_retry(self, **kwargs):  # noqa: ARG002
        return self


def _make_fake_chat_openrouter(result: DiagnosisResult):
    class _FakeRootCauseChatOpenRouter:
        def __init__(self, *args, **kwargs):
            pass

        def with_structured_output(self, schema):  # noqa: ARG002
            return _FakeStructuredLLM(result)

    return _FakeRootCauseChatOpenRouter


def test_synthesizes_single_hypothesis_when_model_omits_hypotheses(monkeypatch):
    """Model returns a `root_cause_category` + `diagnostic_confidence` but an
    empty `hypotheses` list (the GLM-5.3-Flash omission case). The node must
    return exactly one synthesized hypothesis whose category matches
    `root_cause_category` and whose confidence mirrors
    `diagnostic_confidence` -- never an empty list."""
    model_output = DiagnosisResult(
        root_cause_category="database_connection_pool",
        diagnostic_confidence=0.62,
        hypotheses=[],  # the wrapper the weak model failed to emit
        alternative_hypotheses=[],
        evidence=[],
    )
    monkeypatch.setattr(
        rca_module, "ChatOpenRouter", _make_fake_chat_openrouter(model_output)
    )

    node = rca_module.make_root_cause_node()
    update = node(IncidentState(incident_id=1, affected_services=["inventory-service"]))

    assert update["root_cause"] == "database_connection_pool"
    assert update["incident_status"] is IncidentStatus.DIAGNOSED
    assert len(update["hypotheses"]) == 1
    synthesized = update["hypotheses"][0]
    assert synthesized.category == "database_connection_pool"
    assert synthesized.confidence == 0.62
    assert synthesized.rationale  # a non-empty honest note, not blank


def test_model_supplied_hypotheses_pass_through_unchanged(monkeypatch):
    """When the model DOES return hypotheses, the node must not touch them --
    the fallback only fires on an empty list, so a compliant model's ranked
    hypotheses reach state exactly as produced."""
    supplied = [
        Hypothesis(
            category="memory_resource_exhaustion",
            rationale="heap grew unbounded",
            confidence=0.8,
        )
    ]
    model_output = DiagnosisResult(
        root_cause_category="memory_resource_exhaustion",
        diagnostic_confidence=0.8,
        hypotheses=supplied,
        alternative_hypotheses=[],
        evidence=[],
    )
    monkeypatch.setattr(
        rca_module, "ChatOpenRouter", _make_fake_chat_openrouter(model_output)
    )

    node = rca_module.make_root_cause_node()
    update = node(IncidentState(incident_id=2, affected_services=["inventory-service"]))

    assert update["hypotheses"] == supplied
    assert update["hypotheses"][0].rationale == "heap grew unbounded"
