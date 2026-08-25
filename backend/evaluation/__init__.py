"""Phase 7's evaluation package (BUILD_PLAN.md Phase 7).

Home for the ground-truth incident dataset and the A/B/C/D experiment
runner that compares agent architectures against `DiagnosisResult`
(`backend.agents.schemas`):

- A — context-stuffing baseline (`backend.evaluation.experiment_a`):
  no tools, no selective retrieval, one giant prompt, one LLM call.
- B — `backend.agents.investigator.investigate_incident(db, incident,
  include_rag=False)`: single ReAct tool-calling loop, no RAG.
- C — `backend.agents.investigator.investigate_incident(db, incident,
  include_rag=True)`: same loop, with `search_historical_incidents`.
- D — `backend.graph.run_incident_graph`: the full multi-agent
  LangGraph system.

All four experiments emit the same `DiagnosisResult` schema so Phase 7's
scoring code (root-cause accuracy, evidence precision, hallucination
rate, tool-call efficiency, latency, token cost, human override rate) is
written once and applies uniformly across the comparison table.
"""

from __future__ import annotations
