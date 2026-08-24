"""LangGraph node functions (Phase 3 onward).

Phase 3 adds a single node — `backend.agents.investigator.investigate_incident`
— implemented as a plain callable rather than a `StateGraph`, per
`investigator.py`'s module docstring (frozen — this remains Experiment B's
baseline for Phase 7's eval harness).

Phase 5 adds the real `StateGraph` nodes: `state.py` (`IncidentState`),
`triage_node.py`, `investigation_node.py`, `rag_node.py`,
`root_cause_node.py`, and the conditional-edge predicates in `routing.py`,
assembled into the compiled graph in `backend/graph.py` (BUILD_PLAN.md's
Repository Structure). Later phases add `response_planner.py`,
`executor.py`, `recovery.py` here.
"""
