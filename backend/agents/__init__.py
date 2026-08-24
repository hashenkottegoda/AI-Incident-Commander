"""LangGraph node functions (Phase 3 onward).

Phase 3 adds a single node — `backend.agents.investigator.investigate_incident`
— implemented as a plain callable rather than a `StateGraph`, per
`investigator.py`'s module docstring. Later phases add `triage.py`,
`rca.py`, `response_planner.py`, `executor.py`, `recovery.py` here and
assemble them into the real graph in `backend/graph.py` (BUILD_PLAN.md's
Repository Structure).
"""
