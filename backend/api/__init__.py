"""FastAPI routers for the AI Incident Commander backend.

Routers are added here as later phases build them out (incidents,
evaluation). Phase 0 shipped the health check router; Phase 1 adds
`simulation.py` (`POST /api/simulation/failure`, `POST
/api/simulation/reset`).
"""
