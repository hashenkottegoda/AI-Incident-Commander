"""One-off operational scripts (run manually / in CI / at deploy time).

Nothing here is imported by the FastAPI app itself — these are ops-side
setup steps (schema provisioning, etc.), same category as Alembic
migrations, not runtime application code.
"""
