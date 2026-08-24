"""SQLAlchemy models package (Phase 1 core data model).

Every module here defines tables that subclass `backend.db.Base`. Nothing
imports this package automatically — `alembic/env.py` imports it
explicitly so the models register on `Base.metadata` before
`alembic revision --autogenerate` inspects it. Application code should
import model classes from here (`from backend.models import Service`)
rather than reaching into individual submodules.
"""

from backend.models.audit import (
    AuditDecisionStatus,
    AuditEvent,
    ExecutionOutcome,
    RiskClassification,
)
from backend.models.deployment import Deployment
from backend.models.incident import Incident, IncidentStatus, Severity
from backend.models.service import Service
from backend.models.telemetry import LogEntry, LogLevel, MetricPoint, TraceLite

__all__ = [
    "AuditDecisionStatus",
    "AuditEvent",
    "Deployment",
    "ExecutionOutcome",
    "Incident",
    "IncidentStatus",
    "LogEntry",
    "LogLevel",
    "MetricPoint",
    "RiskClassification",
    "Service",
    "Severity",
    "TraceLite",
]
