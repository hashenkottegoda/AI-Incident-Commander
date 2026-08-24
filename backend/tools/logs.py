"""`get_logs` — query `LogEntry` rows for a service/time window/level.

See `backend/tools/README.md`-equivalent notes in this package's
`__init__.py` docstring for the binding pattern (factory closes over a
per-request `db: Session`, `@tool(parse_docstring=True)` derives the
LLM-facing schema from `_get_logs`'s signature/docstring).
"""

from __future__ import annotations

from langchain_core.tools import BaseTool, tool
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.models import LogEntry, LogLevel
from backend.tools.common import parse_time_window, resolve_service
from backend.tools.schemas import LogRecord


def get_logs(
    db: Session,
    service: str,
    start: str,
    end: str,
    level: str | None = None,
) -> list[LogRecord]:
    """Query application log lines for one service within a time window.

    Args:
        db: Request-scoped database session (not LLM-facing).
        service: Service name, e.g. "checkout-service".
        start: ISO 8601 timestamp, inclusive window start.
        end: ISO 8601 timestamp, exclusive window end.
        level: Optional log level filter ("info", "warn", or "error").
            Omit to return all levels.

    Returns:
        Matching `LogRecord`s ordered by timestamp ascending. Empty list
        means the query was valid but matched no log lines.

    Raises:
        ValueError: unknown service, malformed timestamp, `start >= end`,
            or an unrecognized `level`.
    """
    svc = resolve_service(db, service)
    start_dt, end_dt = parse_time_window(start, end)

    stmt = select(LogEntry).where(
        LogEntry.service_id == svc.id,
        LogEntry.timestamp >= start_dt,
        LogEntry.timestamp < end_dt,
    )
    if level is not None:
        try:
            level_enum = LogLevel(level)
        except ValueError as exc:
            known = [member.value for member in LogLevel]
            raise ValueError(f"unknown log level {level!r}; expected one of {known}") from exc
        stmt = stmt.where(LogEntry.level == level_enum)
    stmt = stmt.order_by(LogEntry.timestamp)

    rows = db.execute(stmt).scalars().all()
    return [
        LogRecord(
            id=row.id,
            service=svc.name,
            timestamp=row.timestamp,
            level=row.level.value,
            message=row.message,
            attributes=row.attributes,
        )
        for row in rows
    ]


def make_get_logs_tool(db: Session) -> BaseTool:
    """Bind `get_logs` to `db`, returning a LangChain tool with `db` hidden
    from the LLM-facing schema (only `service`/`start`/`end`/`level` are
    exposed — see this package's `__init__.py` docstring for why a closure
    factory is the correct pattern here rather than `@tool` directly on a
    function whose first parameter is `db`)."""

    @tool("get_logs", parse_docstring=True)
    def _get_logs_tool(
        service: str, start: str, end: str, level: str | None = None
    ) -> list[dict]:
        """Query application log lines for one service within a time window.

        Use this to look for error bursts, warnings, or discrete events
        (e.g. a feature flag being enabled) around an incident.

        Args:
            service: Service name, e.g. "checkout-service".
            start: ISO 8601 timestamp, inclusive window start.
            end: ISO 8601 timestamp, exclusive window end.
            level: Optional log level filter ("info", "warn", or "error").
                Omit to return all levels.
        """
        # `mode="json"` here matters: LangChain's tool-result serializer
        # (`langchain_core.tools.base._stringify`) does `json.dumps` and
        # falls back to Python `repr()` on TypeError for anything that
        # isn't already a JSON-native type — a `list[BaseModel]` hits that
        # fallback and the LLM would see repr syntax
        # (`datetime.datetime(...)`, single-quoted strings) instead of
        # clean JSON. Returning plain dicts here keeps the direct-call
        # `get_logs()` above typed/Pydantic for tests while making what the
        # LLM actually receives real JSON.
        return [
            record.model_dump(mode="json") for record in get_logs(db, service, start, end, level)
        ]

    return _get_logs_tool
