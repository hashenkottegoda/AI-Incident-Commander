"""Shared input validation for the Phase 2 tool layer.

Every tool in this package takes the same two "coordinates" — a service
name and an ISO 8601 time window — so the validation lives here once
rather than four times. Both helpers raise a plain `ValueError` with a
specific, actionable message on bad input (unknown service, malformed
timestamp, `start >= end`) rather than returning an empty list: BUILD_PLAN's
task spec is explicit that an agent needs to be able to tell "no evidence
found" apart from "you asked the tool something invalid." An empty list is
only ever returned for a *valid* query that legitimately matched no rows.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.models import Service


def resolve_service(db: Session, service_name: str) -> Service:
    """Look up `Service` by name, raising `ValueError` if it doesn't exist."""
    service = db.execute(select(Service).where(Service.name == service_name)).scalar_one_or_none()
    if service is None:
        known = sorted(db.execute(select(Service.name)).scalars().all())
        raise ValueError(f"unknown service {service_name!r}; known services: {known}")
    return service


def parse_time_window(start: str, end: str) -> tuple[datetime, datetime]:
    """Parse and validate an ISO 8601 `[start, end)` window.

    Accepts a trailing `Z` (Python 3.11+ `datetime.fromisoformat` handles
    it natively). Naive timestamps (no offset) are assumed UTC, since every
    telemetry timestamp in this system is stored timezone-aware in UTC.
    Raises `ValueError` if either string doesn't parse, or if
    `start >= end`.
    """
    start_dt = _parse_timestamp(start, "start")
    end_dt = _parse_timestamp(end, "end")
    if start_dt >= end_dt:
        raise ValueError(f"start ({start!r}) must be strictly before end ({end!r})")
    return start_dt, end_dt


def _parse_timestamp(value: str, label: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"invalid ISO 8601 {label} timestamp {value!r}: {exc}"
        ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed
