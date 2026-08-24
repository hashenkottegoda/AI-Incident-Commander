"""Phase 2's tool layer: typed Python functions bound to Claude's tool-use
format via LangChain (`langchain-anthropic`/`langchain-core`, per
BUILD_PLAN.md's "single integration path" rule — no raw `anthropic` SDK
tool schemas here).

## Tool set

Grounded in what Phase 1's schema actually makes queryable — no tool
exists for data we don't have:

- `get_logs`        -> `backend.models.LogEntry`   (service, time window, level)
- `get_metrics`     -> `backend.models.MetricPoint` (service, metric_name, time window)
- `get_deployments` -> `backend.models.Deployment`  (service, time window)
- `get_dependencies`-> `backend.models.TraceLite`   (service, time window; downstream + duration_ms)
- `search_historical_incidents` (Phase 4, `backend/tools/historical_incidents.py`)
  -> Qdrant `historical_incidents` collection (structured incident summary
  in, ranked historical matches + similarity score out). Bound to a
  `QdrantClient`, not a `db: Session` -- see `build_rag_tools` below and
  that module's docstring for why it's a parallel aggregator rather than
  folded into `build_tools`.

BUILD_PLAN.md's Agent Architecture section also mentions "db-status" and
"config" in the Investigation node's tool list. There is no separate
config/db-status table in Phase 1 — `LogEntry.attributes` already carries
structured event data (e.g. feature-flag state via
`payment_canary_flag_enabled`), and `MetricPoint` already carries
`db_connections_active`. So that evidence is reached through `get_logs`/
`get_metrics` rather than dedicated tools; adding empty-shell
"db-status"/"config" tools with no distinct backing data would just be
`get_logs`/`get_metrics` again under a different name.

No `get_incident` tool: an incident's `affected_service`/`detected_at`/
`severity` seed the Investigation node's starting window, but that's the
*graph* bootstrapping `IncidentState` from the `Incident` row directly
(a Phase 3 concern) before the ReAct loop starts — not something the LLM
should spend a tool call "discovering" about the very incident it was
handed. Out of scope for Phase 2.

## LangChain binding pattern

Each tool module (`logs.py`, `metrics.py`, `deployments.py`,
`dependencies.py`) defines two things:

1. A plain, directly-testable function whose first parameter is
   `db: Session` (e.g. `logs.get_logs(db, service, start, end, level)`) —
   call this directly in tests, no LangChain machinery involved.
2. A `make_get_<x>_tool(db: Session) -> BaseTool` factory that closes over
   `db` and returns a `@tool(parse_docstring=True)`-decorated inner
   function whose signature deliberately does *not* include `db` — only
   `service`/`start`/`end`/... are LLM-facing.

Why a closure factory rather than decorating the `db`-taking function
directly: `@tool` (`langchain_core.tools.tool`, installed `langchain-core`
1.6.0) infers the LLM-facing args schema from *every* parameter of the
decorated function via `infer_schema=True`. There's no built-in
"exclude this one positional param" switch for a plain sync tool
function, so a required `db: Session` argument would otherwise leak into
the tool's `args_schema` — the LLM would be asked to fill in a
`Session`, which is both wrong and unfillable, and would 400 on Claude's
side (no JSON schema for an opaque `Session` type). A factory bound per
request is also exactly the shape Phase 3 needs anyway: it builds one
`Session` per request/graph run and must hand each node a *set* of tools
already wired to that session (see `build_tools` below) — a global
session would break the "independently testable, session explicit"
requirement BUILD_PLAN.md calls for.

`parse_docstring=True` is what makes the schema *docstring*-derived
(BUILD_PLAN.md Phase 2: "a docstring-derived schema") rather than just
type-hint-derived: with it, `@tool` parses a Google-style `Args:` block
and attaches each arg's description to that field's JSON schema, and the
function's summary line becomes the tool's top-level `description` — both
of which `ChatAnthropic.bind_tools()` forwards into the Claude tool-use
schema.

## Return shape

Every tool returns a `list[...]` of a `backend.tools.schemas` Pydantic
model (`LogRecord`/`MetricRecord`/`DeploymentRecord`/`TraceRecord`), never
raw ORM rows or dicts. Every record carries its own row `id` — the
building block a later Phase 3+ RCA node needs to construct a
`source_ref` (tool name + record id), per BUILD_PLAN.md's evidence
citation requirement. This module does not build that `source_ref`/
evidence schema itself; that's explicitly Phase 3+'s job.
"""

from __future__ import annotations

from langchain_core.tools import BaseTool
from sqlalchemy.orm import Session

from backend.tools.dependencies import get_dependencies, make_get_dependencies_tool
from backend.tools.deployments import get_deployments, make_get_deployments_tool
from backend.tools.historical_incidents import (
    build_rag_tools,
    make_search_historical_incidents_tool,
    search_historical_incidents,
)
from backend.tools.logs import get_logs, make_get_logs_tool
from backend.tools.metrics import get_metrics, make_get_metrics_tool
from backend.tools.schemas import DeploymentRecord, LogRecord, MetricRecord, TraceRecord

__all__ = [
    "DeploymentRecord",
    "LogRecord",
    "MetricRecord",
    "TraceRecord",
    "build_rag_tools",
    "build_tools",
    "get_deployments",
    "get_dependencies",
    "get_logs",
    "get_metrics",
    "make_get_deployments_tool",
    "make_get_dependencies_tool",
    "make_get_logs_tool",
    "make_get_metrics_tool",
    "make_search_historical_incidents_tool",
    "search_historical_incidents",
]


def build_tools(db: Session) -> list[BaseTool]:
    """Return every investigation tool bound to one request-scoped `db`.

    Convenience for Phase 3's ReAct loop (`ChatAnthropic.bind_tools(...)`
    / `ToolNode(...)`), which needs the full set wired to the same session
    in one call rather than importing and invoking each factory
    individually.
    """
    return [
        make_get_logs_tool(db),
        make_get_metrics_tool(db),
        make_get_deployments_tool(db),
        make_get_dependencies_tool(db),
    ]
