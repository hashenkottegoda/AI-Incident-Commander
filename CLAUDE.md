# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Agentic system for investigating simulated production incidents: LangGraph-orchestrated
investigation, evidence-backed root-cause analysis, Qdrant-based historical incident RAG,
human-in-the-loop gated remediation, and a 4-way (A/B/C/D) architecture evaluation harness.
Full design rationale, phased build order, and locked-in decisions live in `BUILD_PLAN.md` —
read it before making any architectural change; it is the source of truth for *why* things are
built the way they are, not just *what* to build next. As of the latest commits, Phases 0–8 are
complete (backend, evaluation framework, React dashboard); only Phase 9 (optional stretch: MCP
wrapper, OTel tracing, CI hardening) remains.

## Commands

Backend (Python 3.12, `uv`):
```bash
uv sync --frozen                    # install deps exactly as locked (matches CI)
uv run ruff check .                 # lint
uv run pytest                       # full test suite
uv run pytest tests/test_graph.py   # single test file
uv run pytest tests/test_graph.py::test_name -v   # single test
docker compose up -d                # postgres + qdrant + backend (dev infra)
uv run alembic upgrade head         # apply DB migrations
uv run python -m backend.evaluation.run_experiments   # A/B/C/D eval harness (see below)
```

Frontend (`frontend/`, Node 22):
```bash
npm ci                              # install (matches CI)
npm run dev                         # vite dev server
npm run lint                        # oxlint
npm run build                       # tsc -b && vite build (type-check + bundle)
```

CI (`.github/workflows/ci.yml`) runs two jobs on every push/PR to `main`: `uv sync --frozen` +
`ruff check` + `pytest` against a real Postgres service container, and `npm ci` + `oxlint` +
`tsc -b && vite build` in `frontend/`.

Environment: copy `.env.example` to `.env`. `OPENROUTER_API_KEY` is required (no default — the
app fails fast at startup if missing). Per-role model IDs (`TRIAGE_MODEL`, `INVESTIGATION_MODEL`,
`RCA_MODEL`, `EMBEDDING_MODEL`) are read from env via `backend/config.py`'s `get_settings()` —
**never hard-code a model ID in code**; swapping models must be a `.env` change. No temperature/
top_p override is set anywhere — behavior is steered by prompting only, kept consistent across
whichever OpenRouter model a role is pointed at. Tests set a dummy
`OPENROUTER_API_KEY` in `tests/conftest.py` before any module import, and clear the
`get_settings()` lru_cache around every test.

## Architecture

### LLM integration: one path only

`langchain-openrouter` (`ChatOpenRouter`) is the **single** integration path, bound to
LangGraph/`ToolNode`. Do not mix raw `openrouter` SDK calls with LangChain calls anywhere — one
message/tool-serialization path, one place to reason about caching and retries. Structured
outputs (every node that returns JSON) go through LangChain's structured-output binding, not
hand-parsed free text.

### The graph (`backend/graph.py`)

`build_incident_graph(db, qdrant_client)` assembles the full `StateGraph` (uncompiled — callers
attach a checkpointer). Node functions live in `backend/agents/*_node.py`, one file per node,
each a factory (`make_x_node(db)`) closing over a request-scoped `Session`/Qdrant client rather
than using global state.

```
START -> triage -> investigation -> rag -> root_cause -> response_planner
              ^                        |
              +---- reinvestigate -----+   (confidence-gap OR evidence-insufficiency, bounded)
response_planner -> human_approval (HIGH_IMPACT actions) -> action_executor
response_planner -> action_executor (SAFE-only plan, skips human_approval)
action_executor -> END (SAFE-only, nothing to verify: incident_status = DIAGNOSED)
action_executor -> recovery_check (a HIGH_IMPACT remediation just ran)
recovery_check -> END (RESOLVED)
recovery_check -> investigation (still degraded, re-investigation budget remains)
recovery_check -> END (MANUAL_INTERVENTION_REQUIRED, budget exhausted)
```

- **Risk Classifier is a deterministic code-level rule table, never an LLM decision.** SAFE
  actions (report/note/tag/gather-diagnostics) auto-execute; HIGH-IMPACT actions
  (rollback/restart/scale/config/disable) are gated behind human approval — default-deny.
- **`human_approval_node` uses LangGraph's `interrupt()`, resumed via `POST /approve` with
  `Command(resume=...)`.** LangGraph re-executes an interrupted node from its start on resume,
  so `interrupt()` must be side-effect safe: nothing irreversible happens before the interrupt
  call. `action_executor` runs strictly after resume, never before. The approval flow is
  idempotent — a duplicate `/approve` re-enters the node but guards on
  `incident_status`/`execution_result_id` so remediation executes at most once. `POST /reject`
  never resumes the graph at all (`backend/api/approvals.py`).
- **Every node is wrapped in `_with_progress`** (in `graph.py`, applied centrally at
  `add_node()` time, not inside each node module) which writes a `NodeProgressEvent` row before
  the node runs — this is the live investigation trace the dashboard polls
  (`GET /incidents/{id}/progress`). A resumed interrupted node legitimately writes a second
  progress row for the same node name; that's accurate, not a bug.
- **`run_incident_graph_to_diagnosis`** compiles the *same* graph with
  `interrupt_before=["response_planner"]` (a static breakpoint, not `interrupt_after`) so the
  Phase 7 eval harness can score diagnosis immediately after RCA without response-planning's
  extra LLM call/latency/tokens leaking into the comparison. Uses a distinct thread_id suffix
  (`-diagnostic-eval`) so it never collides with the real operational thread for the same
  incident.
- **`IncidentState` (Pydantic, `backend/agents/state.py`) holds references + compact reasoning
  state, not bulk data.** Raw logs/metrics/tool results live in Postgres keyed by id; state
  carries ids/refs plus distilled evidence — keeps LangGraph Postgres checkpoints small.
- **`diagnostic_confidence` is a model-reported heuristic, not a calibrated probability** — used
  only as a secondary/tie-break signal and display value. The re-investigation loop trigger is a
  disjunction: confidence gap between top-2 hypotheses below threshold **OR** an
  evidence-sufficiency check fails — never confidence alone (LLM confidence is poorly
  calibrated). Bounded by `routing.MAX_REINVESTIGATION_LOOPS`, shared by both the root-cause loop
  and the post-recovery-check loop.
- **`incident_status` lifecycle:** `detected -> triaging -> investigating -> diagnosed ->
  awaiting_approval -> executing -> verifying -> resolved | manual_intervention_required`
  (verifying can loop back to investigating). `manual_intervention_required`, not `unresolved` —
  a rejected remediation or failed recovery means the incident is still live and needs a human,
  never a silent dead-end.

### Tools (`backend/tools/`)

Each tool is a typed Python function with a docstring-derived schema bound to the model's tool-use
format via LangChain (logs, metrics, deployments, dependencies, historical incidents via RAG). Unit-tested
directly against seeded data, independent of any agent.

### RAG (`backend/rag/`)

Qdrant + local `sentence-transformers` embeddings by default (keeps `docker compose up && uv run
...` fully reproducible with zero extra API keys); Voyage AI is documented as an optional
production-grade swap, not an MVP dependency. The retrieval query is a **structured incident
summary** (service, symptoms, recent_changes, observed_dependencies, timeline) assembled then
embedded — not a raw evidence dump — so "why was this historical incident considered similar" is
explainable rather than a black-box cosine score.

### Simulation (`backend/simulation/`)

Synthetic baseline telemetry for the checkout/payment/inventory services, plus a failure
injection engine that reads a `failure_scenarios/*.yaml` ground-truth file and writes a
temporally coherent anomaly timeline into Postgres (deployment at T, connections rising at T+1,
latency at T+2, errors at T+3, incident at T+4 — never a single `error=true` flag). The injector
also supports `--count N --seed S` batch generation; the same seed must always reproduce a
byte-identical dataset since the Phase 7 eval compares experiments against the exact same
incidents.

### Failure scenario ground truth (`failure_scenarios/*.yaml`)

Six scenarios (`db_connection_exhaustion`, `memory_leak`, `bad_deployment`, `dependency_failure`,
`slow_query`, `cascading_payment_timeout`). Each stores an ordered `causal_chain` (not just the
final root cause — this is what makes `cascading_payment_timeout` evaluable: the eval checks
whether the agent traced back to the true payment-dependency root cause rather than stopping at
the louder DB-overload symptom) and `remediation_effects` (which remediation actually recovers
the incident, what "recovered" telemetry looks like, and which remediations are ineffective) —
this is what makes the Recovery Check's resolved/still-degraded decision and the
wrong-remediation eval metric deterministic rather than eyeballed.

### Evaluation framework (`backend/evaluation/`)

The project's key differentiator — treat it as core, not an afterthought. Four experiments
against the **same seeded dataset**, all producing the identical `DiagnosisResult` schema so
scoring code is written once:
- **A** — context-stuffing baseline (all telemetry dumped into one prompt, no tools)
- **B** — tool-using single agent (`backend/agents/investigator.py`)
- **C** — tools + RAG
- **D** — full orchestrated graph (Triage → Investigation → RAG → RCA → Response)

Diagnostic metrics (all four, scored immediately after RCA, before response planning) are
deterministic set operations against structured output: root-cause accuracy (enum equality),
evidence precision / hallucination rate (cited `source_ref`s checked against the real tool-call
log), tool-call efficiency, latency, token cost. Operational metrics (D only, via
`run_experiment_d_operational`) measure the closed remediation loop: remediation success rate,
recovery-verification accuracy, wrong-remediation rate. Run via
`uv run python -m backend.evaluation.run_experiments`; results served at
`GET /api/evaluation/results`. A full 4×100 run is thousands of OpenRouter calls against
rate-limited free-tier quota — use smaller `--count` subsets for development, not the full benchmark.

### Data model (`backend/models/`)

SQLAlchemy models: `Service`, `Deployment`, `LogEntry`, `MetricPoint`, `TraceLite`, `Incident`,
`AuditEvent` (every approval decision + executed action, including approver identity — a
stubbed/header-supplied user, since full RBAC is deliberately out of scope), `NodeProgressEvent`
(live investigation trace). LangGraph's Postgres checkpoint tables live in the same database
instance, provisioned once via `backend/scripts/setup_checkpointer.py` (not run on every request
or wired into app startup — same reasoning as `alembic upgrade head` not running automatically).

### Frontend (`frontend/`)

React 19 + TypeScript + Tailwind 4 + React Router, built with Vite. Incident list, incident
detail (evidence, root cause/confidence, approve/reject, live investigation trace polled from
`GET /incidents/{id}/progress`), and an evaluation results page. `oxlint` for linting (not
eslint).

## Repository structure

```
backend/
  api/            FastAPI routers: incidents, simulation, evaluation, approvals, health
  agents/         LangGraph node functions, state, routing, schemas
  graph.py        StateGraph assembly — start here to understand orchestration
  tools/          Model tool-bound functions (logs, metrics, deployments, dependencies, RAG)
  models/         SQLAlchemy models
  rag/            Qdrant client, embedding pipeline, historical incident seed data
  simulation/     synthetic telemetry generator + failure injection engine
  evaluation/     eval dataset harness, scoring functions, A/B/C/D experiment runner
  scripts/        one-off ops scripts (dataset generation, checkpointer setup, seeding)
failure_scenarios/   ground-truth YAML per scenario (read by simulation + eval)
historical_incidents/  seed data for the RAG layer
evaluation/         generated datasets + experiment result output (gitignored contents)
frontend/          React dashboard
tests/              pytest suite, mirrors backend/ modules
alembic/            DB migrations
```

## Working within this repo

- Follow `BUILD_PLAN.md`'s Execution Discipline: identify the smallest logical task within a
  phase before writing code, don't touch files unrelated to that task, and don't start a new
  phase until the current one has passing tests and is committed.
- If a step exposes an architectural problem, stop and resolve it rather than papering over it.
- Specialized subagents exist per subsystem (`backend-engineer`, `langgraph-agent-engineer`,
  `rag-engineer`, `simulation-engineer`, `eval-engineer`, `frontend-engineer`,
  `devops-engineer`) — see `.claude/agents/`; each is scoped to the directories described above.
