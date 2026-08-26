# AI Incident Commander

An agentic system that investigates simulated production incidents end to end: a
LangGraph-orchestrated multi-step investigation over synthetic telemetry, evidence-backed
root-cause analysis grounded in real tool-call citations, RAG retrieval over historical incident
writeups, and human-in-the-loop gated remediation with closed-loop recovery verification. Built
alongside a quantitative evaluation harness that compares four investigation architectures
(context-stuffing, tool-using agent, tools + RAG, and the full orchestrated graph) on root-cause
accuracy, evidence precision, hallucination rate, tool-call efficiency, latency, and token cost.

## Why this exists

Most "AI agent" demos show a model producing a plausible-sounding answer with no way to check if
it's actually right. This project is built around the opposite bet: every diagnosis is scored
against ground truth, every cited piece of evidence must trace back to a real tool call or it
counts as a hallucination, and nothing destructive ever executes without a human approving it
first. The evaluation framework — not just the agent — is the deliverable.

## How it works

```
INCIDENT
   │
   ▼
TRIAGE (cheap/fast model, no tools)
   │
   ▼
INVESTIGATION (ReAct tool loop: logs, metrics, deployments, dependencies)
   │
   ▼
RAG (structured incident summary → embed → search historical incidents)
   │
   ▼
ROOT CAUSE (structured output: ranked hypotheses + cited evidence)
   │
   ├── confidence gap too small, or evidence incomplete ──► back to INVESTIGATION (bounded)
   │
   ▼
RESPONSE PLANNER → RISK CLASSIFIER (deterministic, never an LLM decision)
   │
   ├── SAFE actions ──────────────────────────────► ACTION EXECUTOR
   └── HIGH-IMPACT actions ──► HUMAN APPROVAL ──► ACTION EXECUTOR
                                (approve/reject)         │
                                                          ▼
                                                  RECOVERY CHECK
                                            (compares post-action telemetry
                                             against the pre-incident baseline)
                                                          │
                                        ┌─────────────────┴─────────────────┐
                                    RESOLVED                        still degraded
                                                                             │
                                                              back to INVESTIGATION
                                                          (bounded; exhausted → manual
                                                              intervention required)
```

A few things that don't show up in the diagram:

- **Every citation is checkable.** Evidence is never free text — each item carries a
  `source_ref` (the tool call it came from), so the evaluation harness can verify a claimed fact
  against the actual tool-call log rather than trusting the model's word for it.
- **The risk classifier is a plain code-level rule table, never an LLM judgment call.** Rollbacks,
  restarts, scaling, and config changes are always gated behind a human approval `interrupt()`;
  the model never decides for itself that something is safe enough to skip review.
- **The re-investigation loop doesn't trust confidence scores alone.** Self-reported LLM
  confidence is a poorly-calibrated heuristic, so the loop also fires on an evidence-sufficiency
  check (did the investigation actually cover the affected service's recent deployments and
  downstream dependencies?) — this is what catches a cascading failure where the loudest symptom
  isn't the root cause.
- **A rejected remediation or a fix that doesn't stick never dead-ends silently** — it routes to
  an explicit `manual_intervention_required` state, not a generic "unresolved."

## Evaluation

Four architectures are scored against the same seeded synthetic incident dataset, all producing
the same output schema so the comparison isolates architecture, not data access:

| Experiment | Data access |
|---|---|
| A | All relevant telemetry dumped into one prompt (no tools) |
| B | Tool-using agent, selective retrieval |
| C | Tools + RAG over historical incidents |
| D | Full orchestrated graph (triage → investigation → RAG → RCA → response) |

Diagnostic metrics (root-cause accuracy, evidence precision, hallucination rate, tool-call
efficiency, latency, token cost) are scored immediately after root-cause analysis for all four.
Operational metrics (remediation success rate, recovery-verification accuracy, wrong-remediation
rate) are scored for D's full closed loop only. Run it yourself:

```bash
uv run python -m backend.evaluation.run_experiments --count 5 --seed 42
```

## Tech stack

Python 3.12 / FastAPI / Pydantic v2 · LangGraph (`StateGraph`, `interrupt()`, Postgres
checkpointer) · `langchain-anthropic` (Claude) · PostgreSQL / SQLAlchemy · Qdrant + local
`sentence-transformers` embeddings · React 19 / TypeScript / Tailwind 4 dashboard · pytest / ruff
/ oxlint · Docker Compose · GitHub Actions CI.

## Running it

```bash
cp .env.example .env    # fill in ANTHROPIC_API_KEY
docker compose up -d    # postgres + qdrant + backend
uv run alembic upgrade head

# inject a failure and watch the dashboard
curl -X POST localhost:8000/api/simulation/failure \
  -H "Content-Type: application/json" -d '{"failure_type": "db_connection_exhaustion"}'

cd frontend && npm ci && npm run dev   # http://localhost:5173
```

See `frontend/README.md` for dashboard-specific commands. Backend tests: `uv run pytest`.

## Notes

The failure scenarios, historical incident corpus, and simulated telemetry are all synthetic —
nothing here talks to a real production system, and the "remediation" the Action Executor
performs only ever writes synthetic post-action telemetry, never touches anything real.
