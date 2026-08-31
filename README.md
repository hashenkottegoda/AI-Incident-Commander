# 🚨 AI Incident Commander

**An AI that gets paged at 3am, investigates a production outage, finds the root cause, and asks a human before it touches anything.**

Then it proves it was right — because every diagnosis is scored against ground truth, and every "fact" it cites has to trace back to a real tool call or it's counted as a lie.

> The interesting part isn't that an agent *can* diagnose an incident. It's building the harness that tells you **whether it actually did** — and by how much better than a naive approach.

---

## The 30-second version

A fake production outage happens (a database runs out of connections, a deploy goes bad, a dependency times out). An LLM-powered on-call engineer wakes up and:

1. 🔍 **Investigates** — pulls logs, metrics, deploys, and service dependencies with real tools (not one giant prompt-stuffed context)
2. 🧠 **Remembers** — searches a library of past incidents to see if it's seen this before
3. 🎯 **Diagnoses** — ranks root-cause hypotheses, each backed by citations you can actually check
4. ✋ **Asks permission** — anything risky (rollback, restart, scale) is *hard-blocked* behind a human click
5. ✅ **Verifies the fix** — checks whether the outage actually cleared, and re-investigates if it didn't

And the whole time, a **scoreboard** is running: it pits four different agent designs against the same incidents and measures which one is actually more accurate, more grounded, and cheaper.

---

## Why it's built this way (the opinionated bits)

Most "AI agent" demos hand you a confident paragraph and no way to know if it's nonsense. This project is the opposite bet:

| The problem with typical demos | What this does instead |
|---|---|
| "Trust me, that's the root cause" | **Every citation is checkable** — each evidence item carries a `source_ref` pointing at the exact tool call. Made-up facts show up as a measured *hallucination rate*. |
| The model decides what's safe to run | **A dumb, deterministic rule table decides** — never the LLM. Rollbacks/restarts/scaling always stop for a human. Default-deny. |
| Confidence score = "I'm 90% sure!" | LLM confidence is famously miscalibrated, so the re-investigation loop **doesn't trust it alone** — it also checks *did we actually look at the recent deploys and downstream dependencies?* This is what catches a cascading failure where the loudest symptom isn't the cause. |
| "Couldn't fix it 🤷" | A rejected or failed fix routes to an explicit **`manual_intervention_required`** state — never a silent dead-end. |

---

## How the agent thinks

```
                        🚨 INCIDENT
                             │
                             ▼
                   TRIAGE  (fast, cheap, no tools)
                             │
                             ▼
              INVESTIGATION  (ReAct tool loop:
                             logs · metrics · deploys · dependencies)
                             │
                             ▼
                        RAG  (summarize → embed → search past incidents)
                             │
                             ▼
                  ROOT CAUSE  (ranked hypotheses + cited evidence)
                             │
        confidence gap too small, OR evidence incomplete?
                     └──────────────► back to INVESTIGATION  (bounded retries)
                             │
                             ▼
             RESPONSE PLANNER → RISK CLASSIFIER  (deterministic, never the LLM)
                             │
              ┌─────────────┴─────────────┐
          SAFE action                HIGH-IMPACT action
              │                            │
              │                      ✋ HUMAN APPROVAL  (approve / reject)
              │                            │
              └────────────┬───────────────┘
                           ▼
                    ACTION EXECUTOR
                           │
                           ▼
                   RECOVERY CHECK  (post-fix telemetry vs. pre-incident baseline)
                           │
              ┌────────────┴────────────┐
          ✅ RESOLVED            still degraded → re-investigate
                                  (budget exhausted → 🚑 manual intervention)
```

Built on **LangGraph** — the human approval step is a real `interrupt()` that pauses the graph mid-flight and resumes on a `POST /approve`. Nothing irreversible happens before that pause.

---

## The actual point: the scoreboard 📊

Four architectures, **same incidents, same output format**, so the comparison measures *the design* — not who got luckier data:

| | Architecture | The question it answers |
|---|---|---|
| **A** | Stuff all telemetry into one prompt | Do tools even help? |
| **B** | Tool-using agent | Does selective retrieval beat brute force? |
| **C** | Tools **+ RAG** | Does remembering past incidents help? |
| **D** | Full orchestrated graph | Is the whole multi-agent dance worth it? |

Every run is scored on hard numbers — no vibes:

- 🎯 **Root-cause accuracy** (vs. known ground truth)
- 🔬 **Evidence precision & hallucination rate** (citations checked against the real tool-call log)
- ⚡ **Tool-call efficiency · latency · token cost**
- 🔧 **Remediation success & recovery-verification accuracy** (for the full loop)

```bash
uv run python -m backend.evaluation.run_experiments --count 5 --seed 42
```

The same seed always reproduces the exact same incidents — so the comparison is apples-to-apples every time.

---

## Run it yourself

```bash
cp .env.example .env          # add your OPENROUTER_API_KEY
docker compose up -d          # postgres + qdrant + backend
uv run alembic upgrade head

# 💥 inject an outage and watch the dashboard light up
curl -X POST localhost:8000/api/simulation/failure \
  -H "Content-Type: application/json" \
  -d '{"failure_type": "db_connection_exhaustion"}'

cd frontend && npm ci && npm run dev    # → http://localhost:5173
```

Backend tests: `uv run pytest` · dashboard commands: see `frontend/README.md`.

---

## Built with

**Python 3.12** · FastAPI · Pydantic v2 · **LangGraph** (`StateGraph`, `interrupt()`, Postgres checkpointer) · `langchain-openrouter` (**OpenRouter**, free-tier models) · PostgreSQL / SQLAlchemy · **Qdrant** + local `sentence-transformers` embeddings · **React 19** / TypeScript / Tailwind 4 · pytest · ruff · oxlint · Docker Compose · GitHub Actions CI

---

## One honest disclaimer

Everything here is **synthetic** — the outages, the telemetry, the historical incident library, even the "fixes." Nothing touches a real production system; the Action Executor only ever writes simulated post-fix telemetry. The point was to build the *investigation and evaluation machinery* rigorously, not to wire it to a live datacenter. 🙂
