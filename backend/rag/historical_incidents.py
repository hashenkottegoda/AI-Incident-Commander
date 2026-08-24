"""Pydantic schema + loader for `historical_incidents/historical_incidents.yaml`.

Mirrors `backend.simulation.scenario_schema`'s role for
`failure_scenarios/*.yaml`: this module only parses/validates YAML into a
typed shape, it never touches Qdrant or Postgres itself, so it (and its
tests) run with zero infrastructure.

## Why one YAML file with a list, not 20 separate files

`failure_scenarios/*.yaml` is one-file-per-scenario because each scenario
is substantial config consumed by the injector (causal chains, remediation
effects) and there are only 6 of them. Here there are ~20 short, uniformly-
shaped records with no cross-references between them -- a single list-of-
records file is the more honest shape for "20 rows of the same structured
type" (easier to eyeball the whole seed set / category distribution at a
glance, easier to keep genuinely varied rather than 20 near-duplicate
files), while `HistoricalIncident` below still keeps every record fully
typed and validated, so this isn't a step down from "structured, not raw
prose blobs" -- it's the same structuring, just batched into one file.

## Embedding scope: structured fields only, not `narrative`

Per `backend.rag.schemas.IncidentSummary`, only `service`/`symptoms`/
`recent_changes`/`observed_dependencies`/`timeline` get embedded
(`HistoricalIncident.summary()` below extracts exactly that subset).
`narrative` (the human-readable resolution note) is deliberately *not*
embedded. Two reasons: (1) BUILD_PLAN.md's Agent Architecture section
specifies the retrieval query is assembled from those five structured
fields, and a live investigation's query and a seeded incident's write-up
must be embedded through the *same* field set for a similarity score to
mean anything comparable -- embedding `narrative` on one side only would
compare structured-summary-vs-symptoms against prose-vs-symptoms, an
apples-to-oranges score; (2) narrative text (resolution steps, hedging,
"the actual root cause was...") is exactly the kind of prose the idea doc
calls out as *conclusions*, not observed symptoms -- embedding it would let
retrieval match on how an incident was *resolved* rather than how it
*presented*, which is backwards for "does this look like something we've
seen before" at diagnosis time (before a resolution is even known).
`narrative` is still carried on every seeded point's payload (see
`backend.rag.schemas.HistoricalIncidentMatch`) so it's available to the
agent as context once a match is found.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from backend.rag.schemas import IncidentSummary

# The 6 real scenario `root_cause_category` values
# (`backend.agents.schemas.RootCauseCategory`, minus the "unknown" escape
# hatch -- every *historical* incident has a settled, known cause by
# definition, so "unknown" has no meaning here).
HistoricalRootCauseCategory = Literal[
    "database_connection_pool",
    "memory_resource_exhaustion",
    "application_bug",
    "upstream_dependency_failure",
    "inefficient_database_query",
    "upstream_dependency_timeout",
]

# Repo layout: backend/rag/historical_incidents.py -> backend/ -> repo root.
HISTORICAL_INCIDENTS_PATH: Path = (
    Path(__file__).resolve().parents[2] / "historical_incidents" / "historical_incidents.yaml"
)


class HistoricalIncident(BaseModel):
    """One hand-authored historical incident write-up.

    Structured-summary fields (`service`/`symptoms`/`recent_changes`/
    `observed_dependencies`/`timeline`) match `IncidentSummary` exactly --
    `summary()` below extracts them as one -- plus `id`/`root_cause_category`
    (the retrieval ground truth) and `narrative` (human-readable, not
    embedded; see this module's docstring).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    root_cause_category: HistoricalRootCauseCategory
    service: str
    symptoms: list[str] = Field(min_length=1)
    recent_changes: str | None = None
    observed_dependencies: str | None = None
    timeline: str
    narrative: str = Field(description="Human-readable resolution/root-cause note.")

    def summary(self) -> IncidentSummary:
        """This incident's structured-summary fields as an `IncidentSummary`
        -- the exact shape `backend.rag.embeddings.assemble_incident_text`
        embeds, and the exact shape a live query is built from."""
        return IncidentSummary(
            service=self.service,
            symptoms=self.symptoms,
            recent_changes=self.recent_changes,
            observed_dependencies=self.observed_dependencies,
            timeline=self.timeline,
        )


def load_historical_incidents(
    path: Path | str = HISTORICAL_INCIDENTS_PATH,
) -> list[HistoricalIncident]:
    """Read, parse, and validate every record in
    `historical_incidents/historical_incidents.yaml`."""
    with Path(path).open("r") as f:
        rows = yaml.safe_load(f)
    incidents = [HistoricalIncident.model_validate(row) for row in rows]

    ids = [incident.id for incident in incidents]
    duplicates = {i for i in ids if ids.count(i) > 1}
    if duplicates:
        raise ValueError(f"duplicate historical incident id(s): {sorted(duplicates)}")

    return incidents
