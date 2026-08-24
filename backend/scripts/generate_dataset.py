"""Generate a seeded batch of failure-scenario incidents for the eval dataset.

BUILD_PLAN.md Phase 1's final simulation sub-step: `--count N --seed S`
generation mode so the eval dataset scales from ~30 (dev) to 100+
(portfolio) "without new design work." `--seed` is mandatory (no silent
default) — per BUILD_PLAN.md, "the seed is mandatory for a fair eval":
`--count 100 --seed 42` must always produce the identical 100 scenarios
so experiments A/B/C/D are scored against the exact same incidents.

Thin CLI wrapper around `backend.simulation.dataset.generate_dataset`,
following `setup_checkpointer.py`'s one-off-script pattern (own
`argparse` entry point, `logging` for output, `if __name__ == "__main__"`).

Usage:

    uv run python -m backend.scripts.generate_dataset --count 30 --seed 42

`--count` defaults to 30 (BUILD_PLAN.md's "~30 (dev)" dataset size);
`--seed` has no default and argparse errors without it.
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter

from backend.db import SessionLocal
from backend.simulation.dataset import generate_dataset

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a seeded batch of failure-scenario incidents "
            "(BUILD_PLAN.md Phase 1's eval-dataset generator)."
        )
    )
    parser.add_argument(
        "--count",
        type=int,
        default=30,
        help="Number of incidents to generate (default: 30, the 'dev' dataset size).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help=(
            "Mandatory RNG seed. Required for a fair eval: --count N --seed S must "
            "always reproduce the identical N scenarios."
        ),
    )
    return parser.parse_args(argv)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()

    db = SessionLocal()
    try:
        incidents = generate_dataset(db, count=args.count, seed=args.seed)
        db.commit()
        # Read attributes while the session is still open: `commit()`
        # expires instances by default (SQLAlchemy's `expire_on_commit`),
        # and they'd raise `DetachedInstanceError` once `db.close()` runs.
        failure_types = [incident.failure_type for incident in incidents]
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

    breakdown = Counter(failure_types)
    logger.info(
        "Generated %d incidents (seed=%d, count=%d).", len(incidents), args.seed, args.count
    )
    for failure_type, n in sorted(breakdown.items()):
        logger.info("  %-28s %d", failure_type, n)


if __name__ == "__main__":
    main()
