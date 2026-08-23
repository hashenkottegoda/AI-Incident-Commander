"""initial empty revision

No-op baseline revision. Phase 0 has no SQLAlchemy models yet (those land
in Phase 1 under backend/models/, subclassing backend.db.Base) — this
revision exists only to prove the Alembic pipeline runs cleanly
end-to-end against the live database (creates the `alembic_version` table)
before there's anything real to autogenerate. Phase 1's first
`alembic revision --autogenerate` will chain off this one.

Revision ID: 332b6130d461
Revises:
Create Date: 2026-08-24 01:03:36.215413

"""

from collections.abc import Sequence

# revision identifiers, used by Alembic.
revision: str = "332b6130d461"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """No-op: nothing to create yet."""


def downgrade() -> None:
    """No-op: nothing to drop."""
