from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

# Import model modules so they register their tables on Base.metadata
# before autogenerate inspects it below. `backend.db` intentionally does
# not import this itself (it only owns engine/session/Base), so this
# import has to happen somewhere in the alembic process — here.
import backend.models  # noqa: E402,F401
from alembic import context

# Import the app's own settings/metadata rather than re-parsing .env or
# hardcoding a URL here — `backend.config.get_settings()` is the single
# source of truth for `database_url` (BUILD_PLAN.md Phase 0), and
# `backend.db.Base.metadata` is what Phase 1's models will register onto
# for `alembic revision --autogenerate` to pick up.
from backend.config import get_settings
from backend.db import Base

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Override whatever placeholder is in alembic.ini with the real app URL.
config.set_main_option("sqlalchemy.url", get_settings().database_url)

# Autogenerate support: backend/models/ registers its tables on this
# metadata via the `import backend.models` above.
target_metadata = Base.metadata


def include_name(name, type_, parent_names):
    """Exclude LangGraph's own checkpoint tables from autogenerate diffing.

    `checkpoints` / `checkpoint_blobs` / `checkpoint_writes` /
    `checkpoint_migrations` live in the same Postgres instance (see
    `backend/scripts/setup_checkpointer.py`) but are created and migrated
    by `langgraph-checkpoint-postgres`'s own setup, not by SQLAlchemy
    models/Alembic. Without this filter, autogenerate sees them as
    "extra tables in the DB not in our metadata" and proposes dropping
    them on every future `--autogenerate` run.
    """
    if type_ == "table" and name is not None and name.startswith("checkpoint"):
        return False
    return True


# NOTE: the enum-as-VARCHAR CHECK constraints on IncidentStatus/Severity/
# LogLevel (see backend/models/incident.py, backend/models/telemetry.py)
# don't round-trip cleanly through Alembic's autogenerate comparator — it
# can't reliably match the CHECK constraint text it renders from
# `Enum(native_enum=False)` metadata against the introspected DB constraint,
# so every future `alembic revision --autogenerate` will likely propose a
# spurious drop/recreate of `incident_status`, `incident_severity`, and
# `log_level`. This is a known SQLAlchemy/Alembic limitation, not a bug in
# this schema. Always review the CHECK-constraint section of any future
# autogenerate diff and strip these no-op operations before committing the
# migration.

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        include_name=include_name,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            include_name=include_name,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
