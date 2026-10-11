"""backfill requests for the planner's channel_backfill task

Revision ID: 0012
Revises: 0011
Create Date: 2026-10-11

Adds ``backfill_requests`` (issue #115): the API records one request per
channel and the planner lists and enqueues in bounded slices, outside the HTTP
request. The partial unique index allows at most one ``pending`` or
``running`` request per channel, so two concurrent POSTs are decided by the
index, not by a read-then-insert.

Raw SQL through ``op.execute`` (convention set by #7). Runs only as the
standalone ``migrate`` command.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0012"
down_revision: str | None = "0011"
branch_labels: Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE backfill_requests (
            id          BIGSERIAL PRIMARY KEY,
            channel_id  TEXT NOT NULL REFERENCES channels (channel_id),
            limit_n     INTEGER NOT NULL CHECK (limit_n BETWEEN 1 AND 5000),
            dry_run     BOOLEAN NOT NULL,
            state       TEXT NOT NULL
                        CHECK (state IN ('pending', 'running', 'done', 'failed')),
            attempts    INTEGER NOT NULL DEFAULT 0,
            enqueued    INTEGER NOT NULL DEFAULT 0,
            result      JSONB,
            error_class TEXT,
            created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
            finished_at TIMESTAMPTZ
        )
        """
    )
    op.execute(
        """
        CREATE UNIQUE INDEX backfill_requests_active_uniq
            ON backfill_requests (channel_id)
            WHERE state IN ('pending', 'running')
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE backfill_requests")
