"""last run and outcome of each planner task

Revision ID: 0009
Revises: 0008
Create Date: 2026-10-10

Adds ``planner_task_runs`` (issue #98): one row per periodic planner task,
overwritten in place, so a stalled or failing planner is visible with one
query. Written only by the planner (``services/planner/main.py``).

Raw SQL through ``op.execute`` (convention set by #7).
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0009"
down_revision: str | None = "0008"
branch_labels: Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE planner_task_runs (
            task_name               TEXT PRIMARY KEY,
            interval_sec            DOUBLE PRECISION NOT NULL,
            last_started_at         TIMESTAMPTZ,
            last_finished_at        TIMESTAMPTZ,
            last_outcome            TEXT CHECK (last_outcome IN ('ok', 'failed')),
            last_error              TEXT,
            last_count              INTEGER,
            last_ok_at              TIMESTAMPTZ,
            last_skipped_locked_at  TIMESTAMPTZ
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE planner_task_runs")
