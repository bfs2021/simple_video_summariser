"""Durable cumulative planner task counts (issue #138)

Revision ID: 0014
Revises: 0013
Create Date: 2026-10-22

Adds ``planner_task_runs.cumulative_count`` for counts accumulated from
successful completed planner runs. Existing rows begin at zero.

Raw SQL through ``op.execute`` (convention set by #7). Runs only as the
standalone ``migrate`` command.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0014"
down_revision: str | None = "0013"
branch_labels: Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE planner_task_runs "
        "ADD COLUMN cumulative_count BIGINT NOT NULL DEFAULT 0"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE planner_task_runs DROP COLUMN cumulative_count")
