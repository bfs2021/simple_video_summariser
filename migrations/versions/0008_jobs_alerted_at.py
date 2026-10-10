"""when a dead-lettered job was announced to the operator

Revision ID: 0008
Revises: 0007
Create Date: 2026-10-10

Adds ``jobs.alerted_at`` (issue #67): set by the planner when a
``TOOL_FAILURE`` / ``RESOURCE`` dead-letter has been included in an operator
alert. ``NULL`` means not announced. A dead job whose ``finished_at`` is later
than ``alerted_at`` (retried from the Ops view, then dead again) counts as not
announced.

Backfill: every job already dead is marked announced (``finished_at``, else
now), so deploying this does not send an alert for old history.

Raw SQL through ``op.execute`` (convention set by #7).
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0008"
down_revision: str | None = "0007"
branch_labels: Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE jobs ADD COLUMN alerted_at TIMESTAMPTZ NULL")
    op.execute("UPDATE jobs SET alerted_at = COALESCE(finished_at, now()) WHERE state = 'dead'")


def downgrade() -> None:
    op.execute("ALTER TABLE jobs DROP COLUMN alerted_at")
