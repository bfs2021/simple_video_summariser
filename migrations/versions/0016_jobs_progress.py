"""live progress of a running transcribe job

Revision ID: 0016
Revises: 0015
Create Date: 2026-10-23

Adds ``jobs.progress`` (issue #245): a nullable JSONB written by the transcribe
worker every 30 seconds while the job runs (percent, elapsed, ETA, the up-front
estimate and ``updated_at``). ``NULL`` means no figures: every transition out of
``running`` clears it, so a finished or retried job never shows stale numbers.

Raw SQL through ``op.execute`` (convention set by #7).
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0016"
down_revision: str | None = "0015"
branch_labels: Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE jobs ADD COLUMN progress JSONB NULL")


def downgrade() -> None:
    op.execute("ALTER TABLE jobs DROP COLUMN progress")
