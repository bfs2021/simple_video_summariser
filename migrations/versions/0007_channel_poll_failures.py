"""how long a channel's RSS feed has been failing

Revision ID: 0007
Revises: 0006
Create Date: 2026-10-10

Adds ``channels.poll_failing_since`` (start of the current run of failed RSS
polls; ``NULL`` when the last poll succeeded or the channel was never polled)
and ``channels.poll_failures`` (failed polls in a row), issue #90. The two
``CHECK`` constraints keep them consistent: ``poll_failures >= 0`` and
``(poll_failures = 0) = (poll_failing_since IS NULL)``.

Backfill: a row with ``last_poll_err IS NOT NULL`` gets
``poll_failing_since = last_polled`` and ``poll_failures = 1``. These values
are a LOWER BOUND: the real start of the run and the real number of failures
are unknown, only the latest outcome was stored before this migration. Every
other row gets ``NULL`` / ``0``.

Raw SQL through ``op.execute`` (convention set by #7).
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE channels ADD COLUMN poll_failing_since TIMESTAMPTZ NULL")
    op.execute("ALTER TABLE channels ADD COLUMN poll_failures INTEGER NOT NULL DEFAULT 0")
    # Lower bound: the real start of the failure run is unknown (see docstring).
    # last_polled is set whenever last_poll_err is, but coalesce keeps the
    # consistency CHECK safe for a hand-edited row.
    op.execute(
        """
        UPDATE channels
        SET poll_failing_since = coalesce(last_polled, now()), poll_failures = 1
        WHERE last_poll_err IS NOT NULL
        """
    )
    op.execute(
        "ALTER TABLE channels ADD CONSTRAINT channels_poll_failures_nonneg"
        " CHECK (poll_failures >= 0)"
    )
    op.execute(
        "ALTER TABLE channels ADD CONSTRAINT channels_poll_failures_consistent"
        " CHECK ((poll_failures = 0) = (poll_failing_since IS NULL))"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE channels DROP CONSTRAINT channels_poll_failures_consistent")
    op.execute("ALTER TABLE channels DROP CONSTRAINT channels_poll_failures_nonneg")
    op.execute("ALTER TABLE channels DROP COLUMN poll_failures")
    op.execute("ALTER TABLE channels DROP COLUMN poll_failing_since")
