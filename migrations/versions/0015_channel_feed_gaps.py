from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0015"
down_revision: str | None = "0014"
branch_labels: Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("ALTER TABLE channels ADD COLUMN feed_gap_detected_at TIMESTAMPTZ NULL")
    op.execute("ALTER TABLE channels ADD COLUMN feed_gap_reason TEXT NULL")
    op.execute(
        "ALTER TABLE channels ADD CONSTRAINT channels_feed_gap_reason_check"
        " CHECK (feed_gap_reason IS NULL OR feed_gap_reason IN"
        " ('suspected', 'no_anchor', 'listing_window_exhausted'))"
    )
    op.execute(
        "ALTER TABLE channels ADD CONSTRAINT channels_feed_gap_pair_check"
        " CHECK ((feed_gap_detected_at IS NULL) = (feed_gap_reason IS NULL))"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE channels DROP CONSTRAINT channels_feed_gap_pair_check")
    op.execute("ALTER TABLE channels DROP CONSTRAINT channels_feed_gap_reason_check")
    op.execute("ALTER TABLE channels DROP COLUMN feed_gap_reason")
    op.execute("ALTER TABLE channels DROP COLUMN feed_gap_detected_at")
