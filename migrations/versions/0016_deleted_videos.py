from __future__ import annotations

from collections.abc import Sequence

from alembic import op

revision: str = "0016"
down_revision: str | None = "0015"
branch_labels: Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        "CREATE TABLE deleted_videos ("
        "video_id TEXT PRIMARY KEY, "
        "deleted_at TIMESTAMPTZ NOT NULL DEFAULT now()"
        ")"
    )


def downgrade() -> None:
    op.execute("DROP TABLE deleted_videos")
