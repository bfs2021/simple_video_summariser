"""Anthropic batch checkpoints for a job (issue #89)

Revision ID: 0013
Revises: 0012
Create Date: 2026-10-12

Adds ``llm_batches``: one row per (job, step) with the Message Batch that step
submitted, so a retried analyze job resumes that batch instead of submitting
and paying for the same request again. Rows go when the job's analysis is
saved, or with the job itself (``ON DELETE CASCADE``).

Raw SQL through ``op.execute`` (convention set by #7). Runs only as the
standalone ``migrate`` command.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0013"
down_revision: str | None = "0012"
branch_labels: Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE llm_batches (
            job_id         BIGINT NOT NULL REFERENCES jobs (id) ON DELETE CASCADE,
            step_key       TEXT NOT NULL,
            batch_id       TEXT NOT NULL,
            request_sha256 TEXT NOT NULL,
            submitted_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
            UNIQUE (job_id, step_key)
        )
        """
    )


def downgrade() -> None:
    op.execute("DROP TABLE llm_batches")
