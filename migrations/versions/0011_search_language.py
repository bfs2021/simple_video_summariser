"""language-aware full-text search for transcripts

Revision ID: 0011
Revises: 0010
Create Date: 2026-10-11

Issue #112. ``transcripts.fts`` was fixed to the ``'english'`` text-search
configuration, so every non-English (Whisper) transcript was stemmed with
English rules. Now ``transcript_ts_config(language)`` maps the recorded ISO
639-1 code to a built-in Postgres configuration and the stored ``fts`` column
is generated with it: ``to_tsvector(transcript_ts_config(language), ...)``.
One column and one GIN index stay.

* ``transcript_ts_config(text) RETURNS regconfig`` is the single source of
  truth for the mapping (Snowball configs of ``postgres:18-alpine``;
  ``NULL``/empty/whitespace -> ``english``; anything else -> ``simple``).
* ``transcript_search_tsquery(text) RETURNS tsquery`` ORs
  ``websearch_to_tsquery(cfg, q)`` over every configuration the mapping can
  return, so the index condition does not depend on the row. The search then
  re-checks each candidate with its own configuration.

The mapping is ``IMMUTABLE`` and stored ``fts`` values depend on it. Changing
it later with ``CREATE OR REPLACE`` would leave existing rows stale: any
future change must be a new migration that replaces the function **and**
re-runs ``ALTER COLUMN fts SET EXPRESSION`` to rewrite the column (and
refreshes ``search_docs``).

``SET EXPRESSION`` (PostgreSQL 17+) rewrites ``transcripts`` and rebuilds
``transcripts_fts_idx`` under the same name. **The rewrite takes an
``ACCESS EXCLUSIVE`` lock on ``transcripts`` for its duration.** A rewrite
fires no row triggers, so ``search_docs`` (0010), which embeds the stored
vector, is refreshed for every video afterwards.

Raw SQL through ``op.execute`` (convention set by #7). Runs only as the
standalone ``migrate`` command.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0011"
down_revision: str | None = "0010"
branch_labels: Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

#: ISO 639-1 code -> built-in Snowball configuration.
_CODES = {
    "ar": "arabic", "hy": "armenian", "eu": "basque", "ca": "catalan", "da": "danish",
    "nl": "dutch", "en": "english", "et": "estonian", "fi": "finnish", "fr": "french",
    "de": "german", "el": "greek", "hi": "hindi", "hu": "hungarian", "id": "indonesian",
    "ga": "irish", "it": "italian", "lt": "lithuanian", "ne": "nepali", "no": "norwegian",
    "nb": "norwegian", "nn": "norwegian", "pt": "portuguese", "ro": "romanian",
    "ru": "russian", "sr": "serbian", "es": "spanish", "sv": "swedish", "ta": "tamil",
    "tr": "turkish", "yi": "yiddish",
}


def _config_function() -> str:
    arms = "\n".join(
        f"            WHEN '{code}' THEN 'pg_catalog.{cfg}'::regconfig"
        for code, cfg in _CODES.items()
    )
    return f"""
        CREATE FUNCTION transcript_ts_config(lang TEXT) RETURNS regconfig
        IMMUTABLE PARALLEL SAFE LANGUAGE sql AS
        $$
        -- IMMUTABLE and baked into transcripts.fts: changing this mapping needs a
        -- new migration that also re-runs ALTER COLUMN fts SET EXPRESSION.
        SELECT CASE split_part(
                   replace(lower(regexp_replace(coalesce(lang, ''), '^\\s+|\\s+$', '', 'g')),
                           '_', '-'), '-', 1)
            WHEN '' THEN 'pg_catalog.english'::regconfig
{arms}
            ELSE 'pg_catalog.simple'::regconfig
        END
        $$
        """


def _search_tsquery_function() -> str:
    configs = sorted({*_CODES.values(), "simple"})
    ored = "\n               || ".join(
        f"websearch_to_tsquery('pg_catalog.{cfg}'::regconfig, q)" for cfg in configs
    )
    return f"""
        CREATE FUNCTION transcript_search_tsquery(q TEXT) RETURNS tsquery
        IMMUTABLE PARALLEL SAFE LANGUAGE sql AS
        $$
        SELECT {ored}
        $$
        """


def upgrade() -> None:
    op.execute(_config_function())
    op.execute(_search_tsquery_function())
    op.execute(
        "ALTER TABLE transcripts ALTER COLUMN fts SET EXPRESSION AS "
        "(to_tsvector(transcript_ts_config(language), coalesce(full_text, '')))"
    )
    op.execute("SELECT refresh_search_doc(video_id) FROM videos")


def downgrade() -> None:
    op.execute(
        "ALTER TABLE transcripts ALTER COLUMN fts SET EXPRESSION AS "
        "(to_tsvector('english', coalesce(full_text, '')))"
    )
    op.execute("DROP FUNCTION transcript_search_tsquery(TEXT)")
    op.execute("DROP FUNCTION transcript_ts_config(TEXT)")
    op.execute("SELECT refresh_search_doc(video_id) FROM videos")
