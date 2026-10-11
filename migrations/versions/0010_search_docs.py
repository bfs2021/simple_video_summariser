"""search_docs: one materialised search vector per video

Revision ID: 0010
Revises: 0009
Create Date: 2026-10-10

Issue #111. ``GET /search`` matches a video's title and its latest analysis
(TL;DR, topic titles and summaries, claim text) as well as its preferred
transcript. A generated column can only read its own row, and those texts
live in four other tables, so the combined vector is materialised once per
video in ``search_docs.fts`` and kept in sync by the database:

* ``refresh_search_doc(video_id)`` rebuilds and upserts one row. Weights:
  title A, TL;DR B, topics and claims C, preferred transcript D (the default
  weight). Title and analysis come first so their positions stay small.
* "Preferred transcript" is the lowest ``transcript_rank(source)``, ties by
  ``id`` (what ``get_best_transcript`` returns). "Latest analysis" is the
  newest by ``created_at DESC, id DESC`` (what ``latest_analysis`` returns).
  Neither rule is re-implemented in Python.
* Triggers on ``videos`` (INSERT, UPDATE OF title), ``transcripts`` (INSERT,
  UPDATE, DELETE) and ``analyses`` (INSERT, DELETE) call it. The ``analyses``
  one is a ``DEFERRABLE INITIALLY DEFERRED`` constraint trigger, so it runs
  at commit, after ``save_analysis`` has inserted topics and claims.
* The transcript part is the stored ``transcripts.fts`` of the preferred
  transcript, so it follows whatever text-search configuration that column
  uses (see 0011). Title and analysis text are always ``'english'``.

``tsquery_positive_lexemes(tsquery)`` lists the non-negated lexemes of a
query; the search uses it to report which sources matched.

Raw SQL through ``op.execute`` (convention set by #7). Runs only as the
standalone ``migrate`` command. The upgrade backfills a row for every video.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0010"
down_revision: str | None = "0009"
branch_labels: Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE search_docs (
            video_id   TEXT PRIMARY KEY REFERENCES videos(video_id) ON DELETE CASCADE,
            fts        tsvector NOT NULL,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
        """
    )
    op.execute("CREATE INDEX search_docs_fts_idx ON search_docs USING GIN (fts)")

    op.execute(
        """
        CREATE FUNCTION refresh_search_doc(vid TEXT) RETURNS void LANGUAGE plpgsql AS
        $$
        DECLARE
            a_id BIGINT;
        BEGIN
            SELECT id INTO a_id FROM analyses
            WHERE video_id = vid ORDER BY created_at DESC, id DESC LIMIT 1;

            INSERT INTO search_docs (video_id, fts, updated_at)
            SELECT v.video_id,
                   setweight(to_tsvector('english', coalesce(v.title, '')), 'A')
                   || setweight(to_tsvector('english', coalesce(
                          (SELECT tldr FROM analyses WHERE id = a_id), '')), 'B')
                   || setweight(to_tsvector('english', coalesce(
                          (SELECT string_agg(
                                      coalesce(title, '') || ' ' || coalesce(summary, ''),
                                      ' ' ORDER BY seq, id)
                           FROM topics WHERE analysis_id = a_id), '')), 'C')
                   || setweight(to_tsvector('english', coalesce(
                          (SELECT string_agg(text, ' ' ORDER BY id)
                           FROM claims WHERE analysis_id = a_id), '')), 'C')
                   || coalesce(
                          (SELECT t.fts FROM transcripts t WHERE t.video_id = vid
                           ORDER BY transcript_rank(t.source), t.id LIMIT 1),
                          to_tsvector('english', '')),
                   now()
            FROM videos v WHERE v.video_id = vid
            ON CONFLICT (video_id) DO UPDATE
                SET fts = EXCLUDED.fts, updated_at = EXCLUDED.updated_at;
        END
        $$
        """
    )

    op.execute(
        """
        CREATE FUNCTION search_docs_sync() RETURNS trigger LANGUAGE plpgsql AS
        $$
        BEGIN
            IF TG_OP = 'DELETE' THEN
                PERFORM refresh_search_doc(OLD.video_id);
            ELSE
                PERFORM refresh_search_doc(NEW.video_id);
            END IF;
            RETURN NULL;
        END
        $$
        """
    )
    op.execute(
        """
        CREATE TRIGGER videos_search_docs_trg
            AFTER INSERT OR UPDATE OF title ON videos
            FOR EACH ROW EXECUTE FUNCTION search_docs_sync()
        """
    )
    op.execute(
        """
        CREATE TRIGGER transcripts_search_docs_trg
            AFTER INSERT OR UPDATE OR DELETE ON transcripts
            FOR EACH ROW EXECUTE FUNCTION search_docs_sync()
        """
    )
    op.execute(
        """
        CREATE CONSTRAINT TRIGGER analyses_search_docs_trg
            AFTER INSERT OR DELETE ON analyses
            DEFERRABLE INITIALLY DEFERRED
            FOR EACH ROW EXECUTE FUNCTION search_docs_sync()
        """
    )

    op.execute(
        r"""
        CREATE FUNCTION tsquery_positive_lexemes(q tsquery) RETURNS text[]
        IMMUTABLE PARALLEL SAFE LANGUAGE sql AS
        $$
        SELECT coalesce(
            array_agg(replace(replace(m[1], '''''', ''''), '\\', '\')), ARRAY[]::text[])
        FROM regexp_matches(
            regexp_replace(q::text, '!\s*(''(?:[^'']|'''')*''|\([^()]*\))', '', 'g'),
            '''((?:[^'']|'''')*)''', 'g') AS m
        $$
        """
    )

    op.execute("SELECT refresh_search_doc(video_id) FROM videos")


def downgrade() -> None:
    op.execute("DROP FUNCTION tsquery_positive_lexemes(tsquery)")
    op.execute("DROP TRIGGER analyses_search_docs_trg ON analyses")
    op.execute("DROP TRIGGER transcripts_search_docs_trg ON transcripts")
    op.execute("DROP TRIGGER videos_search_docs_trg ON videos")
    op.execute("DROP FUNCTION search_docs_sync()")
    op.execute("DROP FUNCTION refresh_search_doc(TEXT)")
    op.execute("DROP TABLE search_docs")
