# ytdigest — Architecture Specification

**Status:** implementable specification
**Derived from:** `plan.md` (design draft, 2026-09-13)
**Scope:** concrete component, data, interface, and contract definitions. This
document assumes every decision in `plan.md` §4 and does not re-argue them; it
specifies *how* they are realized.

---

## 0. Corrections to `plan.md`

Four issues surfaced while making the design concrete. Three are defects that
would have caused real bugs; one is a naming collision. All are resolved below
and reflected throughout this document.

### C1 — `jobs.UNIQUE(video_id, kind)` breaks re-analysis (defect)

`plan.md` §7 makes enqueue idempotent via `UNIQUE(video_id, kind)`, and §6.2
enqueues an `analyze` job for every transcript when `PROMPT_VERSION` changes.
**These contradict each other.** After a video has been analyzed once, an
`analyze` row exists for it forever, so `ON CONFLICT DO NOTHING` silently
discards every subsequent re-analysis request.

This defeats D7 — the ability to re-run a new prompt against stored
transcripts — which the plan identifies as its highest-leverage structural
choice.

**Resolution:** jobs carry an explicit `dedupe_key`, and uniqueness is enforced
by a **partial** unique index covering only *active* states:

```sql
dedupe_key TEXT NOT NULL
UNIQUE INDEX jobs_active_uniq ON jobs (video_id, kind, dedupe_key)
    WHERE state IN ('pending', 'running')
```

- `ingest` / `transcribe` → `dedupe_key = 'default'`
- `analyze` → `dedupe_key = '<prompt_version>:<summarizer_name>'`

Completed jobs no longer block new ones, and re-analysis at a new prompt
version is a distinct key. Replays within a version remain idempotent.

### C2 — Fixed lease vs. multi-hour transcription (defect)

§8 sets a ~6 h lease and has the planner requeue anything `running` past it.
But D3 accepts that a long episode *legitimately* takes multiple hours. A
stricter lease kills healthy work; a looser one leaves genuinely dead jobs
stranded for hours. A fixed timeout cannot distinguish the two.

**Resolution:** worker **heartbeat**. Workers update `heartbeat_at` every 60 s
while processing. The reaper requeues jobs whose heartbeat is stale (default
5 min), not jobs that have merely been running a long time. A 9-hour
transcription with a live heartbeat is untouched; a worker killed 90 s ago is
reclaimed promptly.

### C3 — `speaker_roster` placed on the wrong table (defect)

§7 puts `speaker_roster` on `transcripts`. But per D4b the roster is **derived
by the LLM at analysis time** — it is model- and prompt-dependent output, not a
property of the transcript. Storing it on `transcripts` means a re-analysis
under a new prompt either overwrites another version's roster or is unable to
store its own.

**Resolution:** split by provenance.

| Column | Table | Why |
|---|---|---|
| `speaker_source` | `transcripts` | How transcript-level labels arose (`subtitle_labels` / `none`) — a property of acquisition |
| `speaker_roster` | `analyses` | LLM-derived, versioned with `prompt_version` and `model` |

### C4 — "Backfill" names two unrelated operations (collision)

§6.2 uses *backfill* for the prompt-version re-analysis sweep; D9b uses
*backfill* for ingesting a channel's historical catalog. Different triggers,
different cost profiles, different priorities.

**Resolution:** distinct names used consistently from here on.

| Term | Meaning | Trigger | Priority |
|---|---|---|---|
| **Channel backfill** | Ingest historical videos of a channel | Explicit API call | Low (`-10`) |
| **Re-analysis sweep** | Re-run `analyze` on stored transcripts | `PROMPT_VERSION` change | Low (`-5`) |

### Minor corrections

- §7 shows `fts tsvector GENERATED` under `transcript_chunks`; per §6.1 search
  is over transcripts. FTS is defined on `transcripts.full_text` (§6 below),
  with a second optional index on chunks for v2 semantic/hybrid search.
- `plan.md` never states **who writes `transcript_chunks`**. Specified in §7.3:
  the transcriber writes chunks at transcript-save time, deterministically.
- `CHUNK_SEC` is listed in config but `OVERLAP_SEC` (D9, 60 s) is not. Added.

---

## 1. Component topology

```
                          ┌──────────────────────────┐
                          │   browser (React SPA)    │
                          └────────────┬─────────────┘
                                       │ HTTPS (reverse proxy)
                                       ▼
   ┌───────────────────────────────────────────────────────────────┐
   │  api  (FastAPI, stateless, N replicas possible)               │
   │  • enqueue  • read models  • SSR render path  • ops endpoints │
   └───────────────┬───────────────────────────────┬───────────────┘
                   │ write jobs                    │ read
                   ▼                               ▼
   ┌───────────────────────────────────────────────────────────────┐
   │                        PostgreSQL 17                          │
   │   domain tables │ jobs (queue) │ tsvector FTS │ (v2 pgvector) │
   └───┬───────────────────┬────────────────────┬─────────────┬────┘
       │ claim             │ claim              │ read/write  │ maintain
       ▼                   ▼                    ▼             ▼
 ┌────────────┐     ┌────────────┐       ┌────────────┐  ┌──────────┐
 │transcriber │     │  analyzer  │       │  (v2)      │  │ planner  │
 │ CPU-bound  │     │ LLM-bound  │       │  notifier  │  │ periodic │
 │ scale: N   │     │ scale: N   │       │            │  │ scale: 1 │
 └─────┬──────┘     └─────┬──────┘       └────────────┘  └──────────┘
       │                  │
       ▼                  ▼
 ┌───────────┐     ┌──────────────┐
 │ yt-dlp    │     │ Summarizer   │
 │ ffmpeg    │     │  ├ ollama    │
 │ whisper   │     │  └ anthropic │
 │ audio vol │     └──────────────┘
 └───────────┘
```

**Singleton constraint:** `planner` must run at exactly one replica. Its
operations (RSS polling, reaping, audio purge, sweep enqueue) are not safe to
run concurrently without advisory locking. If replicas ever become necessary,
wrap each periodic task in `pg_try_advisory_lock`.

All other services are stateless and horizontally scalable; `SKIP LOCKED`
makes worker concurrency safe by construction.

---

## 2. Repository layout

```
ytdigest/
├── common/                    # shared library, no service entrypoints
│   ├── config.py              # pydantic-settings; single source of env truth
│   ├── models.py              # domain dataclasses + Protocol definitions
│   ├── db.py                  # connection/pool management
│   ├── repo/                  # repository layer, one module per aggregate
│   │   ├── channels.py
│   │   ├── videos.py
│   │   ├── transcripts.py
│   │   ├── analyses.py
│   │   └── jobs.py
│   ├── queue.py               # JobQueue port + PostgresQueue adapter
│   ├── worker.py              # claim loop, heartbeat, signal handling
│   ├── errors.py              # error taxonomy (§8.4)
│   └── logging.py             # structlog config, correlation ids
│
├── adapters/                  # outbound integrations (ports implemented here)
│   ├── youtube/
│   │   ├── metadata.py        # yt-dlp --dump-json
│   │   ├── subtitles.py       # VTT fetch + parse
│   │   ├── audio.py           # bestaudio → 16 kHz mono opus
│   │   ├── feed.py            # RSS (monitoring)
│   │   └── catalog.py         # uploads playlist (channel backfill)
│   ├── transcription/
│   │   └── faster_whisper.py
│   └── summarize/
│       ├── prompts/           # versioned prompt templates (v1/, v2/ …)
│       ├── schema.py          # pydantic models for LLM JSON output
│       ├── ollama.py
│       └── anthropic.py
│
├── services/
│   ├── api/                   # FastAPI app
│   │   ├── main.py
│   │   ├── routes_read.py     # separated per D13
│   │   ├── routes_write.py
│   │   ├── routes_ops.py
│   │   ├── render.py          # SSR path for PDF/email (D12c)
│   │   ├── deps.py            # auth dependency (no-op in v1), rate-limit hook
│   │   └── templates/
│   ├── planner/main.py
│   ├── transcriber/main.py
│   └── analyzer/main.py
│
├── web/                       # React SPA (Vite, TS) — build artifact only
├── migrations/                # alembic
├── ops/
│   ├── Dockerfile.backend     # api, planner, analyzer, migrate  (§11.3)
│   ├── Dockerfile.whisper     # transcriber (derives from backend)
│   ├── Dockerfile.frontend    # node build → nginx runtime
│   └── nginx.conf             # SPA fallback, CSP, /api proxy
├── compose.yml
├── compose.override.yml       # dev only (§11.7)
├── .dockerignore
└── tests/
```

**Dependency rule:** `services/ → adapters/ → common/`. Nothing in `common/`
imports an adapter; nothing in `adapters/` imports a service. This is what
keeps D5's summarizer swap and D10's queue swap to single-file changes.

---

## 3. Ports (internal interfaces)

Defined in `common/models.py` as `Protocol` classes. These are the seams
`plan.md` §15 relies on.

```python
class MetadataSource(Protocol):
    def fetch(self, video_id: str) -> VideoMeta: ...

class SubtitleSource(Protocol):
    def available(self, meta: VideoMeta) -> SubtitleAvailability: ...
    def fetch(self, video_id: str, lang: str, kind: str) -> list[Segment]: ...

class AudioSource(Protocol):
    # returns path to 16 kHz mono opus (D6b)
    def fetch_normalized(self, video_id: str, dest: Path) -> AudioRef: ...

class Transcriber(Protocol):
    def transcribe(self, audio: AudioRef) -> TranscriptResult: ...

class Summarizer(Protocol):
    name: str
    def derive_roster(self, meta: VideoMeta, opening: str) -> Roster: ...
    def analyze_chunk(self, chunk: Chunk, roster: Roster,
                      meta: VideoMeta) -> ChunkAnalysis: ...
    def reduce(self, partials: list[ChunkAnalysis],
               meta: VideoMeta) -> str: ...

class JobQueue(Protocol):
    def enqueue(self, kind: str, video_id: str, *, dedupe_key: str = "default",
                payload: dict | None = None, priority: int = 0,
                run_after: datetime | None = None) -> int | None: ...
    def claim(self, kinds: list[str]) -> AbstractContextManager[Job | None]: ...
    def heartbeat(self, job_id: int) -> None: ...
```

`Summarizer` has three methods rather than one because D4b's roster pass and
D9's reduce step are distinct LLM calls with distinct prompts and distinct
failure modes. Collapsing them into one `summarize()` would hide that a roster
failure is recoverable while a chunk failure may not be.

---

## 4. Job kinds

| Kind | Producer | Consumer | Typical duration | Cost of retry |
|---|---|---|---|---|
| `ingest` | api, planner | transcriber | 2–10 s | negligible |
| `transcribe` | transcriber | transcriber | 20 min – 6 h | **very high** |
| `analyze` | transcriber, planner | analyzer | 1–10 min | moderate (LLM spend) |
| `notify` *(v2)* | analyzer | notifier | seconds | negligible |

`ingest` and `transcribe` are both handled by the transcriber service but are
**separate kinds** so the expensive path is independently observable, retryable,
and rate-limitable.

### Priority bands

| Band | Value | Used by |
|---|---|---|
| Interactive | `+10` | `POST /videos` — a human is waiting |
| Normal | `0` | RSS-discovered new videos |
| Re-analysis sweep | `-5` | `PROMPT_VERSION` change (C4) |
| Channel backfill | `-10` | historical catalog ingest (D9b, C4) |

Claim ordering is `priority DESC, run_after ASC`. This is what prevents a
200-episode backfill from starving today's episodes (D9b).

RSS-gap reconciliation (#93) queues at the Normal band with `origin = 'rss'`:
a channel that uploads more than the ~15 feed entries between two polls loses
videos the feed never showed, and those are forward-monitored content, not
history. Only uploads positioned above a known post-`monitor_from` video (the
anchor) in the channel's newest-50 listing are recovered; anything below it may
be pre-cutoff history and stays for an explicit backfill (D9b, #41).

Backfills beyond `POST /channels/{id}/backfill`'s 500 uploads (#115) are not a
job kind: `jobs.video_id` is `NOT NULL`, so a per-channel job does not fit, and a
multi-minute listing would hold a transcriber slot. The API records a row in
`backfill_requests`, and the planner task `channel_backfill` (every
`BACKFILL_INTERVAL_SEC`) takes one bounded slice of the oldest active request per
run: at most 500 new videos, fewer while 1000 backfill `ingest` jobs are pending.

---

## 5. Job lifecycle

```
                    ┌─────────┐
      enqueue ─────▶│ pending │◀──── reaper (stale heartbeat)
                    └────┬────┘      │
                  claim  │           │
                         ▼           │
                    ┌─────────┐──────┘
                    │ running │──── heartbeat every 60s
                    └────┬────┘
            ┌────────────┼────────────┐
       ok   │      transient          │ permanent
            ▼            ▼            ▼
       ┌────────┐   ┌─────────┐  ┌────────┐
       │  done  │   │ pending │  │  dead  │
       └────────┘   │(backoff)│  └────────┘
                    └─────────┘   retryable only by operator
```

### Claim query

```sql
UPDATE jobs SET state='running', locked_at=now(), heartbeat_at=now(),
                locked_by=%(worker)s, attempts=attempts+1
WHERE id = (
    SELECT id FROM jobs
    WHERE state='pending' AND kind = ANY(%(kinds)s) AND run_after <= now()
    ORDER BY priority DESC, run_after
    FOR UPDATE SKIP LOCKED
    LIMIT 1)
RETURNING *;
```

An analyzer adds `AND dedupe_key = %(dedupe_key)s` (an exact match, never a
pattern) with its own `<PROMPT_VERSION>:<summarizer>` (C1), so it never claims
a job just to defer it. Each planner tick, `stale_analyze` finds `pending`
`analyze` jobs on any other version and re-keys each to the current version in
place (keeping its priority), or marks it `dead` as superseded when the target
key already has work or the video already has an analysis at that version (#100).

### Reaping (C2)

```sql
UPDATE jobs SET state='pending', locked_by=NULL, locked_at=NULL,
                error_class=NULL, last_error='worker lost: heartbeat stale'
WHERE state='running' AND heartbeat_at < now() - interval '5 minutes';
```

### Backoff

`run_after = now() + LEAST(5min * 2^(attempts-1), 6h)`, with ±10% jitter.
Max attempts: 4 for `ingest`/`analyze`, **2 for `transcribe`** — a retry there
costs hours of CPU, so it gets one second chance, not three.

---

## 6. Data model (DDL)

```sql
-- ============ channels & videos ============

CREATE TABLE channels (
    channel_id    TEXT PRIMARY KEY,
    title         TEXT,
    active        BOOLEAN     NOT NULL DEFAULT true,
    monitor_from  TIMESTAMPTZ NOT NULL DEFAULT now(),  -- D9b forward-only cutoff
    last_polled   TIMESTAMPTZ,
    last_poll_err TEXT,
    added_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    poll_failing_since TIMESTAMPTZ,                    -- start of the current failed-poll run; NULL = last poll ok (#90)
    poll_failures INTEGER     NOT NULL DEFAULT 0,      -- failed RSS polls in a row; 0 iff poll_failing_since IS NULL (#90)
    CHECK (poll_failures >= 0),
    CHECK ((poll_failures = 0) = (poll_failing_since IS NULL))
);

CREATE TABLE videos (
    video_id      TEXT PRIMARY KEY,
    channel_id    TEXT REFERENCES channels(channel_id),
    title         TEXT,
    duration_sec  INTEGER,
    published_at  TIMESTAMPTZ,
    description   TEXT,
    origin        TEXT NOT NULL DEFAULT 'adhoc',   -- adhoc | rss | backfill
    unavailable   TEXT,                            -- removed | private | geoblocked | agegated
    discovered_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX videos_channel_pub_idx ON videos (channel_id, published_at DESC);
```

`channels.monitor_from` is what makes D9b's forward-only rule explicit and
auditable, rather than implicit in "whatever RSS returned first".

```sql
-- ============ transcripts ============

CREATE TABLE transcripts (
    id             BIGSERIAL PRIMARY KEY,
    video_id       TEXT NOT NULL REFERENCES videos(video_id) ON DELETE CASCADE,
    source         TEXT NOT NULL,   -- youtube_manual | youtube_auto | whisper
    language       TEXT,
    speaker_source TEXT NOT NULL DEFAULT 'none',  -- subtitle_labels | none  (C3)
    segments       JSONB NOT NULL,  -- [{start,end,text,speaker?}]
    full_text      TEXT NOT NULL,
    engine_meta    JSONB,           -- whisper model, compute type, beam, rtf
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (video_id, source)
);

ALTER TABLE transcripts ADD COLUMN fts tsvector
    GENERATED ALWAYS AS (to_tsvector(transcript_ts_config(language), coalesce(full_text,''))) STORED;
CREATE INDEX transcripts_fts_idx ON transcripts USING GIN (fts);

-- language -> built-in text-search config (#112, migration 0011); NULL/empty ->
-- english, unsupported codes -> simple. IMMUTABLE and baked into fts: changing it
-- needs a new migration that also re-runs ALTER COLUMN fts SET EXPRESSION.
-- transcript_ts_config(lang TEXT) RETURNS regconfig
-- transcript_search_tsquery(q TEXT) RETURNS tsquery  -- OR over all those configs

-- source preference (D2): manual > whisper > auto
CREATE FUNCTION transcript_rank(src TEXT) RETURNS INT IMMUTABLE LANGUAGE sql AS
$$ SELECT CASE src WHEN 'youtube_manual' THEN 0
                   WHEN 'whisper' THEN 1
                   WHEN 'youtube_auto' THEN 2 ELSE 3 END $$;
```

Encoding the D2 preference order as a SQL function keeps it in one place —
otherwise it gets re-implemented inconsistently in the analyzer, the API, and
the SPA.

Search (#111) matches more than the transcript, so a second table holds one
combined, weighted vector per video. A generated column cannot read other
tables, so `search_docs.fts` is a plain column kept in sync by the database
(triggers on `videos`, `transcripts` and `analyses`; the `analyses` one is a
deferred constraint trigger that fires at commit, after topics and claims are
inserted). It is created by migration `0010_search_docs`:

```sql
CREATE TABLE search_docs (
    video_id   TEXT PRIMARY KEY REFERENCES videos(video_id) ON DELETE CASCADE,
    fts        tsvector NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX search_docs_fts_idx ON search_docs USING GIN (fts);
-- refresh_search_doc(video_id) builds: title (A) || latest analysis TL;DR (B)
-- || topic titles+summaries (C) || claim texts (C) || preferred transcript's
-- transcripts.fts (D). Title and analysis text are always 'english'.
```

Query weights are `'{0.1, 0.2, 0.4, 1.0}'` (D, C, B, A).

```sql
-- ============ chunks (D9c) ============

CREATE TABLE transcript_chunks (
    id             BIGSERIAL PRIMARY KEY,
    transcript_id  BIGINT NOT NULL REFERENCES transcripts(id) ON DELETE CASCADE,
    seq            INTEGER NOT NULL,
    start_sec      NUMERIC(10,3) NOT NULL,
    end_sec        NUMERIC(10,3) NOT NULL,
    text           TEXT NOT NULL,
    chunk_strategy TEXT NOT NULL,      -- e.g. 'time:900:60'
    -- v2: embedding vector(N), added without touching anything above
    UNIQUE (transcript_id, chunk_strategy, seq)
);
CREATE INDEX chunks_lookup_idx ON transcript_chunks (transcript_id, chunk_strategy, seq);
```

`chunk_strategy` encodes the parameters (`time:<CHUNK_SEC>:<OVERLAP_SEC>`), so
changing `CHUNK_SEC` produces a *new* chunk set rather than silently
invalidating stored ones — the failure mode D9c warns about.

```sql
-- ============ analyses ============

CREATE TABLE analyses (
    id             BIGSERIAL PRIMARY KEY,
    video_id       TEXT NOT NULL REFERENCES videos(video_id) ON DELETE CASCADE,
    transcript_id  BIGINT NOT NULL REFERENCES transcripts(id),
    chunk_strategy TEXT NOT NULL,
    model          TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    tldr           TEXT NOT NULL,
    speaker_roster JSONB,              -- LLM-derived, versioned  (C3)
    input_tokens   INTEGER NOT NULL DEFAULT 0,
    output_tokens  INTEGER NOT NULL DEFAULT 0,
    cost_usd       NUMERIC(10,5),
    duration_ms    INTEGER,
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX analyses_video_idx ON analyses (video_id, created_at DESC);
CREATE INDEX analyses_version_idx ON analyses (prompt_version, model);

CREATE TABLE topics (
    id          BIGSERIAL PRIMARY KEY,
    analysis_id BIGINT NOT NULL REFERENCES analyses(id) ON DELETE CASCADE,
    seq         INTEGER NOT NULL,
    title       TEXT NOT NULL,
    summary     TEXT,
    start_sec   NUMERIC(10,3)
);

CREATE TABLE claims (
    id          BIGSERIAL PRIMARY KEY,
    analysis_id BIGINT NOT NULL REFERENCES analyses(id) ON DELETE CASCADE,
    text        TEXT NOT NULL,
    speaker     TEXT NOT NULL DEFAULT 'unknown',
    start_sec   NUMERIC(10,3),
    confidence  TEXT,              -- high | medium | low  (covers attribution, D4b)
    source_chunk_seq INTEGER       -- provenance for debugging map-reduce
);
CREATE INDEX claims_analysis_idx ON claims (analysis_id, start_sec);

CREATE TABLE quotes (
    id          BIGSERIAL PRIMARY KEY,
    analysis_id BIGINT NOT NULL REFERENCES analyses(id) ON DELETE CASCADE,
    text        TEXT NOT NULL,
    speaker     TEXT NOT NULL DEFAULT 'unknown',
    start_sec   NUMERIC(10,3),
    source_chunk_seq INTEGER
);
```

`source_chunk_seq` exists because map-reduce failures are otherwise very hard
to diagnose — when a claim looks wrong, you need to know which chunk produced
it.

```sql
-- ============ media (D6b) ============

CREATE TABLE media (
    id            BIGSERIAL PRIMARY KEY,
    video_id      TEXT NOT NULL REFERENCES videos(video_id) ON DELETE CASCADE,
    path          TEXT NOT NULL,
    bytes         BIGINT NOT NULL,
    format        TEXT NOT NULL DEFAULT 'opus16k',
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at    TIMESTAMPTZ NOT NULL,
    UNIQUE (video_id, format)
);
CREATE INDEX media_expiry_idx ON media (expires_at);
CREATE INDEX media_lru_idx ON media (created_at);   -- oldest-first eviction

-- ============ jobs (C1, C2) ============

CREATE TABLE jobs (
    id           BIGSERIAL PRIMARY KEY,
    video_id     TEXT NOT NULL,
    kind         TEXT NOT NULL,      -- ingest | transcribe | analyze | notify
    dedupe_key   TEXT NOT NULL DEFAULT 'default',      -- C1
    state        TEXT NOT NULL DEFAULT 'pending',      -- pending|running|done|dead
    priority     INTEGER NOT NULL DEFAULT 0,
    payload      JSONB NOT NULL DEFAULT '{}',
    attempts     INTEGER NOT NULL DEFAULT 0,
    last_error   TEXT,
    error_class  TEXT,               -- §8.4
    run_after    TIMESTAMPTZ NOT NULL DEFAULT now(),
    locked_by    TEXT,
    locked_at    TIMESTAMPTZ,
    heartbeat_at TIMESTAMPTZ,        -- C2
    finished_at  TIMESTAMPTZ,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at   TIMESTAMPTZ,        -- latest claim (#60)
    alerted_at   TIMESTAMPTZ         -- dead-letter announced to the operator (#67)
);

CREATE UNIQUE INDEX jobs_active_uniq ON jobs (video_id, kind, dedupe_key)
    WHERE state IN ('pending', 'running');                       -- C1
CREATE INDEX jobs_claim_idx ON jobs (kind, priority DESC, run_after)
    WHERE state = 'pending';
CREATE INDEX jobs_reap_idx ON jobs (heartbeat_at) WHERE state = 'running';
CREATE INDEX jobs_dead_idx ON jobs (created_at DESC) WHERE state = 'dead';

-- ============ planner ============

-- One row per periodic planner task, overwritten in place (#98). Written only
-- by the planner; nothing deletes rows, so a task removed from the registry
-- keeps its last row. A stalled planner shows as last_started_at /
-- last_finished_at that stop moving; a killed run as last_started_at >
-- last_finished_at (or last_finished_at NULL).
CREATE TABLE planner_task_runs (
    task_name              TEXT PRIMARY KEY,
    interval_sec           DOUBLE PRECISION NOT NULL,  -- Task.interval_sec at the last write
    last_started_at        TIMESTAMPTZ,                -- lock acquired, before fn() runs
    last_finished_at       TIMESTAMPTZ,
    last_outcome           TEXT CHECK (last_outcome IN ('ok', 'failed')),
    last_error             TEXT,                       -- "<Class>: <message>", <= 500 chars, no traceback
    last_count             INTEGER,                    -- value fn() returned; NULL on failure or None
    last_ok_at             TIMESTAMPTZ,                -- finish time of the last ok run
    last_skipped_locked_at TIMESTAMPTZ                 -- another planner held the lock; touches no other column
);
```

### v2 tables (declared, not created in v1)

```sql
-- recipients, notifications  (D12c)
-- transcript_chunks.embedding vector(N) + HNSW index  (D9c)
```

---

## 7. Pipeline stages

### 7.1 `ingest`

```
1. yt-dlp --dump-json                       → VideoMeta
   ├─ error → classify (§8.4); mark videos.unavailable; dead-letter
2. upsert videos
3. evaluate subtitle availability:
     manual EN present?        → fetch, parse VTT, speaker_source per labels
     PREFER_WHISPER=1?         → skip to (5)
     auto EN present?          → fetch only if allow_auto (see below)
     none                      → (5)
4. if transcript obtained: persist, chunk (§7.3), enqueue analyze
5. else: enqueue transcribe (same priority as this job)
```

**Premieres and live streams (#65).** After step 1, `live_status` of
`is_upcoming`, `is_live` or `post_live` (or yt-dlp's "Premieres in ..." error)
makes ingest raise `Defer` instead of continuing: the job returns to `pending`
with a future `run_after`, its attempt refunded, nothing written. A premiere
runs again 10 minutes after its scheduled start (1 hour if the start is unknown
or already past and the video is still upcoming); `is_live` rechecks in 2 hours
(a stream's end is not knowable), `post_live` in 30 minutes. `was_live` is an
ordinary video. When `UPCOMING_MAX_WAIT_DAYS` have passed since
`videos.discovered_at` the job dead-letters as `PERMANENT_SOURCE` with the
reason in `last_error`; `videos.unavailable` is left alone.

**Auto-caption policy.** D2 ranks `youtube_auto` *below* Whisper. Concretely
`ingest` does not take auto-captions when Whisper is available; it enqueues
`transcribe`. Auto-captions are used only when `AUTO_CAPTION_FALLBACK=1` and
transcription has already dead-lettered — a degraded result beats none, and
`transcripts.source` records the degradation so §13's comparison can measure it.
The planner's `auto_caption_fallback` sweep (#92) is what triggers it: it
enqueues one `ingest` for a video whose newest `transcribe` job is `dead`.

### 7.2 `transcribe`

```
1. acquire audio: yt-dlp bestaudio → ffmpeg → 16 kHz mono opus   (D6b)
2. register media row with expires_at = now() + AUDIO_TTL_DAYS
3. faster-whisper (WHISPER_MODEL, int8, beam 5, VAD)
   └─ heartbeat every 60 s throughout                            (C2)
4. persist transcript (source='whisper', engine_meta incl. measured RTF)
5. chunk (§7.3)
6. enqueue analyze
7. if AUDIO_KEEP=0: delete audio + media row
```

Recording measured RTF in `engine_meta` turns §11's one-off benchmark into
continuous data — the D3 revisit trigger ("backlog grows monotonically")
becomes a query rather than a guess.

### 7.3 Chunking — who and when

**The transcriber writes chunks at transcript-save time**, not the analyzer.
Chunking is deterministic and depends only on the transcript plus
`CHUNK_SEC`/`OVERLAP_SEC`; doing it once at save keeps the analyzer pure
(read chunks → call LLM → write analysis) and makes map-reduce reproducible.

```
strategy = f"time:{CHUNK_SEC}:{OVERLAP_SEC}"
windows  = [start, start+CHUNK_SEC) stepping by CHUNK_SEC,
           each extended backwards by OVERLAP_SEC for context
single chunk if total_duration <= CHUNK_SEC
INSERT ... ON CONFLICT (transcript_id, chunk_strategy, seq) DO NOTHING
```

If the analyzer finds no chunks for the configured strategy (because
`CHUNK_SEC` changed), it generates and persists them for that strategy before
proceeding. This makes a config change safe rather than silently inconsistent.

### 7.4 `analyze`

```
1. select best transcript via transcript_rank()                  (D2)
2. load chunks for current chunk_strategy (generate if absent)
3. roster pass:  derive_roster(meta, first ~10 min of text)      (D4b)
4. map:  for each chunk → analyze_chunk(chunk, roster, meta)
         roster injected as a CLOSED set; model emits 'unknown' or a roster name
5. reduce: single-chunk → partial tldr; multi-chunk → reduce(partials)
6. dedupe claims/quotes across overlap seams (normalized-text key)
7. persist analysis + children in ONE transaction
```

Steps 3–5 are separate LLM calls with separate retry semantics: a roster
failure falls back to an empty roster (all speakers `unknown`) and continues;
a chunk failure fails the job.

---

## 8. Contracts

### 8.1 LLM output schema (D8)

Validated with pydantic before any database write. Invalid JSON is a retryable
error with the validation message fed back into one repair attempt, then
dead-lettered.

```jsonc
// roster pass
{ "speakers": [ { "name": "…", "role": "host|guest|panelist|unknown" } ] }

// per-chunk analysis
{
  "topics": [ { "title": "…", "summary": "…", "start_sec": 0 } ],
  "claims": [ { "text": "…", "speaker": "…", "start_sec": 0,
                "confidence": "high|medium|low" } ],
  "quotes": [ { "text": "…", "speaker": "…", "start_sec": 0 } ]
}
```

**Validation rules enforced in code, not prompt:**

- `speaker` ∈ roster names ∪ `{"unknown"}` — anything else is coerced to
  `unknown` and counted in a metric. This is the hard guarantee behind D4b;
  prompts alone cannot enforce a closed set.
- `start_sec` must fall within the chunk's span, else clamped and flagged.
- Empty `claims` on a non-trivial chunk is a warning signal, not an error.

### 8.2 Prompt versioning

Prompts are files under `adapters/summarize/prompts/<version>/`, never inline
strings. `PROMPT_VERSION` selects the directory. A prompt change without a
version bump is a deployment error — the version is what makes D7's comparison
and C4's sweep meaningful.

### 8.3 HTTP API

Read routes (`routes_read.py`) and write routes (`routes_write.py`) are
separated per D13.

| Method | Path | Notes |
|---|---|---|
| `POST` | `/videos` | body `{url}`; priority `+10`; returns job id |
| `POST` | `/channels` | forward-only; sets `monitor_from` |
| `POST` | `/channels/{id}/backfill` | `{limit, dry_run}`; priority `-10` (D9b) |
| `POST` | `/channels/{id}/backfill-requests` | `{limit <= 5000, dry_run}`; only records a request, the planner `channel_backfill` task runs it (#115); `409` while the channel has a pending or running one |
| `GET` | `/channels/{id}/backfill-requests/{request_id}` | state, attempts, enqueued, timestamps and `result` (the shape of the synchronous response) (#115) |
| `POST` | `/channels/{id}/backfill/cancel` | `{backfill_id}`; deletes that backfill's pending `ingest` jobs (#116) |
| `POST` | `/videos/{id}/reanalyze` | force new `analyze` at current version |
| `GET` | `/videos` | paginated library view; filters: channel, state, date |
| `GET` | `/videos/{id}` | latest analysis + job state |
| `GET` | `/videos/{id}/analyses` | all runs (Compare view, D7) |
| `GET` | `/videos/{id}/transcript` | paginated segments |
| `GET` | `/videos/{id}/render` | **server-side HTML render** (D12c) |
| `GET` | `/search?q=` | FTS over `search_docs` (title, latest analysis, preferred transcript); `matched_in` and transcript `ts_headline` excerpts |
| `GET` | `/ops/jobs` | filter by state/kind — powers Ops view |
| `POST` | `/ops/jobs/{id}/retry` | resurrect a dead job |
| `GET` | `/healthz` | liveness + queue depth by kind/state |
| `GET` | `/metrics` | Prometheus text format (optional) |

Every route passes through `deps.require_auth` — a no-op returning `None` in
v1, one implementation change away from enforcing a bearer token (D13).

`POST /channels/{id}/backfill` (#41) stays synchronous for `limit <= 500`
(90 s listing, inside nginx's `proxy_read_timeout`). Larger backfills, up to the
catalog adapter's 5000, go through a backfill request that the planner
processes outside the HTTP request. Cancelling with the request's `backfill_id`
(in its `result`) also ends the request (`failed`, `error_class = CANCELLED`).

#### 8.3.1 `POST /channels/{id}/backfill` and the catalog's `REMOVED` ambiguity

**Known limitation, recorded and not worked around (#107, follow-up to #27).**
A catalog `PermanentSourceError(REMOVED)` — the error `CatalogSource.list_uploads`
raises from yt-dlp's "The playlist does not exist" — means only "this channel
has nothing listable". It does not mean "the channel was removed". The catalog
adapter cannot, and does not try to, tell apart:

- a channel ID that was never real,
- a channel YouTube terminated, and
- a real, existing channel that simply has no uploads (including one whose
  uploads are all private, members-only, or Shorts — see below).

Every caller, including this backfill route, treats `REMOVED` as an empty
listing (`200`, `listed: 0`, an INFO log, no retry) and never as proof that the
channel is gone. It is never persisted as channel or video state: not
`channels.active`, `channels.last_poll_err`, `videos.unavailable`, nor
`videos.unavailable_reason`. Whether a registered channel still exists is
judged only through RSS poll health (§8.4, #34, #90), not through the catalog.

**The evidence.** `tests/fixtures/ytdlp_errors/cases.toml` and
`tests/adapters/youtube/fixtures/catalog/README.md` (yt-dlp 2026.08.19,
recorded 2026-09-28) show a made-up channel ID (`UCaaaaaaaaaaaaaaaaaaaaaa`), a
terminated channel (`UCx7T6qYK4VaP2-OhorrFS3Q`), and several real channels with
no uploads (YouTube's own "Sports", "Music", "Gaming" and others) all printing
the identical line, "The playlist does not exist.". `adapters/youtube/errors.py`
maps that line to `REMOVED` for all of them; `tests/adapters/youtube/test_catalog.py`
pins all three cases to the same outcome. No real channel was ever observed
returning an empty-JSON playlist (`entries: []`) — the
`empty_channel.json` fixture is synthetic, kept only to exercise that shape in
the parser.

**Rejected workarounds:**

- **An RSS cross-check.** Rejected: `adapters/youtube/feed.py` documents that
  the RSS feed answers 404 even for valid channels (`FeedNotFoundError` is
  treated as transient), and nobody has recorded what the feed of a genuinely
  empty channel looks like. That makes it a noisy signal, not a way to
  distinguish the cases.
- **A second yt-dlp call to the channel page.** Rejected: only the terminated
  case has a recorded page line ("This channel was removed because it violated
  our Community Guidelines."). The page output for a made-up ID and for an
  empty channel was never recorded. Matching on that wording would also be
  fragile (YouTube can change it without notice) and would add a network round
  trip inside #41's 90 s backfill request budget.

**What was not recorded**, and so must not be assumed to behave like the cases
above: the channel-page wording for a made-up channel ID and for an empty
channel, and the uploads-playlist behaviour for a channel whose uploads are all
private, members-only, or Shorts-only. These are presumed to fall into the same
`REMOVED`/"nothing listable" bucket, but that has not been checked live.

### 8.4 Error taxonomy

`plan.md` §8 flags this as "worth designing explicitly". It is the difference
between a fast dead-letter and four wasted Whisper runs.

| Class | Examples | Handling |
|---|---|---|
| `PERMANENT_SOURCE` | video removed, private, geo-blocked, age-gated | dead-letter immediately; set `videos.unavailable`; **no retry** |
| `TRANSIENT_NETWORK` | timeout, 5xx, DNS | retry with backoff |
| `RATE_LIMITED` | 429, bot-check | retry with long backoff; respect `Retry-After` |
| `TOOL_FAILURE` | yt-dlp extractor broken | retry twice, then dead-letter **with an alert** — means "go update yt-dlp" (§16.11) |
| `LLM_INVALID_OUTPUT` | schema validation failed | one repair attempt, then dead-letter |
| `LLM_UNAVAILABLE` | provider 5xx, connection refused | retry with backoff |
| `RESOURCE` | disk full, OOM | dead-letter; alert — retrying makes it worse |
| `BUG` | unhandled exception | dead-letter with traceback in `last_error` |

`error_class` is stored on the job so the Ops view can distinguish "YouTube
changed something" from "my LLM is down" at a glance.

A NULL `error_class` on a `pending`/`dead` job means the reaper reset it after a
lost heartbeat (`last_error = 'worker lost: heartbeat stale'`); it is not a
class and has no retry policy.

---

## 9. Frontend architecture

```
web/src/
├── api/           # generated TS client from OpenAPI (D12) — never hand-written
├── hooks/         # TanStack Query hooks, one per resource
├── routes/        # Library, VideoDetail, Transcript, Search, Compare, Ops
├── components/
│   ├── Player.tsx        # youtube-nocookie IFrame; exposes seekTo()  (D12b)
│   ├── Timestamp.tsx     # single click-to-seek primitive, used everywhere
│   └── ClaimList.tsx
└── main.tsx
```

- **No global state library.** TanStack Query owns server state; the only
  genuine client state is the player handle, held in one context.
- **`Timestamp` is a single component.** Claims, quotes, topics, and transcript
  segments all render through it, so seek behavior exists in exactly one place.
- **Types generated from OpenAPI** at build time; a backend contract change
  fails the frontend build rather than surfacing at runtime (D12).
- **Transcript virtualization deferred** (D12b) by paginating segments
  server-side — `GET /videos/{id}/transcript?offset=&limit=`.
- **Stack:** React 19.3 + Vite 8 (Rolldown) + TypeScript; React Compiler
  enabled (§16.4) so memoization is automatic rather than hand-managed.
  Routing uses `react-router` (owner decision, 2026-10-03; #122). Tests run on
  Vitest, with `jsdom` and `@testing-library/react` for component tests
  (#127).
- **CSP:** `frame-src https://www.youtube-nocookie.com` only; `default-src
  'self'`. Enforced in `ops/nginx.conf` (§11.3), not in application code.
- **Served by nginx, same origin as the API** via `/api` proxy (§11.2), so no
  CORS configuration exists to get wrong.

### 9.1 Player transport: direct `postMessage`, not the `iframe_api` script

**Decision (owner, 2026-10-03; #48):** `Player` controls the
`youtube-nocookie.com` embed by speaking the IFrame Player API's `postMessage`
protocol to an `enablejsapi=1` iframe directly. It loads no YouTube script and
uses no npm wrapper.

**Why:**

- **The CSP stays as specified.** `default-src 'self'` plus
  `frame-src https://www.youtube-nocookie.com` is enough. Nothing from a
  third party runs in our origin. D12b requires keeping the rest of the CSP
  strict, and this is the only option that does.
- **No third-party code with access to our page.** The official script would
  run with full access to the DOM and to the same-origin `/api`.
- **Small and testable.** The transport is a few dozen lines in one module
  with no React in it. Tests fake `contentWindow` and the incoming messages,
  with no network.
- **Incoming messages are checked.** A message is accepted only if both its
  `origin` and its `source` are the embed, and it parses as a JSON object.

**Accepted cost:** the wire format (`listening` handshake, `command` messages,
`onReady`/`onError`/`initialDelivery` events) is what YouTube's own widget
script sends. It is not documented on its own and could change without notice.
If it breaks, seeking stops working, but the embed still plays. The
embed-error fallback (open `youtube.com/watch?v=<id>&t=<s>s` in a new tab)
still works.

**Discarded alternative: the official `https://www.youtube.com/iframe_api`
script.** It is the documented, supported way to use the API, and it pulls in
`www-widgetapi.js`. It was rejected because:

- it needs `script-src 'self' https://www.youtube.com` in the CSP, and possibly
  more YouTube script origins as its loader changes;
- it runs third-party JavaScript in our origin.

The `react-youtube` and `youtube-player` wrappers were rejected for the same
reason, since they load that script too.

**When to revisit:** switch to the official script if any of these happens:
- the undocumented protocol breaks and cannot be fixed quickly;
- we need player features the protocol does not expose (for example,
  bidirectional sync, which D12b defers);
- the threat model changes so that a YouTube `script-src` becomes acceptable.

The change stays inside the transport module
(`web/src/components/player/`), plus `script-src` in `ops/nginx.conf` (§11.3)
and in this section's CSP line. Do not widen the CSP without updating this
section.

---

## 10. Configuration

Single `common/config.py` using pydantic-settings; services import it rather
than reading `os.environ` directly.

| Variable | Default | Consumer |
|---|---|---|
| `DATABASE_URL` | — | all |
| `SUMMARIZER` | `ollama` | analyzer |
| `OLLAMA_MODEL` / `OLLAMA_HOST` | `qwen3.5:4b` | analyzer (see 16.7) |
| `OLLAMA_OUTPUT_RESERVE_TOKENS` | `2048` | analyzer: tokens of `OLLAMA_NUM_CTX` kept free for the reply; prompt budget is `OLLAMA_NUM_CTX - OLLAMA_OUTPUT_RESERVE_TOKENS`, and startup fails if it is `>= OLLAMA_NUM_CTX` (#85) |
| `ANTHROPIC_API_KEY` / `ANTHROPIC_MODEL` | — / `claude-haiku-4-5` | analyzer (D6: API key only) |
| `ANTHROPIC_BATCH` | `1` | use Batch API — 50% cost (16.8) |
| `PROMPT_VERSION` | `v1` | analyzer, planner |
| `SWEEP_MAX_JOBS_ANTHROPIC` | `100` | planner: cumulative per-`PROMPT_VERSION` count of re-analysis sweep jobs allowed while `SUMMARIZER=anthropic` (`0` = none); ignored for ollama (#99) |
| `CHUNK_SEC` | `900` | transcriber, analyzer |
| `OVERLAP_SEC` | `60` | transcriber, analyzer |
| `WHISPER_MODEL` | `large-v3` | transcriber |
| `WHISPER_COMPUTE` | `int8` | transcriber |
| `WHISPER_THREADS` | `0` (auto) | transcriber |
| `PREFER_WHISPER` | `0` | transcriber |
| `AUTO_CAPTION_FALLBACK` | `1` | transcriber, planner |
| `AUDIO_KEEP` / `AUDIO_TTL_DAYS` / `AUDIO_MAX_GB` | `1` / `30` / `20` | transcriber, planner |
| `AUDIO_MIN_FREE_GB` / `AUDIO_MIN_FREE_PCT` | `2` / `5` | planner: free-space floor on the `AUDIO_DIR` volume (#105). When free space is below the larger of the two, audio retention evicts the oldest audio beyond what `AUDIO_MAX_GB` would, never a running job's. `0` for either disables that half; `AUDIO_MIN_FREE_PCT` is `0 <= x < 100`. A floor at or above the volume size evicts nothing and logs `planner.audio_retention.free_space_floor_invalid` |
| `POLL_INTERVAL_SEC` | `3600` | planner |
| `RECONCILE_INTERVAL_SEC` | `21600` | planner: how often each active channel's newest uploads are checked against the RSS feed for videos it dropped (#93) |
| `BACKFILL_INTERVAL_SEC` | `60` | planner: how often the `channel_backfill` task takes one bounded slice of the oldest pending backfill request (#115) |
| `UPCOMING_MAX_WAIT_DAYS` | `14` | transcriber: ingest reschedules a premiere or live stream until it is available, and dead-letters it this many days after discovery (#65) |
| `HEARTBEAT_SEC` / `REAP_AFTER_SEC` | `60` / `300` | workers, planner |
| `MAX_ATTEMPTS_TRANSCRIBE` | `2` | transcriber |
| `INGEST_MAX_RUNTIME_SEC` / `ANALYZE_MAX_RUNTIME_SEC` / `TRANSCRIBE_MAX_RTF` | `600` / `3600` / `3.0` | workers (§11.5) |
| `TRANSCRIBER_CPUS` | `3.0` | compose |
| `ALERT_WEBHOOK_URL` | unset (no channel) | planner: secret; `TOOL_FAILURE` / `RESOURCE` dead-letters are POSTed here (#67) |
| `ALERT_WEBHOOK_FORMAT` | `json` | planner: `json` (`{"text": ...}`, Slack-compatible) or `text` (plain body plus `Title` header, ntfy) |
| `ALERT_MIN_INTERVAL_SEC` | `3600` | planner: at most one alert per error class in this window; later dead-letters are counted into the next one |
| `DB_POOL_MIN_SIZE` / `DB_POOL_MAX_SIZE` | `1` / `5` | api: bounds of the Postgres connection pool opened at startup (#95); positive, and `MIN <= MAX` (equal is fine) |
| `DB_POOL_TIMEOUT_SEC` | `2` | api: how long a request waits for a pooled connection before `503`; positive, at most `4` so the wait fits inside the 5 s healthcheck timeout (11.4) |
| `LOG_LEVEL` / `LOG_FORMAT` | `INFO` / `json` | all |

Pool sizing rule: `api replicas x DB_POOL_MAX_SIZE + workers + planner + migrate` must stay below Postgres `max_connections` (default 100). Only the api pools; workers, planner, CLI and migrate open one connection each through `common.db.connect()`.

---

## 11. Deployment

### 11.1 Container inventory

Six runtime containers, four build contexts, three custom images.

| Container | Image | Built from | Replicas | Role |
|---|---|---|---|---|
| `db` | `postgres:18-alpine` | upstream | 1 | data + queue |
| `migrate` | `ytdigest-backend` | `ops/Dockerfile.backend` | one-shot | `alembic upgrade head` |
| `api` | `ytdigest-backend` | same image | 1..N | HTTP API |
| `planner` | `ytdigest-backend` | same image | **exactly 1** | periodic tasks (§1) |
| `analyzer` | `ytdigest-backend` | same image | 1..N | worker: `analyze` |
| `transcriber` | `ytdigest-whisper` | `ops/Dockerfile.whisper` | 1..N | worker: `ingest`, `transcribe` |
| `web` | `ytdigest-web` | `ops/Dockerfile.frontend` | 1 | nginx: static SPA + reverse proxy |
| `metabase` *(optional)* | upstream | — | 1 | ad-hoc queries, read-only role |

**One backend image serves four containers.** `api`, `planner`, `analyzer`, and
`migrate` differ only by their `command`. Building four near-identical images
would triple build time and create four things to keep in sync for no benefit.
The transcriber is the exception because Whisper's dependency tree (torch,
ctranslate2, ffmpeg) adds roughly 2 GB that the other three would otherwise
carry for nothing.

### 11.2 Revision to earlier draft: the SPA gets its own container

An earlier version of this section had `api` serving the built SPA from
`/app/static`. **That is superseded.** The frontend is a separate nginx
container, for reasons that are concrete rather than stylistic:

- nginx handles gzip/brotli, cache headers, and `immutable` fingerprinted
  assets correctly out of the box; FastAPI's `StaticFiles` does none of this
  well.
- Security headers (CSP, HSTS, `X-Content-Type-Options`) belong in one place.
  D12b requires a specific `frame-src` for the YouTube embed — putting that in
  nginx config keeps it reviewable rather than buried in Python.
- The SPA rebuilds far more often than the backend. Separate images means a UI
  change rebuilds a ~30 MB image, not a ~400 MB one.
- It makes the frontend genuinely independent — replaceable, or served from a
  CDN later, without touching the API.

nginx also reverse-proxies `/api/*` to the `api` container, so the browser sees
a single origin. **This eliminates CORS entirely** — no preflight requests, no
`Access-Control-Allow-Origin` configuration, and D13's "CORS default-deny" rule
becomes trivially satisfied because there is no cross-origin request to
permit.

### 11.3 Image definitions

#### `ops/Dockerfile.backend`

Multi-stage; the builder compiles wheels, the runtime carries none of the build
toolchain.

```dockerfile
# ---- builder ----
FROM python:3.14-slim AS builder
ENV PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential && rm -rf /var/lib/apt/lists/*
WORKDIR /build
COPY requirements.backend.txt .
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install -r requirements.backend.txt

# ---- runtime ----
FROM python:3.14-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg curl && rm -rf /var/lib/apt/lists/*
COPY --from=builder /opt/venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH PYTHONUNBUFFERED=1 PYTHONPATH=/app
WORKDIR /app
COPY common/ ./common/
COPY adapters/ ./adapters/
COPY services/ ./services/
COPY migrations/ ./migrations/
COPY alembic.ini .
RUN useradd -r -u 10001 -m app && chown -R app:app /app
USER app
```

`ffmpeg` is in the backend image because `ingest` may normalize audio even on
the subtitle path; `yt-dlp` is a Python dependency, not a system package.

**Layer ordering matters:** requirements are copied and installed *before*
application code. Editing a Python file then rebuilds only the final `COPY`
layer, not the dependency install — the difference between a 5-second and a
3-minute rebuild during Phase 1–5 iteration.

#### `ops/Dockerfile.whisper`

```dockerfile
FROM ytdigest-backend:latest
USER root
WORKDIR /tmp
# --only-binary: ctranslate2 ships no sdist, so a missing wheel must fail
# loudly here rather than as an opaque resolver error. See 16.2.
RUN --mount=type=bind,source=requirements.whisper.txt,target=requirements.whisper.txt \
    /opt/venv/bin/pip install --no-cache-dir --only-binary=:all: \
        -r requirements.whisper.txt \
    || (echo "ERROR: no binary wheel for a requirements.whisper.txt package for this Python minor, or the pin is wrong - see architecture.md 16.2" \
        && exit 1)
# /models is the whisper-models volume; a fresh volume copies its ownership.
RUN mkdir -p /models && chown app:app /models
WORKDIR /app
USER app
ENV OMP_NUM_THREADS=4 \
    HF_HOME=/models \
    CT2_VERBOSE=0
```

Deriving from the backend image rather than rebuilding from `python:3.14-slim`
means shared layers are stored and pulled once. `HF_HOME` points at a named
volume so the multi-GB `large-v3` model survives container recreation —
otherwise every `docker compose up --build` re-downloads it.

`OMP_NUM_THREADS` is set explicitly because ctranslate2 otherwise detects the
*host's* core count, ignoring the container's CPU limit, and oversubscribes.

#### `ops/Dockerfile.frontend`

```dockerfile
# ---- build ----
FROM node:24-alpine AS build
WORKDIR /src
COPY web/package.json web/package-lock.json ./
RUN npm ci                      # ci, not install — respects the lockfile exactly
COPY web/ .
ARG VITE_API_BASE=/api
RUN npm run build               # prebuild runs gen:api first → /src/dist

# ---- runtime ----
FROM nginx:1.27-alpine
RUN rm -rf /usr/share/nginx/html/*   # the base image's own index.html, 50x.html
COPY --from=build /src/dist /usr/share/nginx/html
COPY ops/nginx.conf /etc/nginx/conf.d/default.conf
```

**This is where D12's "no Node in production" becomes literal.** Node exists
only in the `build` stage; the shipped image is nginx plus static files, ~30 MB,
with zero npm packages present at runtime. A compromised transitive dependency
can affect the build, but there is no `node_modules` on the running host to
exploit.

`npm ci` rather than `npm install` is deliberate — it fails if `package.json`
and the lockfile disagree, which is what makes D12's "commit the lockfile"
mitigation actually enforced rather than aspirational.

#### `ops/nginx.conf` (essentials)

```nginx
# Web server for the SPA and the /api/ reverse proxy (architecture.md 9, 11.3,
# issue #57). Installed as /etc/nginx/conf.d/default.conf, so it is included
# inside the base image's `http {}` block.

# Deviation from 11.3: the access log carries nginx's request id, the same id
# sent upstream as X-Request-Id (12 correlation).
log_format ytdigest '$remote_addr - [$time_local] "$request" $status '
                    '$body_bytes_sent "$http_referer" rid=$request_id';

# Raw request URI minus the /api prefix; always begins with "/".
map $request_uri $api_path {
    default            /;
    ~^/+api(?<rest>/.*)$ $rest;
}


# Deviation from 11.3: the CSP below adds object-src, base-uri, form-action and
# frame-ancestors, which only restrict further (default-src does not cover
# them). Names no origin beyond YouTube's embed host and thumbnail host.
# Deviation from 11.3: each CSP is written on ONE line. 11.3's original listing
# used `\` continuations inside the quoted string; nginx does not interpret
# them, so the header would have carried the backslashes and the indentation
# literally (a latent bug in the listing itself).
# NOTE on add_header: a location with any add_header of its own inherits none
# from the server level, so the three security headers are repeated in every
# location nginx answers itself, and /api/ has none (the API's own headers pass
# through untouched).

server {
    listen 80;
    root /usr/share/nginx/html;
    access_log /var/log/nginx/access.log ytdigest;

    # Deviation from 11.3: redirects nginx issues are relative, never naming
    # the container's :80 (wrong behind the 127.0.0.1:8080 mapping and proxy).
    absolute_redirect off;
    # Deviation from 11.3: no version in the Server header or error pages.
    server_tokens off;

    # Deviation from 11.3: compression of text assets. gzip_proxied stays off.
    gzip on;
    gzip_vary on;
    gzip_proxied off;
    gzip_types text/css text/javascript application/javascript application/json image/svg+xml;

    # Deviation from 11.3: Docker's embedded DNS, so `api` is resolved at
    # request time and followed to a new address when compose recreates it.
    # Without this nginx resolves `api` once at startup, refuses to start when
    # it does not exist yet, and keeps sending to a dead IP afterwards.
    resolver 127.0.0.11 valid=10s ipv6=off;
    resolver_timeout 5s;

    # Fingerprinted assets: cache hard. No SPA fallback here, so a missing
    # asset is a real 404 and never index.html cached as JavaScript for a
    # year. Cache-Control has no `always`: a 404 must not be immutable.
    location /assets/ {
        add_header Cache-Control "public, max-age=31536000, immutable";
        add_header Content-Security-Policy "default-src 'self'; frame-src https://www.youtube-nocookie.com; img-src 'self' https://i.ytimg.com data:; style-src 'self' 'unsafe-inline'; object-src 'none'; base-uri 'self'; form-action 'self'; frame-ancestors 'none'" always;
        add_header X-Content-Type-Options nosniff always;
        add_header Referrer-Policy no-referrer always;
    }

    # A directory without its trailing slash: a relative 301, not the SPA.
    location = /assets {
        add_header Content-Security-Policy "default-src 'self'; frame-src https://www.youtube-nocookie.com; img-src 'self' https://i.ytimg.com data:; style-src 'self' 'unsafe-inline'; object-src 'none'; base-uri 'self'; form-action 'self'; frame-ancestors 'none'" always;
        add_header X-Content-Type-Options nosniff always;
        add_header Referrer-Policy no-referrer always;
        return 301 /assets/;
    }

    # SPA client-side routing: unknown paths serve index.html (200). `$uri/`
    # is left out on purpose so that directories fall through to the app.
    # /apiary and /api-docs are SPA routes: only /api/ is proxied.
    location / {
        try_files $uri /index.html;
        add_header Cache-Control "no-cache";
        add_header Content-Security-Policy "default-src 'self'; frame-src https://www.youtube-nocookie.com; img-src 'self' https://i.ytimg.com data:; style-src 'self' 'unsafe-inline'; object-src 'none'; base-uri 'self'; form-action 'self'; frame-ancestors 'none'" always;
        add_header X-Content-Type-Options nosniff always;
        add_header Referrer-Policy no-referrer always;
    }

    # /api without the slash must never be the SPA.
    location = /api {
        add_header Content-Security-Policy "default-src 'self'; frame-src https://www.youtube-nocookie.com; img-src 'self' https://i.ytimg.com data:; style-src 'self' 'unsafe-inline'; object-src 'none'; base-uri 'self'; form-action 'self'; frame-ancestors 'none'" always;
        add_header X-Content-Type-Options nosniff always;
        add_header Referrer-Policy no-referrer always;
        return 301 /api/$is_args$args;
    }

    # Operational metrics (#60) are for the internal network only; nginx never
    # publishes anything under /api/metrics. Do NOT simplify this to `=`: an
    # exact match leaves /api/metrics/ and /api/metrics/x to fall through to
    # /api/ and be proxied. `^~` is a prefix match that also stops nginx from
    # trying regex locations, so it covers the whole space and wins over /api/
    # (a plain prefix) whatever is added later. It also blocks /api/metricsfoo;
    # failing closed is intended. Matching is on the normalized URI, so
    # /api/%6Detrics is covered too.
    location ^~ /api/metrics {
        add_header Content-Security-Policy "default-src 'self'; frame-src https://www.youtube-nocookie.com; img-src 'self' https://i.ytimg.com data:; style-src 'self' 'unsafe-inline'; object-src 'none'; base-uri 'self'; form-action 'self'; frame-ancestors 'none'" always;
        add_header X-Content-Type-Options nosniff always;
        add_header Referrer-Policy no-referrer always;
        return 404;
    }

    # API reverse proxy, /api prefix stripped. Deviation from 11.3: a variable
    # upstream (lazy DNS, see resolver above). A proxy_pass with variables does
    # no prefix replacement, so the stripped path is built from the raw
    # $request_uri: path encoding and query string reach the API byte for byte.
    # No add_header here, no proxy_intercept_errors, gzip off: the API's own
    # status, headers and body pass through unchanged.
    location /api/ {
        set $api_upstream api:8000;
        proxy_pass http://$api_upstream$api_path;
        proxy_set_header X-Request-Id $request_id;   # replaces any client value
        proxy_connect_timeout 5s;
        proxy_read_timeout 120s;
        gzip off;
    }
}
```

`try_files … /index.html` is required for the SPA: a deep link like
`/videos/abc123` must return the app, not a 404, because routing happens
client-side.

### 11.4 Compose stack

```yaml
name: ytdigest

# The whole v1 stack (architecture.md 11.4-11.6): `db`, the one-shot
# `migrate`, `api`, `planner`, `analyzer`, `transcriber` and `web`.
# `migrate` builds and tags the backend image that `api`, `planner` and
# `analyzer` reuse; `transcriber` and `web` build their own images.
# Production deploys use `docker compose -f compose.yml up -d --build`
# explicitly, so `compose.override.yml` (dev-only) is never applied by
# accident.

# Exactly the keys architecture.md 11.4 lists. Service-specific variables go
# on the service; `ANTHROPIC_API_KEY` is on `analyzer` only, never here.
x-backend-env: &backend-env
  DATABASE_URL: postgresql://ytdigest:${POSTGRES_PASSWORD}@db:5432/ytdigest
  SUMMARIZER: ${SUMMARIZER:-ollama}
  ANTHROPIC_BATCH: ${ANTHROPIC_BATCH:-1}
  PROMPT_VERSION: ${PROMPT_VERSION:-v1}
  CHUNK_SEC: ${CHUNK_SEC:-900}
  OVERLAP_SEC: ${OVERLAP_SEC:-60}
  LOG_FORMAT: json

x-logging: &logging
  driver: json-file
  options: {max-size: "10m", max-file: "3"}

services:
  db:
    image: postgres:18-alpine   # >=18.6 (architecture.md 16.1)
    environment:
      POSTGRES_USER: ytdigest
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:?required}
      POSTGRES_DB: ytdigest
    # postgres:18's image moved PGDATA under /var/lib/postgresql/<major>/docker
    # and declares VOLUME /var/lib/postgresql (not .../data as in older
    # images and architecture.md's literal example): mounting at .../data
    # makes the entrypoint refuse to start, seeing "unused" data there.
    volumes:
      - pgdata:/var/lib/postgresql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ytdigest -d ytdigest"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks: [internal]
    logging: *logging
    restart: unless-stopped

  # One-shot: runs `alembic upgrade head` and exits. Migrations never run on
  # service startup (AGENTS.md -> Rules); the long-running services wait for
  # this to exit 0. It owns the only build of the backend image.
  migrate:
    build: {context: ., dockerfile: ops/Dockerfile.backend}
    image: ytdigest-backend:latest
    command: alembic upgrade head
    environment: *backend-env
    depends_on: {db: {condition: service_healthy}}
    networks: [internal]
    restart: "no"
    logging: *logging

  api:
    image: ytdigest-backend:latest
    command: uvicorn services.api.main:app --host 0.0.0.0 --port 8000
    environment: *backend-env
    depends_on:
      migrate: {condition: service_completed_successfully}
    healthcheck:
      test: ["CMD", "curl", "-fsS", "http://localhost:8000/healthz"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 20s
    networks: [internal, edge]
    logging: *logging
    restart: unless-stopped

  web:
    build: {context: ., dockerfile: ops/Dockerfile.frontend}
    image: ytdigest-web:latest
    ports: ["127.0.0.1:8080:80"]      # loopback only (D13)
    depends_on: {api: {condition: service_healthy}}
    networks: [edge]
    logging: *logging
    restart: unless-stopped

  planner:
    image: ytdigest-backend:latest
    command: python -m services.planner.main
    environment:
      <<: *backend-env
      POLL_INTERVAL_SEC: ${POLL_INTERVAL_SEC:-3600}
      AUDIO_TTL_DAYS: ${AUDIO_TTL_DAYS:-30}
      AUDIO_MAX_GB: ${AUDIO_MAX_GB:-20}
      AUDIO_KEEP: ${AUDIO_KEEP:-1}
    depends_on:
      migrate: {condition: service_completed_successfully}
    volumes: [audio:/data/audio]        # needs it to unlink purged files
    # The loop touches /tmp/heartbeat each pass and sleeps in slices of at
    # most 30 s (#38), so it fits the same 180 s test as the other workers.
    # Dollar signs are doubled so Compose leaves them for the shell.
    healthcheck: &worker-health
      test: ["CMD-SHELL", "test $$(( $$(date +%s) - $$(stat -c %Y /tmp/heartbeat) )) -lt 180"]
      interval: 60s
      retries: 3
      start_period: 60s
      start_interval: 5s    # so `up --wait` does not wait a full interval
    # Hard constraint: exactly one planner - see architecture.md section 1.
    deploy: {replicas: 1}
    # Must stay above WORKER_SHUTDOWN_GRACE_SEC (default 20 s, not set here);
    # Docker's own default of 10 s would SIGKILL a worker mid-drain.
    stop_grace_period: 30s
    networks: [internal, egress]
    logging: *logging
    restart: unless-stopped

  analyzer:
    image: ytdigest-backend:latest
    command: python -m services.analyzer.main
    environment:
      <<: *backend-env
      OLLAMA_HOST: ${OLLAMA_HOST:-http://host.docker.internal:11434}
      OLLAMA_MODEL: ${OLLAMA_MODEL:-qwen3.5:4b}
      ANTHROPIC_MODEL: ${ANTHROPIC_MODEL:-claude-haiku-4-5}
      # The only service that gets the key. From .env or the deploy shell's
      # environment; a Docker secret file is #136.
      ANTHROPIC_API_KEY: ${ANTHROPIC_API_KEY:-}
    depends_on:
      migrate: {condition: service_completed_successfully}
    extra_hosts: ["host.docker.internal:host-gateway"]   # host Ollama (11.5)
    healthcheck: *worker-health
    stop_grace_period: 30s              # above WORKER_SHUTDOWN_GRACE_SEC
    networks: [internal, egress]
    logging: *logging
    restart: unless-stopped

  transcriber:
    build:
      context: .
      dockerfile: ops/Dockerfile.whisper
      # ops/Dockerfile.whisper is `FROM ytdigest-backend:latest`. Naming that
      # image as a build context from the `migrate` service makes Compose
      # build `migrate` first and use the result, so the whisper image is
      # never built on a stale or missing backend image.
      additional_contexts:
        "ytdigest-backend:latest": "service:migrate"
    image: ytdigest-whisper:latest
    command: python -m services.transcriber.main
    environment:
      <<: *backend-env
      WHISPER_MODEL: ${WHISPER_MODEL:-large-v3}
      WHISPER_COMPUTE: ${WHISPER_COMPUTE:-int8}
      # Moves together with TRANSCRIBER_CPUS below: threads above the CPU
      # quota only oversubscribe it. 3 threads for the default 3.0 CPUs.
      WHISPER_THREADS: ${WHISPER_THREADS:-3}
      PREFER_WHISPER: ${PREFER_WHISPER:-0}
      AUTO_CAPTION_FALLBACK: ${AUTO_CAPTION_FALLBACK:-1}
      MAX_ATTEMPTS_TRANSCRIBE: ${MAX_ATTEMPTS_TRANSCRIBE:-2}
      AUDIO_KEEP: ${AUDIO_KEEP:-1}
      AUDIO_MAX_GB: ${AUDIO_MAX_GB:-20}
    volumes:
      - audio:/data/audio
      - whisper-models:/models
    depends_on:
      migrate: {condition: service_completed_successfully}
    healthcheck: *worker-health
    deploy:
      resources:
        limits: {cpus: "${TRANSCRIBER_CPUS:-3.0}", memory: 6G}
    stop_grace_period: 30s              # above WORKER_SHUTDOWN_GRACE_SEC
    networks: [internal, egress]
    logging: *logging
    restart: unless-stopped

volumes:
  pgdata:
  audio:
  whisper-models:

# `internal: true` has no route to the internet or the host. The transcriber
# must reach YouTube and Hugging Face, the planner YouTube RSS, and the
# analyzer Ollama on the host or the Anthropic API, so those three also sit
# on `egress`, an ordinary bridge. `db` and `migrate` never do.
networks:
  internal:
    internal: true
  edge: {}
  egress: {}
```

Corrections to the earlier listing, all shipped in #58: `db` mounts
`pgdata:/var/lib/postgresql` (#7); `planner`, `analyzer` and `transcriber`
also join a new ordinary `egress` network, because an `internal: true`
network has no route to YouTube, Hugging Face, the Anthropic API or the host
Ollama; `analyzer` gets `extra_hosts` for `host.docker.internal`; `planner`
gets the heartbeat healthcheck; the workers get `stop_grace_period: 30s`
(above `WORKER_SHUTDOWN_GRACE_SEC`'s 20 s); the healthcheck escapes `$` as
`$$` and adds `start_interval`; `WHISPER_THREADS` defaults to `3` to match
`TRANSCRIBER_CPUS=3.0`; and the whisper build names `ytdigest-backend:latest`
as an additional context from `migrate`, so it never builds on a stale base.

### 11.5 Deployment mechanics worth calling out

**Migration ordering.** `migrate` is a one-shot container; every service gates
on `service_completed_successfully`. Running `alembic upgrade head` on service
startup instead would mean N replicas racing the same migration — a classic
way to corrupt a schema. One container, runs to completion, exits zero, then
everything else starts.

**Worker health checks.** Workers expose no HTTP port, so liveness is a
heartbeat file: the claim loop touches `/tmp/heartbeat` each iteration, and the
healthcheck asserts it is under 180 s old. This is distinct from the *job*
heartbeat in C2 — that one detects a dead worker holding a job; this one tells
Docker to restart a wedged process. Both are needed; neither substitutes for
the other.

**Hung handlers (#66).** Because the heartbeat thread keeps both of those fresh
while the main thread is wedged, each job also has a maximum wall-clock runtime:
`INGEST_MAX_RUNTIME_SEC` (600), `ANALYZE_MAX_RUNTIME_SEC` (3600), and for
`transcribe` the audio duration (`payload.duration_sec`, set by ingest) ×
`TRANSCRIBE_MAX_RTF` (3.0) plus a fixed 600 s allowance; a transcribe job with
no duration has no limit. Past the limit the worker stops heartbeating, records
the attempt as failed (`TOOL_FAILURE`, `last_error` says it exceeded its runtime
limit, so a job that always hangs dead-letters), logs at ERROR and exits with
status 70 so Docker restarts it.

**Network segmentation.** `internal: true` means the database and `migrate`
have no route off the compose bridge. The workers also sit on `egress`.
Only `api` sits on `internal` and `edge`, and only `web` publishes a port — bound to `127.0.0.1`
so it is reachable solely through the host's existing reverse proxy (D13).

**Transcriber resource limits.** `cpus` prevents Whisper from starving the rest
of the host. The 6 GB memory limit matters because `large-v3` at int8 needs
roughly 2–3 GB resident, and an unbounded container that OOMs takes the host's
OOM killer with it. Note `deploy.resources.limits` requires Compose v2 (it is
honored by `docker compose`, unlike the old v1 behavior where it was
Swarm-only).

**Ollama runs on the host, not in compose.** It is a shared service with its
own model cache, not part of this stack's lifecycle. The analyzer reaches it
via `host.docker.internal` (requires `extra_hosts: ["host.docker.internal:host-gateway"]`
on Linux). If it were containerized here, `docker compose down` would evict
multi-GB models.

**Secrets.** `.env` is gitignored; `.env.example` is committed with placeholder
values. `POSTGRES_PASSWORD` uses `:?required` so the stack refuses to start
rather than silently defaulting. For `ANTHROPIC_API_KEY`, prefer a Docker
secret or host-level env over `.env` once this leaves the development phase.

**`.dockerignore` is load-bearing**, not hygiene: without it, `web/node_modules`
and `.git` enter the build context, inflating it by hundreds of MB and
invalidating cache layers on every build.

```
.git
**/node_modules
**/__pycache__
**/.venv
*.md
tests/
data/
```

### 11.6 Operational commands

```bash
# first run
cp .env.example .env && $EDITOR .env
docker compose up -d --build

# scale the CPU-bound worker (the only one that benefits)
docker compose up -d --scale transcriber=2

# frontend-only change — rebuilds ~30 MB, not the backend
docker compose up -d --build web

# apply a new migration
docker compose run --rm migrate

# re-analysis sweep after a prompt change (C4)
PROMPT_VERSION=v2 docker compose up -d planner analyzer transcriber

# preview that sweep first: prints what it would enqueue and changes nothing (#99)
docker compose run --rm planner python -m services.planner.main --sweep-dry-run

# inspect the queue without psql
curl -s localhost:8080/api/healthz | jq
curl -s 'localhost:8080/api/ops/jobs?state=dead' | jq
```

The transcriber is in the rollout command because it builds the `analyze` key
from its own `PROMPT_VERSION`; left on the old value it keeps enqueueing
old-version jobs (the planner re-keys them, but only after a delay).

### 11.7 Development overlay

`compose.override.yml` (auto-merged by `docker compose` locally, absent in
production):

- bind-mounts `./common`, `./adapters`, `./services` into the backend
  containers, with `uvicorn --reload`, so backend edits don't require a rebuild
- runs the Vite dev server instead of nginx, proxying `/api` to `api:8000`,
  giving hot module reload
- publishes Postgres on `127.0.0.1:5432` for direct `psql` access
- sets `WHISPER_MODEL=tiny` so local iteration doesn't take hours

Decided in #59, beyond the list above:

- The source mounts are read-only (`:ro`). The containers run as uid 10001
  (#55), and a writable mount would let Python write `__pycache__` files owned
  by 10001 into the developer's tree. `api` reloads with uvicorn's built-in
  reloader, `--reload-dir` limited to the three mounted directories; no
  `watchfiles` or other package is added.
- `migrate` gets `./migrations` mounted read-only, so a new revision runs with
  `docker compose run --rm migrate` and no rebuild. Migrations still never run
  on startup.
- Vite runs on `127.0.0.1:5173` (from `node:24-alpine`, `./web` mounted), not
  8080: the overlay removes `web`'s `build:` (`!reset`) so the nginx image
  cannot be tagged `node:24-alpine`, and replaces the 8080 publish
  (`!override`). `API_PROXY_TARGET=http://api:8000` points its `/api` proxy at
  the API. Exactly two ports are published in dev, both on `127.0.0.1`:
  `db` 5432 and `web` 5173.
- `web` binds only the files Vite needs (`web/src`, `index.html`, `package.json`,
  `package-lock.json`, `tsconfig.json`, `vite.config.ts`), read-only, into
  `/home/node/app`; `node_modules` is created there in the container's own layer
  (host packages are glibc, the Alpine image is musl). No volume is mounted
  inside a bind source: on a fresh checkout the daemon would create that
  mountpoint in the host tree as root. The `web` command starts as root only
  to give `/home/node/app` to `node`, then runs `npm ci` and the dev server as
  `node`. A new top-level file Vite needs must be added to the list.
- The workers (`planner`, `analyzer`, `transcriber`) get the same mounts but
  keep their commands; after an edit, `docker compose restart <service>`
  picks it up. Restarting them automatically is #142.

Production deployment uses `docker compose -f compose.yml up -d` explicitly, so
the override is never applied by accident.

---

## 12. Observability

Proportional to a single-operator system: structured logs plus a handful of
gauges, no tracing stack.

- **Structured JSON logs** with `job_id`, `video_id`, `kind`, `attempt` on
  every line. One correlation id per job, propagated through adapter calls.
- **Key metrics** (exposed at `/metrics`, scraped or just curl'd):
  `queue_depth{kind,state}`, `job_duration_seconds{kind}`,
  `whisper_rtf`, `llm_tokens_total{model,direction}`, `llm_cost_usd_total`,
  `speaker_coercions_total` (§8.1 — the D4b quality canary),
  `audio_bytes_used`.
- **The two that matter most:** `queue_depth{kind="transcribe",state="pending"}`
  trending up over a week is D3's revisit trigger; `speaker_coercions_total`
  rising is D4b's.

---

## 13. Testing strategy

| Layer | Approach |
|---|---|
| Queue semantics | Real Postgres (testcontainers); concurrent claim, dedupe (C1), reap (C2) |
| Chunking | Pure function; property test — chunks cover the transcript, overlaps bounded |
| VTT parsing | Fixture files incl. malformed and rolling-caption duplicates |
| Adapters | Recorded `yt-dlp --dump-json` fixtures; no network in CI |
| Summarizer | `FakeSummarizer` returning canned JSON; plus schema-violation fixtures |
| Error classification | Table-driven over real yt-dlp/provider error strings |
| API | FastAPI `TestClient` against a seeded database |
| Quality (manual) | Golden set of ~20 videos; compare runs via the Compare view (D5) |

The quality evaluation is deliberately manual. Automating "is this summary
good" is a research project; comparing two runs side by side is a UI feature
that already exists for other reasons.

---

## 14. Build order

Refines `plan.md` §12 with concrete exit criteria.

| Phase | Deliverable | Exit criterion |
|---|---|---|
| 0 | Coverage + **three-way transcription bake-off** (whisper large-v3 / turbo / parakeet); **confirm `ctranslate2` cp314 wheel** (16.2) | RTF, WER, proper-noun accuracy; TTL 30 vs 90 decided (16.6) |
| 1 | Migrations, `common/`, adapters, CLI single-video path | `ytdigest run <url>` → analysis in DB |
| 2 | `queue.py` + `worker.py`; split transcriber/analyzer | Survives `kill -9` mid-transcribe; no duplicate work |
| 3 | `planner`: RSS, reaper, audio purge, sweep | One week unattended, no manual intervention |
| 4 | `api`: read/write/ops routes, FTS, SSR render path | All data reachable over HTTP |
| 4b | React SPA, six views | Operable without `psql` |
| 5 | Prompt v2, local-vs-cloud comparison | D5 decided on measured evidence |
| 6 | pgvector semantic search (D9c) / diarization (D4) | — |
| 7 | notifier: PDF + email + weekly digest (D12c) | One month unattended delivery |

**Phase 1 depends on Phase 0.** If subtitle coverage is high, the transcriber's
Whisper path is an edge case and Phase 1 can defer it entirely, going
subtitles-only until Phase 2.

---

## 15. Carried-forward open items

| # | Item | Blocks | Resolution path |
|---|---|---|---|
| O1 | Real channel list + subtitle coverage | Phase 1 scope | **Measured 2026-09-27 (task #2); the Phase 1 call is pending the repo owner.** 9 owner-chosen channels, 15 RSS entries each: manual subtitles 6 videos (1.95 h), auto-only 64 (48.72 h), none 0, error 2 (transient), 63 Shorts excluded. By default speech-to-text load is ~0 h; with `PREFER_WHISPER=1` it is ~48.7 h per window. Evidence: `_docs/progress.md`, "Phase 0 findings"; script `scripts/phase0/subtitle_coverage.py`. Whether the Whisper path is deferred from Phase 1 (section 14) is the owner's decision and is not yet recorded here |
| O2 | Transcription engine: whisper vs parakeet | D3 validity, D4 diarization premise, audio sizing | **Open.** Needs the deploy-host bake-off (>=3 real clips, >=1 noisy or accented) from #3, on a host prepared by #78. The containerised harness exists (#76), but numbers from the dev machine do not count and no real figures exist yet. Until then D3 stands, and the D4 revisit (only if Parakeet wins) is not triggered |
| O3 | `AUDIO_TTL_DAYS` 30 vs 90 | Disk provisioning | **Open; follows from O1 and O2.** Disk need = audio hours ingested per day x TTL days x MB per audio hour (16 kHz mono opus, 7.2/D6b), bounded by `AUDIO_MAX_GB` (default 20). The hours-per-day input depends on the O1 call (~0 h by default, ~48.7 h per 15-entry window with `PREFER_WHISPER=1`); the MB-per-hour figure has not been measured. The default stays 30 until the owner decides |
| O4 | Local vs cloud summarizer (D5) | Cost model | Phase 5, via Compare view |
| O5 | Embedding model + dimension (D9c) | v2 schema | Deferred; chunk table already prepared |
| O7 | `ctranslate2` wheel availability for Python 3.14 | transcriber image build | **Resolved 2026-09-27 (task #4): wheels exist for the whole tree (ctranslate2 4.8.2, av 18.1.0, tokenizers 0.23.2, onnxruntime 1.30.0); `faster-whisper` 1.2.1 installs and transcribes on `python:3.14-slim`.** The transcriber can derive from the 3.14 backend image; the 3.13 split is not needed, which is what #56 implements (`FROM ytdigest-backend:latest`, wheels only). Details in 16.2; also `_docs/progress.md`, "Phase 0 findings" |
| O6 | Whether `speaker_coercions_total` justifies diarization (D4) | v2 scope | Observe during Phase 5 |

---

## 16. Technology current review (September 2026)

Every choice in this document was re-verified against current releases. The
table summarizes; the discussion below covers only where the answer changes
something.

| Component | Specified | Current (Sep 2026) | Action |
|---|---|---|---|
| PostgreSQL | 17 | **18.6** | **Upgrade** — §16.1 |
| Python | 3.12 | **3.14.7** (3.15 in Oct) | **3.14 — matches host** — §16.2 |
| Node (build only) | 22 | **24 LTS** (26 → LTS Oct 2026) | **Upgrade** — §16.3 |
| React | 19 | **19.3** (Sep 9, 2026) | **Upgrade + Compiler** — §16.4 |
| Vite | 7-era assumptions | **8.0.9** (Rolldown) | **Adopt** — §16.5 |
| Transcription | faster-whisper `large-v3` | **Parakeet / parakeet.cpp** exists | **Add to Phase 0 bake-off** — §16.6 |
| Local LLM | `qwen3:8b` | Qwen3.5-4B class for CPU | Revised — §16.7 |
| Cloud LLM | `claude-haiku-4-5` | Still current tier; **Batch API** | **Adopt batch** — §16.8 |
| TanStack Query | v5 | v5.102 stable; v6 in RC | **No change** — stay v5 |
| pgvector | assumed | 0.8.2; VectorChord exists | **No change** — §16.9 |
| FastAPI | 0.11x era | **0.139.x** | Minor — §16.10 |
| yt-dlp | current | still the leader | No change; risk reconfirmed — §16.11 |
| nginx | 1.27 | fine | No change |
| Alembic / psycopg3 | current | fine | No change |

### 16.1 PostgreSQL 17 → 18

PostgreSQL 18 has been GA since September 2025 and is at 18.6. It is no longer
a new release and the early-cycle regressions have been through several fix
rounds.

Relevant to this system specifically:

- **Asynchronous I/O subsystem** — reported up to 3× improvement on read-heavy
  storage paths. The corpus grows to thousands of transcripts with large `TEXT`
  and `JSONB` columns, so this is the workload that benefits.
- **`uuidv7()` built in** — time-ordered UUIDs with better index locality. Not
  needed now (`BIGSERIAL` is correct for a single-node system), but worth
  knowing before reaching for `gen_random_uuid()` anywhere.
- **Virtual generated columns** — computed at query time. Note our `fts` column
  must stay **`STORED`**, because a GIN index requires a materialized value.
  Don't be tempted by the new default.

One caution: 18.5 was pulled and never shipped, and 18.3 fixed a volatility
regression affecting `json_strip_nulls()` in indexes. Pin `postgres:18-alpine`
at ≥18.6 rather than tracking `:18` loosely.

**Change:** `image: postgres:18-alpine` in §11.4.

### 16.2 Python 3.12 → 3.14 (host parity)

**Decision:** Python 3.14, matching the Ubuntu server's system Python.

Note that containers do not inherit the host's interpreter — the image version
and the host version are independent. Parity is therefore a *choice*, and a
defensible one: it means a venv on the host, an ad-hoc script, and the container
all behave identically, which removes a class of "works outside the container,
fails inside" confusion during Phase 0–1 when much of the work is CLI-driven.

**Pin the minor, not the patch.** Wheel compatibility and ABI are determined by
the minor version (`cp314`), not the patch. Pinning `3.14.4` exactly would
forgo the fixes in 3.14.5–3.14.7 for no compatibility benefit. Use
`python:3.14-slim`, which tracks the latest patch on the 3.14 line, and let the
host catch up on its own schedule.

**The one real risk — binary wheels for the transcriber.** `ctranslate2`
(under `faster-whisper`) ships platform-specific wheels and has historically
lagged new CPython minors: it notably had no 3.13 wheels *and no source
distribution* for a period, which left 3.12 as the only option. CTranslate2
release notes confirm 3.13 support. 3.14 support was unconfirmed when this
section was first written. It has since been verified, with the result below.
With no sdist fallback, a missing wheel is a hard build failure, not a slow
source compile.

**Verified 2026-09-27 (task #4): 3.14 works.** The checks ran in throwaway
containers on the Docker Desktop daemon, never on the host.

- *Wheel download.* `pip download faster-whisper --only-binary=:all:
  --python-version 3.14` resolved the full tree from wheels for
  Linux x86_64 (glibc), with no sdists. It ran twice with identical results:
  once natively inside `python:3.14-slim` and once cross-platform with
  `--platform manylinux_2_{17..28}_x86_64 --platform manylinux2014_x86_64`.
- *Smoke test.* In `python:3.14-slim` (Python 3.14.7, pip 26.2.1, Debian
  glibc 2.41), `pip install --only-binary=:all: faster-whisper` succeeded. The
  `tiny` model (CPU, `int8`) then transcribed three clips without error: a
  5 s 440 Hz sine passed as a numpy array (0 segments, as expected), the same
  sine with `vad_filter=True` (which loads `onnxruntime`), and a 5.5 s
  espeak-ng speech WAV decoded from a file path through PyAV. For the WAV it
  returned *"the quick brown fox jump over the rainy dog. The low world, this
  is a test."*

| Package | Version | Wheel resolved for cp314 | Result |
|---|---|---|---|
| faster-whisper | 1.2.1 | `py3-none-any` | pass |
| ctranslate2 | 4.8.2 | `cp314-cp314-manylinux_2_27/2_28_x86_64` | pass |
| av (PyAV) | 18.1.0 | `cp311-abi3-manylinux_2_28_x86_64` | pass |
| tokenizers | 0.23.2 | `cp310-abi3-manylinux_2_17_x86_64` | pass |
| onnxruntime | 1.30.0 | `cp314-cp314-manylinux_2_28_x86_64` | pass |
| numpy | 2.5.3 | `cp314-cp314-manylinux_2_27/2_28_x86_64` | pass |
| pyyaml | 6.0.3 | `cp314-cp314-manylinux_2_17/2_28_x86_64` | pass |
| protobuf | 7.36.2 | `cp310-abi3-manylinux2014_x86_64` | pass |
| hf-xet | 1.6.0 | `cp38-abi3-manylinux_2_17_x86_64` | pass |
| flatbuffers 25.12.19, huggingface-hub 1.33.0, httpx 0.28.1, httpcore 1.0.9, h11 0.16.0, anyio 4.15.1, idna 3.20, certifi 2026.7.22, click 8.5.0, filelock 4.0.4, fsspec 2026.9.0, packaging 26.3, tqdm 4.70.1, typing-extensions 4.16.0 | — | pure-Python `py3-none-any` | pass |

Python 3.13 (`python:3.13-slim`, 3.13.15) was also run through both steps.
Both passed, with the same versions and the `cp313` builds of the three
version-specific wheels (ctranslate2, numpy, onnxruntime), and an identical
transcript. So the fallback in mitigation 2 is known to work too, but it is
not needed.

*Pitfall with the download check.* Passing a single
`--platform manylinux_2_28_x86_64` makes pip accept **only** that exact tag.
Unlike an install on a real glibc system, it does not also accept older
`manylinux_2_17` / `manylinux2014` wheels. `tokenizers` publishes only
`manylinux_2_17` abi3 wheels, so that form of the command fails with
"no matching distributions: av, tokenizers" on 3.14 **and on 3.13 alike**.
The failure is a false negative. For cross-platform checks, list every
manylinux tag from `2_17` to the target, or run the check natively inside
the target image.

Three mitigations, in order of preference:

1. **Verify before building.** Check for a `cp314` wheel first:

       pip index versions ctranslate2
       pip download ctranslate2 --only-binary=:all: --python-version 3.14 --no-deps -d /tmp/x

   This belongs in the Phase 0 checklist, not in a debugging session later.

2. **Split the transcriber image if needed.** The backend image
   (`api`, `planner`, `analyzer`, `migrate`) has no such constraint — pure-Python
   dependencies — and can sit on 3.14 regardless. If `ctranslate2` has no 3.14
   wheel, build `Dockerfile.whisper` `FROM python:3.13-slim` independently
   rather than deriving from the backend image. The cost is losing shared layers
   (§11.3), not a redesign — the services communicate only through Postgres, so
   nothing requires them to share an interpreter version.

3. **Fail loudly at build time.** Add to `Dockerfile.whisper`, so a missing
   wheel surfaces as a clear message rather than a confusing resolver error:

       RUN /opt/venv/bin/pip install --no-cache-dir --only-binary=:all: \
               -r requirements.whisper.txt \
           || (echo "ERROR: no binary wheel for this Python minor \
               — see architecture.md 16.2" && exit 1)

**This risk disappears entirely if Parakeet wins the §16.6 bake-off**, since the
C++ ports need no Python runtime at inference and `ctranslate2` leaves the
dependency tree altogether. Another reason to run that measurement in Phase 0
before hardening the image.

Nothing in the design *uses* 3.14's new features — free-threading is irrelevant
here (concurrency is process-level via the job queue), and t-strings,
subinterpreters, and deferred annotations are all unused. The move is for
parity, not capability.

### 16.3 Node 22 → 24 (build stage only)

Node 22 entered Maintenance LTS in October 2025; **24 is Active LTS**; 26 is
Current and becomes LTS on 2026-10-28. Since Node appears only in the frontend
build stage and never at runtime (§11.3), the risk of moving is close to zero
and the benefit is staying on a line that receives fixes.

Also worth noting for planning: Node is moving to **one major per year (April),
LTS promotion each October, every release becoming LTS**. Upgrade cadence
becomes annual and predictable.

**Change:** `node:24-alpine` in `ops/Dockerfile.frontend`.

### 16.4 React 19 → 19.3, and adopt the React Compiler

React 19.3 shipped 2026-09-09 — four days ago — adding View Transitions,
Fragment Refs, and Trusted Types support. **View Transitions are directly
useful here**: navigating library → video detail → transcript is exactly the
kind of route change they smooth, and it is now a stable API rather than a
Canary experiment.

More importantly: **the React Compiler is stable at 1.0.** This matters because
D12 in `plan.md` explicitly accepted a trade-off — React's "re-render semantics,
effect dependencies… the recurring source of subtle bugs for developers who
don't work in it daily." The compiler automates memoization, which removes a
large share of exactly that class of bug. It materially reduces the cost that
decision accepted.

**Change:** React 19.3; enable `babel-plugin-react-compiler` in the Vite
config; consider View Transitions for route changes. Being days old, treat 19.3
itself as worth a beat of caution — 19.2 is a fine fallback, and the compiler is
available either way.

**Wiring (owner-approved, 2026-10-03; #47):** with Vite 8, `@vitejs/plugin-react`
offers two ways to enable the compiler:

- **Chosen: Babel.** `babel({ presets: [reactCompilerPreset()] })` from
  `@rolldown/plugin-babel`, together with `babel-plugin-react-compiler` 1.x
  and `@babel/core` (plus `@types/babel__core`). This is the stable 1.0
  compiler that this section names, and every package it needs is approved.
- **Not chosen: the plugin's native `compiler: true` option.** It uses
  `oxc-transform-react`, a Rust port of the compiler. Upstream calls it
  experimental, and the package is not approved.

Revisit the native option when it is stable, if Babel's build-time cost
becomes noticeable. Switching means one config line plus swapping the
dependencies.

### 16.5 Vite 8 with Rolldown

Vite 8.0 shipped 2026-03-12, replacing esbuild *and* Rollup with **Rolldown**, a
Rust bundler, reporting 10–30× faster builds with plugin compatibility
maintained. Current is 8.0.9.

This is a real win for the workflow in §11.6, where a frontend-only change
rebuilds a container image. Build time is the dominant cost of that loop.

Caveat worth holding: at the 8.0 announcement Rolldown itself was still at
release-candidate status and its minifier was alpha. That has had six months to
settle, but verify your production bundle rather than assuming — and the
`rolldown-vite` migration package is now archived, so the path is plain `vite@8`.

**Change:** Vite 8 in `web/package.json`.

### 16.6 Transcription — Parakeet belongs in the Phase 0 bake-off

**This is the finding that could change a `plan.md` decision.**

D3 chose `faster-whisper large-v3` on the reasoning that latency tolerance buys
quality. That reasoning is sound, but the option set has changed. NVIDIA's
**Parakeet** models, via C++ ports (`parakeet.cpp`), report:

- **~27× faster than whisper.cpp turbo on CPU at comparable accuracy**, and
  ~1.5× faster than NeMo's own PyTorch CPU runtime with byte-identical output.
- Roughly **2× lower peak RAM**, lower still quantized.
- **No Python runtime at inference** — which would remove torch/ctranslate2 from
  the transcriber image entirely, collapsing it toward the size of the base
  image.
- **Sortformer speaker diarization (up to 4 speakers)** in the same toolchain.

That last point is the significant one. **D4 deferred diarization specifically
because `pyannote` would add a heavy CPU dependency to a strained host.** If
Sortformer delivers diarization at these speeds, the premise of that deferral
weakens, and D4b's LLM-inferred speakers — whose accepted weakness is *silent*
misattribution, the worst failure mode in the system per §13 — could be replaced
or cross-checked by real acoustic diarization.

**Honest counterweights, because this is not a clear-cut swap:**

- Whisper is **more robust on noisy or accented audio**, which is precisely the
  profile of user-generated podcast audio. Multiple sources single out podcast
  transcription as where Whisper's training diversity still wins.
- Parakeet V3 covers ~25 European languages against Whisper's 99. Fine for
  English-language podcasts, limiting otherwise.
- Sortformer caps at 4 speakers. Fine for interviews, not for panels.
- The C++ ports are young projects with a thinner operational track record than
  faster-whisper.

**Change:** do **not** rewrite D3. Instead, Phase 0's measurement (§14) becomes
a three-way bake-off on real audio from the actual channel list: `faster-whisper
large-v3`, `faster-whisper turbo`, and `parakeet-tdt-0.6b-v3`, scored on RTF,
WER against a hand-checked sample, and proper-noun accuracy. The `Transcriber`
port in §3 already makes this a swap of one adapter — the architecture was built
for exactly this question, so let the measurement answer it.

If Parakeet wins, revisit D4 in the same pass.

### 16.7 Local LLM — the CPU verdict hardens

The model landscape moved (Qwen3.5, Gemma 4, gpt-oss-20b, DeepSeek V4), but the
hardware arithmetic moved against local inference on *this* host.

Current field guidance for CPU-only inference assumes **32+ cores and 64 GB+ RAM**
to reach 10–25 tok/s on a 7–14B model. The target host is a low-power Intel
mobile-class CPU. The realistic recommendation at that tier is **Qwen3.5 4B
Q4_K_M** or similar — a 4B model, well below the 8B assumed in `plan.md`.

A 4B model doing long-context, multi-speaker claim attribution is not going to
produce the output quality D5 requires. This does not change the decision — D5
already predicted landing on cloud — but it means the local experiment should be
run with **calibrated expectations and a small time budget**, not as a serious
contender.

**Change:** default `OLLAMA_MODEL=qwen3.5:4b` (realistic for the hardware, and
an honest test) rather than `qwen3:8b` (which would mostly measure swap
thrashing). Keep the port; keep the comparison in Phase 5.

### 16.8 Cloud LLM — use the Batch API

`claude-haiku-4-5` remains the correct tier and is current at **$1/$5 per
million tokens**. The current lineup is Haiku 4.5, Sonnet 5 ($3/$15), Opus 5
($5/$25), Fable 5.1 ($10/$50).

**The change worth making: the Batch API halves both input and output costs**,
and this system is the ideal batch candidate — N1 declares latency tolerance as
a first-class requirement, and analyses are queued work with no waiting user.
Prompt caching (90% off cached input) stacks with it, and the map-reduce design
sends the *same system prompt and roster* across every chunk of an episode,
which is exactly the shape caching rewards.

Revised cost estimate for a 2-hour episode under map-reduce (~8 chunks,
~28k input / ~6k output tokens): roughly **$0.06 standard, ~$0.03 batched**.
At 10 videos/day that is **~$9/month standard, ~$5/month batched** — before
caching. `plan.md`'s $5–15/month estimate holds up.

**Change:** the `AnthropicSummarizer` adapter should submit analysis jobs via
the Batch API, with the synchronous path retained for interactive
`POST /videos/{id}/reanalyze`. Mark the system prompt and roster as cacheable.

### 16.9 pgvector stays — do not reach for VectorChord

pgvector is at 0.8.2. VectorChord (successor to pgvecto.rs) advertises
substantially faster indexing and cheaper storage at scale.

**It is the wrong choice here, and the reason is worth stating so nobody
revisits it.** The corpus is ~3–5k videos/year × ~8 chunks ≈ **40k vectors/year**.
VectorChord's advantages materialize at 100M–1B vectors. At 40k, pgvector's
HNSW index is comprehensively sufficient, and it is the extension packaged
everywhere with the widest operational track record. Choosing the scale-oriented
option here would be the exact failure `plan.md` §2.4 warns about — "several
'correct at scale' choices are wrong here."

**No change.** D9c stands as written.

### 16.10 FastAPI

Current is 0.139.x. Two notes:

- FastAPI added **`app.frontend()`**, which serves an SPA build directory
  directly, no reverse proxy needed. This is a genuine alternative to the
  separate nginx container in §11.2. I am **keeping nginx** — the CSP for the
  YouTube embed (D12b), cache-control for fingerprinted assets, and independent
  frontend rebuilds are the reasons, and none are addressed by `app.frontend()`.
  But it is a legitimate simplification if the deployment ever needs to shrink
  to one container.
- Litestar (2.24) benchmarks roughly 2× FastAPI's throughput via msgspec. At
  this system's request volume — a single operator browsing summaries — that
  difference is unmeasurable. FastAPI's ecosystem and OpenAPI generation (which
  §9 depends on for generated TS types) matter far more. **No change.**

### 16.11 yt-dlp — risk reconfirmed, not reduced

yt-dlp remains the clear leader; `pytube` and the rest are not viable
substitutes for a production ingestion path. But **YouTube changes in August
2026 again caused widespread 403 errors and breakage**, which is a live
demonstration of the risk `plan.md` §13 already lists.

**Strengthened mitigations, promoted from "expected maintenance" to concrete
requirements:**

- Pin the yt-dlp version in `requirements.backend.txt`, but run a **scheduled
  update check** — a broken extractor is usually fixed upstream within days, so
  staleness is a bigger risk than churn here.
  Implemented by `.github/workflows/ytdlp-update-check.yml` (#68): daily, runs
  `scripts/ytdlp_update_check.py`, which goes red with an `::error` naming the
  pinned and latest versions when the pin is behind PyPI, stays silent when it
  is current, and ends green on a PyPI outage. It never edits the pin.
- The `TOOL_FAILURE` class in §8.4 should trigger a **visible alert**, not a
  silent dead-letter. A 403 storm means "go update yt-dlp", and that signal is
  useless if it only lands in a database column.
  Implemented by the planner's `dead_letter_alerts` task (#67): one webhook
  message per class (`TOOL_FAILURE`, `RESOURCE`) with a count, at most one per
  `ALERT_MIN_INTERVAL_SEC`; `jobs.alerted_at` records what was announced.
- Support an optional **cookie file** mount on the transcriber for bot-check
  challenges. Design the config now even if unused.

### 16.12 Summary of changes

Applied to this document:

1. `postgres:18-alpine` (≥18.6)
2. `python:3.14-slim` (host parity; wheel guard in §16.2)
3. `node:24-alpine`
4. React 19.3 + React Compiler; Vite 8
5. `OLLAMA_MODEL=qwen3.5:4b` default
6. Batch API path for the Anthropic adapter
7. Phase 0 becomes a three-way transcription bake-off including Parakeet
8. yt-dlp alerting and cookie-file config

Deliberately **not** changed: FastAPI, pgvector, TanStack Query v5, nginx,
Alembic, psycopg3, and the overall architecture. In each case the newer option
targets a scale or throughput problem this system does not have.

---

## Appendix A — Decision traceability

| `plan.md` decision | Realized in |
|---|---|
| D1 transcript-only | §7.1 — no media pipeline beyond audio |
| D2 subtitles first | §6 `transcript_rank()`, §7.1 auto-caption policy |
| D3 quality-first Whisper | §7.2, `engine_meta` RTF recording, §12 metric |
| D4b LLM speaker inference | §7.4 roster pass, §8.1 closed-set enforcement, C3 |
| D5 summarizer seam | §3 `Summarizer` port, §10 `SUMMARIZER` |
| D6 API key only | §10 — no subscription credential exists in config |
| D6b audio TTL + cap | §6 `media` table, §7.2, planner purge |
| D7 durable transcripts | §6 separation, `analyses` append-only, C1 |
| D8 structured JSON | §8.1 pydantic validation before write |
| D9 map-reduce | §7.3, §7.4 |
| D9b forward-only + backfill | §6 `monitor_from`, §4 priority bands, C4 |
| D9c semantic-search-ready | §6 `transcript_chunks`, `chunk_strategy` |
| D10 Postgres queue | §5, §6 `jobs` |
| D11 four services | §1, §2 dependency rule |
| D12 React SPA | §9, §11.3 `Dockerfile.frontend` — no Node at runtime |
| D12b click-to-seek | §9 `Player`/`Timestamp`, §9.1 player transport |
| D12c SSR render path | §8.3 `GET /videos/{id}/render` |
| D13 auth-additive | §2 route split, §8.3 `require_auth` no-op, §11.4 loopback bind + internal network |
