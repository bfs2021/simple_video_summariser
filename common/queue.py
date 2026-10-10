"""Job queue port and Postgres implementation (issue #11).

Implements the job lifecycle from architecture.md §5-§6 and the error
taxonomy's retry policy from §8.4 (``common/errors.py``, #15):

- ``enqueue`` is idempotent per active ``(video_id, kind, dedupe_key)``,
  correction C1 (§0).
- ``claim`` uses the ``SKIP LOCKED`` claim query from §5 and commits the
  claim *before* yielding the job, so a handler never runs inside an open
  transaction (a multi-hour ``transcribe`` job must not hold a row lock).
- The outcome of the ``with`` block - clean exit, a classified exception,
  ``Defer`` or a cancellation - decides the next state and, for retries, the
  backoff (exponential with jitter, floored by the error class's policy).
- ``heartbeat`` and ``reap_stale`` implement §5's "Reaping (C2)".

``JobQueue`` and ``Job`` live here rather than in ``common/models.py``
because architecture.md §2 places ``queue.py`` as "JobQueue port + Postgres
adapter" and wins over §3 where the two disagree (issue #11 constraints).

All timestamps are computed by the database's own ``now()``: the only
exception is ``Defer(until=...)``, whose ``until`` is a caller-supplied
instant, not derived from the app clock. This is what keeps clock skew
between containers from misordering jobs. Jitter is drawn from an injectable
``random.Random`` so tests are deterministic.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Iterator, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Literal, Protocol

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from common.config import Settings, get_settings
from common.errors import Cancelled, Defer, classify, format_error, policy

_logger = logging.getLogger(__name__)

#: The only kinds enqueue() accepts (architecture.md §4). ``notify`` is a v2
#: kind, declared but not created (architecture.md §6), so it is deliberately
#: excluded here. Public so #44's ``routes_ops.py`` validates against the
#: same set instead of a duplicated literal.
VALID_KINDS = frozenset({"ingest", "transcribe", "analyze"})
_VALID_KINDS = VALID_KINDS

#: Priority bands (architecture.md §4). Claims run ``priority DESC``.
PRIORITY_INTERACTIVE = 10  # POST /videos: a human is waiting
PRIORITY_NORMAL = 0  # RSS-discovered new videos
PRIORITY_REANALYSIS = -5  # re-analysis sweep after a PROMPT_VERSION change (C4)
PRIORITY_BACKFILL = -10  # historical channel backfill (D9b, C4)


def analyze_dedupe_key(prompt_version: str, summarizer_name: str) -> str:
    """The ``analyze`` job dedupe key, ``"<prompt_version>:<summarizer_name>"`` (C1).

    The one place it is built; ingest, transcribe and the re-analysis sweep all use it.
    Empty parts or parts containing ``:`` raise ``ValueError``, so two different
    pairs can never produce the same key.
    """
    for label, part in (("prompt_version", prompt_version), ("summarizer_name", summarizer_name)):
        if not part or ":" in part:
            raise ValueError(f"{label} must be non-empty and contain no ':', got {part!r}")
    return f"{prompt_version}:{summarizer_name}"


#: Max attempts per kind, before the error class's own max is applied
#: (architecture.md §5, Backoff). ``transcribe``'s comes from settings
#: instead, since it is operator-tunable (MAX_ATTEMPTS_TRANSCRIBE).
_KIND_MAX_ATTEMPTS = {"ingest": 4, "analyze": 4}

_BASE_BACKOFF_SEC = 5 * 60  # 5 min
_MAX_BACKOFF_SEC = 6 * 60 * 60  # 6 h
_JITTER_FRACTION = 0.10

_REAP_LAST_ERROR = "worker lost: heartbeat stale"

_CLAIM_SQL_TEMPLATE = """
    UPDATE jobs SET state='running', locked_at=now(), started_at=now(),
                    heartbeat_at=now(), locked_by=%(worker)s, attempts=attempts+1
    WHERE id = (
        SELECT id FROM jobs
        WHERE state='pending' AND kind = ANY(%(kinds)s) AND run_after <= now(){key_filter}
        ORDER BY priority DESC, run_after
        FOR UPDATE SKIP LOCKED
        LIMIT 1)
    RETURNING *
"""
_CLAIM_SQL = _CLAIM_SQL_TEMPLATE.format(key_filter="")
#: ``claim(dedupe_key=...)`` (#100): an exact, bound-parameter match, never a pattern.
_CLAIM_SQL_KEYED = _CLAIM_SQL_TEMPLATE.format(key_filter=" AND dedupe_key = %(dedupe_key)s")

#: #100. Pending ``analyze`` jobs whose key is ``<version>:<summarizer>`` with a
#: version other than ``%(prompt_version)s``. ``FOR UPDATE SKIP LOCKED`` leaves a
#: row that is being claimed for the next tick; the regex leaves malformed keys
#: alone (``analyze_dedupe_key`` forbids ``:`` and empty parts in either half).
#: Per (video, target key) the highest ``priority`` (then lowest ``id``) is
#: re-keyed, unless the target already has a pending/running/dead job or the
#: video already has an analysis at the version; everything else is superseded.
#: One statement, so one transaction: the two UPDATEs touch disjoint rows.
_STALE_ANALYZE_SQL = r"""
    WITH stale AS (
        SELECT id, video_id, priority,
               %(prompt_version)s::text || ':' || split_part(dedupe_key, ':', 2) AS target_key
        FROM jobs
        WHERE kind = 'analyze' AND state = 'pending'
          AND dedupe_key ~ '^[^:]+:[^:]+$'
          AND split_part(dedupe_key, ':', 1) <> %(prompt_version)s
        FOR UPDATE SKIP LOCKED
    ),
    ranked AS (
        SELECT s.id, s.target_key,
               row_number() OVER (PARTITION BY s.video_id, s.target_key
                                  ORDER BY s.priority DESC, s.id) = 1
               AND NOT EXISTS (
                   SELECT 1 FROM jobs j
                   WHERE j.video_id = s.video_id AND j.kind = 'analyze'
                     AND j.dedupe_key = s.target_key
                     AND j.state IN ('pending', 'running', 'dead'))
               AND NOT EXISTS (
                   SELECT 1 FROM analyses a
                   WHERE a.video_id = s.video_id
                     AND a.prompt_version = %(prompt_version)s) AS rekey
        FROM stale s
    ),
    rekeyed AS (
        UPDATE jobs j SET dedupe_key = r.target_key
        FROM ranked r WHERE j.id = r.id AND r.rekey
        RETURNING j.id
    ),
    superseded AS (
        UPDATE jobs j SET state = 'dead', finished_at = now(), error_class = NULL,
                          last_error = %(message)s
        FROM ranked r WHERE j.id = r.id AND NOT r.rekey
        RETURNING j.id
    )
    SELECT (SELECT count(*) FROM rekeyed), (SELECT count(*) FROM superseded)
"""

#: #44. ``NOT EXISTS`` only looks at active-index states, matching the
#: superseded-by-a-later-attempt case the AC describes (an auto-caption
#: fallback ingest, a later resubmission); a finished (done) later job also
#: counts, since ``id > j.id`` alone is enough once ``state = 'dead'`` has
#: already excluded this very row.
#: ``candidate`` locks and captures the pre-update ``attempts`` (the route's
#: log line needs "the previous attempts", which the plain ``UPDATE ...
#: RETURNING *`` below would otherwise only show as the new value, 0) in the
#: same statement as the write, so there is still no read-then-write gap.
_RETRY_DEAD_SQL = """
    WITH candidate AS (
        SELECT id, attempts AS prev_attempts
        FROM jobs j
        WHERE j.id = %(id)s AND j.state = 'dead'
          AND NOT EXISTS (
              SELECT 1 FROM jobs newer
              WHERE newer.video_id = j.video_id AND newer.kind = j.kind
                AND newer.dedupe_key = j.dedupe_key AND newer.id > j.id
          )
        FOR UPDATE
    )
    UPDATE jobs j SET state='pending', attempts=0, run_after=now(),
                      finished_at=NULL, locked_by=NULL, locked_at=NULL,
                      heartbeat_at=NULL
    FROM candidate c
    WHERE j.id = c.id
    RETURNING j.*, c.prev_attempts
"""


@dataclass(frozen=True, slots=True)
class StaleAnalyzeResult:
    """What ``PostgresQueue.resolve_stale_analyze`` did (#100)."""

    rekeyed: int
    superseded: int


@dataclass(frozen=True, slots=True)
class Job:
    """One row of ``jobs``, as returned by ``claim`` (architecture.md §6)."""

    id: int
    video_id: str
    kind: str
    dedupe_key: str
    state: str
    priority: int
    payload: dict[str, Any]
    attempts: int
    last_error: str | None
    error_class: str | None
    run_after: datetime
    locked_by: str | None
    locked_at: datetime | None
    heartbeat_at: datetime | None
    finished_at: datetime | None
    created_at: datetime
    #: Latest claim (#60); ``NULL`` for rows from before the column existed.
    #: No transition clears it.
    started_at: datetime | None = None
    #: When the planner announced this dead-letter to the operator (#67).
    alerted_at: datetime | None = None


def _job_from_row(row: dict[str, Any]) -> Job:
    return Job(**row)


@dataclass(frozen=True, slots=True)
class RetryOutcome:
    """Result of ``PostgresQueue.retry_dead`` (#44); the route maps ``status``
    to a response.

    - ``"retried"``: ``job`` is the now-``pending`` job (same id);
      ``prev_attempts`` is its attempt count just before the retry, for the
      route's log line.
    - ``"not_found"``: no job has that id.
    - ``"not_dead"``: the job exists but is ``state`` (not ``dead``).
    - ``"superseded"``: a newer job exists for the same
      ``(video_id, kind, dedupe_key)``; ``newer_job_id`` is its id
      (always set, whether the initial check or a lost race against a
      concurrent ``enqueue`` caught it).
    """

    status: Literal["retried", "not_found", "not_dead", "superseded"]
    job: Job | None = None
    state: str | None = None
    newer_job_id: int | None = None
    prev_attempts: int | None = None


class JobQueue(Protocol):
    """The queue port (architecture.md §3; §2 wins on where it lives)."""

    def enqueue(
        self,
        kind: str,
        video_id: str,
        *,
        dedupe_key: str = "default",
        payload: dict[str, Any] | None = None,
        priority: int = 0,
        run_after: datetime | None = None,
    ) -> int | None: ...

    def claim(
        self, kinds: Sequence[str], *, worker: str, dedupe_key: str | None = None
    ) -> AbstractContextManager[Job | None]: ...

    def heartbeat(self, job_id: int, worker: str) -> bool: ...

    def reap_stale(self, older_than_sec: int | None = None) -> int: ...


class PostgresQueue:
    """``JobQueue`` backed by the ``jobs`` table (architecture.md §5-§6).

    Receives a connection; never opens its own (architecture.md §2's
    "Database connection" acceptance criterion - see ``common/db.py``).
    """

    def __init__(
        self,
        conn: psycopg.Connection[Any],
        *,
        settings: Settings | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self._conn = conn
        self._settings = settings if settings is not None else get_settings()
        self._rng = rng if rng is not None else random.Random()

    def close(self) -> None:
        """Close the underlying connection.

        Not part of the ``JobQueue`` Protocol (issue #106): only a caller that
        holds a ``PostgresQueue`` - the worker discarding a broken heartbeat
        connection - uses it.
        """
        self._conn.close()

    # -- enqueue -----------------------------------------------------------

    def enqueue(
        self,
        kind: str,
        video_id: str,
        *,
        dedupe_key: str = "default",
        payload: dict[str, Any] | None = None,
        priority: int = 0,
        run_after: datetime | None = None,
    ) -> int | None:
        if kind not in _VALID_KINDS:
            raise ValueError(
                f"unknown job kind {kind!r}; expected one of {sorted(_VALID_KINDS)}"
            )
        with self._conn.transaction(), self._conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO jobs (video_id, kind, dedupe_key, payload, priority, run_after)
                VALUES (%(video_id)s, %(kind)s, %(dedupe_key)s, %(payload)s,
                        %(priority)s, COALESCE(%(run_after)s, now()))
                ON CONFLICT (video_id, kind, dedupe_key)
                    WHERE state IN ('pending', 'running')
                DO NOTHING
                RETURNING id
                """,
                {
                    "video_id": video_id,
                    "kind": kind,
                    "dedupe_key": dedupe_key,
                    "payload": Jsonb(payload if payload is not None else {}),
                    "priority": priority,
                    "run_after": run_after,
                },
            )
            row = cur.fetchone()
        return int(row[0]) if row is not None else None

    # -- claim ---------------------------------------------------------------

    @contextmanager
    def claim(
        self, kinds: Sequence[str], *, worker: str, dedupe_key: str | None = None
    ) -> Iterator[Job | None]:
        """Claim the best pending job of ``kinds``, or yield ``None``.

        With ``dedupe_key`` only a job whose key equals it exactly is eligible
        (#100): an analyzer passes ``analyze_dedupe_key(PROMPT_VERSION, name)``
        so it never claims a job it would have to hand back. The filter applies
        to every kind in ``kinds``, so pass a key only for a single-kind claim.
        """
        job = self._do_claim(kinds, worker, dedupe_key)
        if job is None:
            yield None
            return
        try:
            yield job
        except Defer as defer:
            self._resolve(
                job,
                worker,
                """
                UPDATE jobs SET state='pending', run_after=%(until)s,
                       attempts=attempts-1, locked_by=NULL, locked_at=NULL
                WHERE id=%(id)s AND locked_by=%(worker)s AND state='running'
                """,
                {"until": defer.until},
            )
        except (Cancelled, KeyboardInterrupt, SystemExit):
            self._resolve(
                job,
                worker,
                """
                UPDATE jobs SET state='pending', run_after=now(),
                       attempts=attempts-1, locked_by=NULL, locked_at=NULL
                WHERE id=%(id)s AND locked_by=%(worker)s AND state='running'
                """,
                {},
            )
            raise
        except Exception as exc:  # noqa: BLE001
            # Intentionally blind: this is the queue's safety net. Every
            # failure a handler can raise must be classified and recorded
            # (architecture.md 8.4) rather than crashing the worker loop -
            # Cancelled/KeyboardInterrupt/SystemExit and Defer, which do
            # propagate, are already peeled off above.
            self._fail(job, worker, exc)
        else:
            self._resolve(
                job,
                worker,
                """
                UPDATE jobs SET state='done', finished_at=now(),
                       locked_by=NULL, locked_at=NULL
                WHERE id=%(id)s AND locked_by=%(worker)s AND state='running'
                """,
                {},
            )

    def _do_claim(
        self, kinds: Sequence[str], worker: str, dedupe_key: str | None = None
    ) -> Job | None:
        params: dict[str, Any] = {"kinds": list(kinds), "worker": worker}
        sql = _CLAIM_SQL
        if dedupe_key is not None:
            params["dedupe_key"] = dedupe_key
            sql = _CLAIM_SQL_KEYED
        with self._conn.transaction(), self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
        return _job_from_row(row) if row is not None else None

    def fail_job(self, job: Job, worker: str, exc: BaseException) -> None:
        """Record ``exc`` as a failed attempt of a job this worker holds (#66).

        The same outcome ``claim`` gives a handler that raised ``exc``: retry
        with backoff, or dead-letter at max attempts. For a caller outside
        the handler's own ``with queue.claim(...)`` block, namely the worker's
        runtime-limit watchdog, which acts while the handler is still wedged.
        Guarded by ``locked_by`` like every other outcome, so a job already
        reaped and reclaimed is left alone.
        """
        self._fail(job, worker, exc)

    def _fail(self, job: Job, worker: str, exc: BaseException) -> None:
        error_class = classify(exc)
        pol = policy(error_class)
        last_error = format_error(exc)

        dead = not pol.retry
        if not dead:
            kind_max = self._kind_max_attempts(job.kind)
            class_max = pol.max_attempts if pol.max_attempts is not None else kind_max
            effective_max = min(kind_max, class_max)
            dead = job.attempts >= effective_max

        if dead:
            self._resolve(
                job,
                worker,
                """
                UPDATE jobs SET state='dead', finished_at=now(),
                       locked_by=NULL, locked_at=NULL,
                       last_error=%(last_error)s, error_class=%(error_class)s
                WHERE id=%(id)s AND locked_by=%(worker)s AND state='running'
                """,
                {"last_error": last_error, "error_class": error_class.value},
            )
            return

        retry_after_sec = getattr(exc, "retry_after_sec", None)
        delay_sec = self._compute_backoff_sec(
            job.attempts, pol.min_backoff, retry_after_sec
        )
        self._resolve(
            job,
            worker,
            """
            UPDATE jobs SET state='pending',
                   run_after = now() + (%(delay_sec)s * interval '1 second'),
                   locked_by=NULL, locked_at=NULL,
                   last_error=%(last_error)s, error_class=%(error_class)s
            WHERE id=%(id)s AND locked_by=%(worker)s AND state='running'
            """,
            {
                "delay_sec": delay_sec,
                "last_error": last_error,
                "error_class": error_class.value,
            },
        )

    def _compute_backoff_sec(
        self,
        attempts: int,
        min_backoff: timedelta | None,
        retry_after_sec: int | None,
    ) -> float:
        # int.__pow__ types as Any for a non-literal exponent (a negative one
        # would produce a float), so pin the annotation explicitly.
        base: int = min(_BASE_BACKOFF_SEC * (2 ** (attempts - 1)), _MAX_BACKOFF_SEC)
        jitter = self._rng.uniform(-_JITTER_FRACTION, _JITTER_FRACTION)
        delay = base * (1 + jitter)
        if min_backoff is not None:
            delay = max(delay, min_backoff.total_seconds())
        if retry_after_sec is not None:
            delay = max(delay, float(retry_after_sec))
        return delay

    def _kind_max_attempts(self, kind: str) -> int:
        if kind == "transcribe":
            return self._settings.MAX_ATTEMPTS_TRANSCRIBE
        return _KIND_MAX_ATTEMPTS.get(kind, 4)

    def _resolve(
        self, job: Job, worker: str, sql: str, params: dict[str, Any]
    ) -> None:
        with self._conn.transaction(), self._conn.cursor() as cur:
            cur.execute(sql, {**params, "id": job.id, "worker": worker})
            if cur.rowcount == 0:
                _logger.warning(
                    "job outcome update matched no rows; it was likely "
                    "reaped and reclaimed by another worker",
                    extra={"job_id": job.id, "worker": worker, "kind": job.kind},
                )

    # -- retry_dead ------------------------------------------------------------

    def retry_dead(self, job_id: int) -> RetryOutcome:
        """Put a ``dead`` job back to ``pending`` with ``attempts = 0`` (#44).

        Same id, same ``priority``/``payload``/``dedupe_key``/``created_at``;
        ``error_class`` and ``last_error`` are kept so the Ops view still
        shows why it died. Refuses (``RetryOutcome.status``) a job that is
        not ``dead``, one superseded by a newer job for the same
        ``(video_id, kind, dedupe_key)`` (higher id, any state - only the
        newest job for a key may be revived), or one that no longer exists.
        The state check and the write are one conditional ``UPDATE ...
        WHERE ... NOT EXISTS (...) RETURNING``, never a read then a write.
        The exception must propagate out of ``self._conn.transaction()``'s
        own ``with`` block to roll back and leave the connection ``IDLE``
        (as in ``enqueue``'s own ``UniqueViolation`` case) before this method
        catches it: a concurrent ``enqueue`` that still wins the race raises
        ``UniqueViolation``, reported the same way as a pre-existing newer
        job (including its id), never a 500. If no newer job can be found at
        all, ``RuntimeError`` is raised rather than a ``superseded`` outcome
        with no id.
        """
        try:
            with self._conn.transaction(), self._conn.cursor(row_factory=dict_row) as cur:
                cur.execute(_RETRY_DEAD_SQL, {"id": job_id})
                row = cur.fetchone()
        except psycopg.errors.UniqueViolation:
            # Lost the race to a concurrent enqueue. The transaction rolled
            # back, so the job is still ``dead``; fall through to the same
            # lookup the pre-existing-newer-job path uses to name the winner.
            row = None

        if row is not None:
            prev_attempts = row.pop("prev_attempts")
            return RetryOutcome(
                "retried", job=_job_from_row(row), prev_attempts=prev_attempts
            )

        existing = self._conn.execute(
            "SELECT video_id, kind, dedupe_key, state FROM jobs WHERE id = %s", (job_id,)
        ).fetchone()
        if existing is None:
            return RetryOutcome("not_found")
        video_id, kind, dedupe_key, state = existing
        if state != "dead":
            return RetryOutcome("not_dead", state=state)

        newer = self._conn.execute(
            """
            SELECT id FROM jobs
            WHERE video_id = %(video_id)s AND kind = %(kind)s
              AND dedupe_key = %(dedupe_key)s AND id > %(id)s
            ORDER BY id DESC LIMIT 1
            """,
            {"video_id": video_id, "kind": kind, "dedupe_key": dedupe_key, "id": job_id},
        ).fetchone()
        if newer is None:
            # "superseded" without a newer job to name would be a lie (and a
            # ``job_id: null`` in the 409). The only unique index on jobs is
            # the active-key one, so this means the winner vanished (e.g. a
            # cascade delete) between the failure and this lookup: fail loudly
            # rather than invent an answer.
            raise RuntimeError(
                f"retry_dead({job_id}): superseded but no newer job found"
            )
        return RetryOutcome("superseded", newer_job_id=int(newer[0]))

    # -- heartbeat -----------------------------------------------------------

    def heartbeat(self, job_id: int, worker: str) -> bool:
        with self._conn.transaction(), self._conn.cursor() as cur:
            cur.execute(
                """
                UPDATE jobs SET heartbeat_at = now()
                WHERE id = %(id)s AND state = 'running' AND locked_by = %(worker)s
                """,
                {"id": job_id, "worker": worker},
            )
            return cur.rowcount > 0

    def force_pending(self, job_id: int, priority: int) -> bool:
        """Make a ``pending`` job a forced one at no less than ``priority`` (#101).

        Merges ``{"force": true}`` into its payload (other keys stay) and sets
        ``priority = GREATEST(priority, <priority>)``, so a job above it is
        never lowered. One ``UPDATE`` whose own ``WHERE`` requires
        ``state = 'pending'``: a job claimed or finished in the meantime
        matches no row, and the answer is ``False`` with nothing written.
        PostgresQueue only, not on the ``JobQueue`` Protocol.
        """
        with self._conn.transaction(), self._conn.cursor() as cur:
            cur.execute(
                """
                UPDATE jobs
                SET payload = payload || '{"force": true}'::jsonb,
                    priority = GREATEST(priority, %(priority)s)
                WHERE id = %(id)s AND state = 'pending'
                """,
                {"id": job_id, "priority": priority},
            )
            return cur.rowcount > 0

    # -- reap_stale ------------------------------------------------------------

    def reap_stale(self, older_than_sec: int | None = None) -> int:
        """Return stale ``running`` jobs to ``pending``, or dead-letter them.

        Clears ``error_class`` to NULL on every job it touches, whether the
        outcome is ``pending`` or ``dead``. A NULL ``error_class`` together
        with ``last_error = 'worker lost: heartbeat stale'`` means "lost
        heartbeat, cause unknown" - distinct from any non-NULL class a
        handler-raised exception left via ``_fail`` (issue #114).
        """
        threshold = (
            self._settings.REAP_AFTER_SEC if older_than_sec is None else older_than_sec
        )
        if threshold <= 0:
            # A zero threshold would reap every running job.
            raise ValueError(f"older_than_sec must be positive, got {threshold}")
        # One statement, so one transaction. Candidates are locked with
        # SKIP LOCKED: a row an in-flight heartbeat or outcome write holds is
        # left for the next pass, and a row whose heartbeat committed before
        # we got its lock is re-checked against the WHERE by Postgres. The
        # outer WHERE repeats the staleness test for the same reason.
        with (
            self._conn.transaction(),
            self._conn.cursor(row_factory=dict_row) as cur,
        ):
            cur.execute(
                """
                WITH stale AS (
                    SELECT id, video_id, kind, attempts, locked_by FROM jobs
                    WHERE state = 'running'
                      AND (heartbeat_at IS NULL
                           OR heartbeat_at < now() - (%(threshold)s * interval '1 second'))
                    ORDER BY id
                    FOR UPDATE SKIP LOCKED)
                UPDATE jobs j SET
                    state = CASE WHEN j.attempts >= CASE j.kind
                                          WHEN 'transcribe' THEN %(max_transcribe)s
                                          ELSE %(max_default)s END
                                 THEN 'dead' ELSE 'pending' END,
                    finished_at = CASE WHEN j.attempts >= CASE j.kind
                                          WHEN 'transcribe' THEN %(max_transcribe)s
                                          ELSE %(max_default)s END
                                 THEN now() ELSE j.finished_at END,
                    locked_by = NULL, locked_at = NULL,
                    last_error = %(last_error)s, error_class = NULL
                FROM stale
                WHERE j.id = stale.id AND j.state = 'running'
                  AND (j.heartbeat_at IS NULL
                       OR j.heartbeat_at < now() - (%(threshold)s * interval '1 second'))
                RETURNING j.id, stale.video_id, stale.kind, stale.attempts,
                          stale.locked_by AS lost_worker, j.state AS outcome
                """,
                {
                    "threshold": threshold,
                    "max_transcribe": self._kind_max_attempts("transcribe"),
                    "max_default": self._kind_max_attempts("ingest"),
                    "last_error": _REAP_LAST_ERROR,
                },
            )
            reaped = cur.fetchall()
        for row in reaped:
            _logger.warning(
                "queue.job_reaped",
                extra={
                    "job_id": row["id"],
                    "video_id": row["video_id"],
                    "kind": row["kind"],
                    "locked_by": row["lost_worker"],
                    "attempts": row["attempts"],
                    "outcome": row["outcome"],
                },
            )
        return len(reaped)

    # -- resolve_stale_analyze -------------------------------------------------

    def resolve_stale_analyze(self, prompt_version: str) -> StaleAnalyzeResult:
        """Re-key or supersede pending ``analyze`` jobs left on another PROMPT_VERSION (#100).

        Only ``pending`` ``analyze`` jobs whose key is ``<version>:<summarizer>``
        with a version *different from* ``prompt_version`` are touched (not
        "older": versions have no order, and a rollback works the same way).
        Per video and target key, one job is re-keyed in place to
        ``<prompt_version>:<its own summarizer>``, keeping its id, priority,
        payload, attempts and run_after, unless the target already has a
        pending, running or dead job (dead work is never revived) or the video
        already has an analysis at ``prompt_version`` (whatever the model). The
        other matching jobs are marked ``dead`` with ``last_error`` "superseded:
        ..." and keep their old key for the ops view. A row being claimed right
        now is skipped (``SKIP LOCKED``) and handled on the next call. One
        statement. PostgresQueue only, not on the ``JobQueue`` Protocol.
        """
        if not prompt_version or ":" in prompt_version:
            raise ValueError(
                f"prompt_version must be non-empty and contain no ':', got {prompt_version!r}"
            )
        with self._conn.transaction(), self._conn.cursor() as cur:
            cur.execute(
                _STALE_ANALYZE_SQL,
                {
                    "prompt_version": prompt_version,
                    "message": f"superseded: PROMPT_VERSION is now {prompt_version}",
                },
            )
            row = cur.fetchone()
        assert row is not None
        return StaleAnalyzeResult(rekeyed=int(row[0]), superseded=int(row[1]))
