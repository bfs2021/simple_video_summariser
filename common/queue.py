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

_CLAIM_SQL = """
    UPDATE jobs SET state='running', locked_at=now(), started_at=now(),
                    heartbeat_at=now(), locked_by=%(worker)s, attempts=attempts+1
    WHERE id = (
        SELECT id FROM jobs
        WHERE state='pending' AND kind = ANY(%(kinds)s) AND run_after <= now()
        ORDER BY priority DESC, run_after
        FOR UPDATE SKIP LOCKED
        LIMIT 1)
    RETURNING *
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
        self, kinds: Sequence[str], *, worker: str
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
    def claim(self, kinds: Sequence[str], *, worker: str) -> Iterator[Job | None]:
        job = self._do_claim(kinds, worker)
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

    def _do_claim(self, kinds: Sequence[str], worker: str) -> Job | None:
        with self._conn.transaction(), self._conn.cursor(row_factory=dict_row) as cur:
            cur.execute(_CLAIM_SQL, {"kinds": list(kinds), "worker": worker})
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

    # -- reap_stale ------------------------------------------------------------

    def reap_stale(self, older_than_sec: int | None = None) -> int:
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
                    last_error = %(last_error)s
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
