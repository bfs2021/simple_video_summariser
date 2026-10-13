# Progress

Status of the groomed backlog (GitHub issues #1–#20), worked through per
`_docs/process.md`. An issue is closed only after a QA PASS comment.

## Owner decisions — 2026-10-03

- **Python dependencies approved** and pinned with #39: `fastapi==0.142.2` and
  `uvicorn==0.54.0` (runtime), `httpx==0.28.1` (dev).
- **npm dependencies approved** for #47-#54 (pin exact versions and commit the
  lockfile in #47):
  - Runtime: `react` and `react-dom` 19.3.x, `@tanstack/react-query` 5.x,
    `react-router`.
  - Dev:
    - build and types: `vite` 8.x, `@vitejs/plugin-react`, `typescript`,
      `@types/react`, `@types/react-dom`, `@types/node` 24.x (only if
      `vite.config.ts` needs it);
    - React Compiler: `babel-plugin-react-compiler` 1.x, `@babel/core`,
      `@rolldown/plugin-babel`, `@types/babel__core`;
    - tests: `vitest`, `jsdom`, `@testing-library/react`;
    - API types: `openapi-typescript` (#46).
- **Router:** `react-router` (#122).
- **Player transport:** direct `postMessage` to the `enablejsapi=1` embed. The
  `iframe_api` script is discarded, so the CSP is unchanged (#48;
  architecture.md §9.1).
- **React Compiler:** wired through Babel. The native `compiler: true` option
  (`oxc-transform-react`) is experimental (architecture.md §16.4).
- **Testing policy:** no repeat-run loops ("passes N runs in a row", `seq N`,
  `pytest-repeat`). Find the root cause, wait on the exact condition with a
  bounded deadline, and force races deterministically. One suite run is the
  bar (`_docs/testing-guidelines.md`, "Flaky or racy behaviour"). This
  supersedes the loop criteria in #12, #74 and #75: the Postgres readiness
  wait is now 60 s with an explicit failure, and
  `tests/common/test_queue_concurrency.py` forces the enqueue
  commit/rollback races on autocommit and transactional connections, plus a
  claim race (verified to fail against the pre-#73 `enqueue`).
- **Still open:** the visual direction (#124). `_docs/design-system.md` is
  empty.

## Checkpoint 4 — 2026-09-28, batch 2 (#21-#40) done except #39/#40; #41-#60 groomed

Closed after QA PASS since checkpoint 3: #24 Ollama adapter (defines the
shared `Summarizer` port), #25 Anthropic adapter, #26 RSS feed, #27 catalog,
#28 ingest handler, #29 transcribe handler, #30 analyze handler, #31 CLI,
#32 transcriber entrypoint, #33 analyzer entrypoint, #34 polling, #35 reaper,
#37 re-analysis sweep, #38 planner entrypoint, #55 backend image, #56
speech-to-text image. Several needed a second QA round (#26, #27, #31: real
recordings instead of hand-written fixtures; #31: whisper check before audio
work).

Batch 2 remainder: #39 API skeleton and #40 submission endpoints are groomed
but held for the owner's approval of `fastapi`, `uvicorn`, `httpx`.

Batch 3 (#41-#60) is groomed (`groomed` label, follow-ups #109-#142 filed).
Only #55 and #56 could be built now. Blocked on the owner: the API
dependencies (#39-#46, #60), npm dependencies and a router/visual-direction
decision for the frontend (#47-#54), #57-#59 (need #47/#39/#56).

## Checkpoint 3 — 2026-09-28, batch 2 (#21-#40): 5 closed

All 20 issues of batch 2 (#21-#40) were groomed first; follow-ups #80-#105
were filed by the PMs. Closed after QA PASS:

| Issue | Title | Commits | QA |
|---|---|---|---|
| #13 | Worker runtime (carried over) | f924c12 | PASS (2nd run, after reconnect / liveness / file-scope fixes) |
| #21 | Transcript chunker | a0fc4e4, f4c20a9 | PASS (2nd run, after the 1e20 start fix; criterion wording amended by PM) |
| #22 | LLM output schema and validation | 6afffc7 | PASS |
| #23 | Prompt templates v1 | a5be577 | PASS |
| #36 | Planner: audio retention | f5ef376 | PASS |

In flight: #26 feed adapter, #35 reaper, #28 ingest handler (engineers);
Summarizer port shape being reconciled across #24/#25/#30 before #24 starts.
New: #106 heartbeat connection re-creation (ungroomed).

Held for the owner: #39/#40 need `fastapi`, `uvicorn`, `httpx` approved.

## Checkpoint 2 — 2026-09-27, 17 of 20 closed

Closed after QA PASS since checkpoint 1: #6 logging, #7 database bootstrap
(2nd QA, after the dev-network fix), #8 core schema, #9 jobs, #10 media,
#11 queue, #14 repository layer (2nd QA, after the test-marker fix),
#16 metadata adapter, #17 VTT parser (2nd QA), #18 subtitles, #19 audio
(2nd QA, after the real-ffmpeg ENOSPC fix), #20 faster-whisper adapter.
Out-of-backlog bug #71 (Alembic disabling loggers) groomed, fixed, closed.

Still open:

- #3 bake-off harness: passes everything except the deploy-host run;
  containerizing it is #76 (being groomed), deploy-host preparation is a
  separate follow-up
- #12 queue concurrency suite: committed (76df71e); blocked on #73
  (enqueue race, fix in progress) and #74 (flaky testcontainers first
  connect, fix in progress)
- #13 worker runtime: committed (8c9e38a), in QA

New follow-ups: #72 random test order, #73 enqueue race, #74 fixture
readiness, #75 more enqueue races, #77 clear error when Docker is down.

Dependencies added since checkpoint 1 (owner-approved): `alembic==1.20.0`,
`psycopg[binary]==3.3.6`, `testcontainers[postgres]==4.15.0` (dev),
`faster-whisper==1.2.1` (requirements.whisper.txt).

## Checkpoint 1 — 2026-09-27, 5 of 20 closed

| Issue | Title | Commits | QA |
|---|---|---|---|
| #1 | Project skeleton | 4de9e43 | PASS (2nd run, after `python3.14-venv` was installed) |
| #2 | Subtitle coverage script | 6cf2d4a, b5aab6b | PASS |
| #4 | faster-whisper wheels on 3.14 | a722cd7 | PASS |
| #5 | Configuration module | ebd0856 | PASS |
| #15 | Error taxonomy | f8afd58 | PASS |

Dependencies added (owner-approved): `yt-dlp==2026.8.19`,
`pydantic-settings==2.15.0`, `structlog==26.1.0` (with #6, in QA).

### Phase 0 findings

- **O7 resolved (#4):** faster-whisper 1.2.1 and its whole tree (ctranslate2
  4.8.2, av 18.1.0, tokenizers 0.23.2, onnxruntime 1.30.0) install from wheels
  on `python:3.14-slim`; no 3.13 transcriber image needed. The single
  `--platform manylinux_2_28_x86_64` download check gives a false negative;
  see architecture §16.2.
- **Subtitle coverage (#2),** 9 owner-chosen channels, 15 RSS entries each:
  manual 6 videos (1.95 h), auto-only 64 (48.72 h), none 0, error 2
  (transient), 63 Shorts excluded. Speech-to-text load is ~0 by default and
  ~48.7 h per this window with `PREFER_WHISPER=1`. Input for #64.
- **Phase 0 decisions recorded (#64), owner, 2026-10-13** (architecture §15):
  - **O1 — STT deferred from Phase 1; subtitles-only.** Coverage is ~100%
    (0 videos with no captions), so the default does ~0 h of speech-to-text.
    The #20 adapter stays built but off (`PREFER_WHISPER` unset); revisit for
    STT quality or an uncaptioned channel.
  - **O2 — `faster-whisper large-v3` (int8), Option A / D3 default**, as the
    Phase-1 engine. Provisional, not measured: the #3 deploy-host bake-off is
    still owed, but O1 means no Phase-1 flow depends on the engine. The
    `Transcriber` port keeps B (`large-v3-turbo`) a config flip and C
    (Parakeet) a one-adapter add (§16.6). D4 diarization revisit untriggered.
  - **O3 — `AUDIO_TTL_DAYS=30`.** With STT deferred, audio fetch is ~0 by
    default and `AUDIO_MAX_GB` (20) is the binding cap, not the TTL.
  - **O7 — resolved (see above):** no 3.13 transcriber split; #56 derives
    from the 3.14 backend.

### In flight

- #6 structured logging — committed 2dd0695, in QA
- #17 WebVTT parser — QA FAIL (leading-zero numeric entity raises
  `ValueError`; whitespace-only input returns `[]`), fix in progress
- #16 metadata adapter, #7 database bootstrap — in implementation

### Follow-ups filed (not groomed yet)

- #69 map yt-dlp "rate-limited by YouTube" to `RATE_LIMITED`
- #70 config secrets leak via `ValidationError.errors()`; TID251 rule untested

### Blocked or owner-dependent

- #3 bake-off must run on the deploy host with real clips
- #20 speech-to-text adapter: unblocked by #4; also check #64 first
