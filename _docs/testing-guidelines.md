# Testing guidelines

Per-layer approach is in architecture.md §13 — follow it. These rules apply on top.

## Workflow

- Write the test first, from the issue's acceptance criteria. Run it and see it
  fail on an assertion (not an ImportError), then implement until it passes.
- Exempt: Phase 0 measurement scripts (tasks 2–4).

## What to cover

- Boundaries: empty input, exactly-at-limit, one-past-limit (e.g. transcript
  shorter than / equal to one chunk window, attempts == max, last page).
- Invariants over many inputs: use Hypothesis property tests (chunker, VTT parser).
- Untrusted input, at trust boundaries only:
  - video IDs/URLs before they reach yt-dlp: accept only 11 chars of
    `[A-Za-z0-9_-]`, reject anything else. A leading `-` is valid (e.g.
    `-wNyEUrxzFU`), so never pass a bare ID - always pass the full watch URL
    so yt-dlp cannot read the ID as an option
  - API query/path params (search `q`, pagination, backfill limit)
  - LLM output: invented speakers, malformed JSON, code fences, prompt text
    echoed from the transcript
- Failure paths: every error class in `common/errors.py` maps to the right retry
  behaviour; a raising handler releases its job.
- Resource hygiene: temp files and audio removed, models loaded lazily, list
  endpoints bounded. (Memory growth is watched via metrics, not unit tests.)

## Rules

- No network: recorded yt-dlp fixtures, `FakeSummarizer`; never a real API key.
- Needs Postgres → `@pytest.mark.integration`; everything else runs under
  `pytest -m "not integration"`.
- Deterministic: no sleep, inject the clock, no test-order
  dependence. The suite runs shuffled; replay a failing order with
  `pytest --shuffle-seed=<N>` using the seed printed by the run.
- Test via public interfaces; prefer fakes over mocking internals.
- `tests/` mirrors the package layout; name tests after the behaviour.

## Frontend

The rules above (test first, cover boundaries, treat untrusted input at trust
boundaries, no network, deterministic, no repeat-run loops) apply unchanged to
`web/`. Only the frontend specifics are listed here.

- **Naming picks the environment.** A test that renders DOM is named
  `*.dom.test.tsx` (or `.ts`) and runs in jsdom with `src/test/setup.ts`.
  Everything else stays `*.test.ts(x)` and runs in Node. The extension never
  decides: `App.test.tsx` is a `.tsx` file that runs in Node.
- **Running.** `npm test` runs once and exits (`vitest run`), from `web/`, in
  a login shell: `bash -lc 'cd ~/projects/simple_video_summary/web && npm test'`.
  Never bare `vitest`, no watch mode. Frontend tests may run shuffled; use
  `npm test -- --sequence.shuffle --sequence.seed=<N>` and report the seed.
- **Time.** Use `vi.useFakeTimers()` and advance explicitly instead of
  waiting. Testing Library's `waitFor` only detects Jest's fake timers, so
  advance with `vi.advanceTimersByTimeAsync` first, start `waitFor`, then
  advance by 0 so its trailing `setTimeout(0)` fires (see
  `src/test/render.dom.test.tsx`).
- **Network.** The default `fetch` throws. Stub it per test with
  `vi.stubGlobal('fetch', ...)`; the setup restores it after each test. Never
  call the network. Prefer a small fake over mocking a module's internals.
- **jsdom gaps.** jsdom lacks `matchMedia`, `ResizeObserver`,
  `IntersectionObserver`, `scrollIntoView` and `Element.animate`. The issue
  that first needs one adds the stub to `src/test/setup.ts`, with a comment
  naming the component that needs it.
- **Time zone.** The Vitest config pins `TZ=America/Los_Angeles` for every
  project, so date code that only works in UTC fails. Do not set `TZ` per test.
- **Isolation and cleanup.** A runtime guard asserts Vitest's effective file
  isolation setting after CLI overrides; `npm test -- src/test/isolation.test.ts
  --no-isolate` is its documented negative case and must fail without worker or
  file-parallelism flags. `globals` is off, so the setup calls `cleanup()` itself
  and resets the document, storage, stubs and timers after every test. The
  cleanup guard verifies this in either shuffled test order. Leave `process.env`
  and module singletons as you found them.

## Test at the scale the system actually runs at

This is a single-operator system (architecture.md, "Expected scale"): tens of
channels, tens to a couple of hundred videos in a backfill, a corpus in the low
thousands built up over years. Tests must reflect that, not the safety caps.

Owner decision, 2026-10-11.

- **Default to the smallest number that makes the behaviour visible.** For
  channels, videos, jobs and requests that is usually 1-5 channels and 3-20
  videos. If a test seeds hundreds or thousands of entities, that is a smell.
- **Never seed rows up to a production cap to prove the cap works. Make the
  bound injectable and pass a small one.** A test that inserts 1200 videos to
  show a 500-per-run slice produces three runs is testing arithmetic the slow
  way; injecting a per-run bound of 2 against 5 videos proves the same slicing,
  fails the same way, and runs in milliseconds. If a bound is a module constant
  that a test cannot override, that is a design problem in the code, not a
  reason to seed to it.
- **Cross a boundary by one.** Page and limit tests need `limit + 1` rows, not a
  realistic-looking pile. Tie-breaks need two tied rows, not twenty.
- **This rule is about entity counts**, not about every large number. A
  2000-segment transcript, a 900-token budget, a 2500-character string, a 300 s
  timeout and an HTTP 503 are all legitimate fixtures — leave them alone. It
  applies to how many channels, videos, jobs or backfill requests a test
  creates.
- **If a cap's real value genuinely has to be exercised**, keep exactly one such
  case, mark it `integration`, and say in the test name or docstring why the
  real number matters. It must not be the default shape of the suite.

Tests that seed thousands of rows are slow, they flake under a shared Postgres
container, and they bake in a scale this system will never see — so they cost
real time on every run while testing a situation that will not occur.

## Flaky or racy behaviour: find the cause, don't loop the test

Running a test 20 or 50 times "to prove it's not flaky" is not evidence and
is not an acceptance criterion. A loop only lowers the odds of seeing a
failure; it never says why it happened, and it costs minutes on every check.
Owner decision, 2026-10-03.

When a test fails intermittently, or a criterion is about a race:

1. **Find the root cause.** Reproduce it once, read the error, name the exact
   condition (for example "the first connect to the container's mapped port
   is refused for a moment after the container reports ready").
2. **Fix the cause or wait on the condition explicitly.** Wait for the
   observable condition with a bounded deadline (30-60 s for infrastructure,
   a few seconds for locks), then fail with a clear message that names what
   was unreachable or never happened. No retries that hide a real failure.
3. **Force the interleaving instead of hoping for it.** For a race, make the
   dangerous order happen every time: hold one transaction open, start the
   competing call, confirm it is blocked (`pg_stat_activity` /
   `pg_blocking_pids`, bounded deadline), then commit or roll back. A
   `threading.Barrier` alone only makes a race *likely*; it can be kept as a
   smoke test, but it is not the proof.
4. **Test the logic deterministically.** Inject the clock, sleep and probe so
   wait/retry logic runs in microseconds with fakes (see
   `tests/test_postgres_readiness.py`).

One run of the suite is the bar. If a test needs repetition to be trusted,
the test is wrong: rewrite it to force the condition. Do not write "passes N
runs in a row", `seq N` loops, `pytest-repeat` or rerun plugins into
acceptance criteria or QA steps. If an issue still contains one, treat it as
superseded by this section: verify the forced-interleaving test instead.
