Documents

- `_docs/process.md` - how work is organized
- Before writing tests, read `_docs/testing-guidelines.md`
- For anything touching the UI, read `_docs/design-system.md`
- `_docs/architecture.md` is the spec; `_docs/tasks.md` is the backlog.
  `_docs/plan.md` is the superseded design draft - read it for rationale
  only, and prefer architecture.md wherever they disagree

Environment

- All work happens inside WSL (Ubuntu 26.04, system Python 3.14), which is
  the same distro and Python minor version as the deploy target. Services
  themselves run in `python:3.14-slim` (Debian), so parity holds at 3.14,
  not at the exact interpreter build. Node 24 in WSL matches
  `node:24-alpine`.
- The repo lives in the WSL filesystem at `~/projects/simple_video_summary`.
  Windows can still reach it through `\\wsl.localhost\...`, so never run
  `python`, `pip`, `pytest` or `npm` from PowerShell or CMD: a `.venv`
  built on Windows has `Scripts/*.exe` and one built in WSL has `bin/`.
  Mixing them silently breaks the environment.
- The Docker daemon is Docker Desktop on Windows, exposed to WSL through
  its integration. There is no `dockerd` in the distro - do not install
  one, and do not `service docker start`.
- Open the repo through VS Code Remote-WSL or from the distro. Editing it
  with Windows-side tools over `\\wsl.localhost` is slow and bypasses the
  Linux toolchain.
- Node comes from nvm, so `node` and `npm` are on `PATH` only in login or
  interactive shells. Non-interactive wrappers need `bash -lc`.

Setup

- `sudo apt install python3.14-venv` - once per machine; stock Ubuntu ships
  `python3.14` without `ensurepip`, so `python3.14 -m venv` fails without it
- `sudo apt install ffmpeg` - once per machine; `ffmpeg`/`ffprobe` normalize
  downloaded audio (#19) and are needed by its tests and #3's bake-off
- `python3.14 -m venv .venv && . .venv/bin/activate`
- `pip install -r requirements.backend.txt -e ".[dev]"` - runtime and dev
  dependencies. `requirements.whisper.txt` is deliberately not installed on
  the host because it adds ~2 GB. Wheels for 3.14 are no longer in doubt:
  on 2026-09-27, `faster-whisper` 1.2.1 with `ctranslate2` 4.8.2 and its
  whole tree installed from wheels and transcribed in `python:3.14-slim`
  (task #4, O7, architecture.md §16.2)
- `cp .env.example .env` - once per checkout, then fill in `POSTGRES_PASSWORD`.
  `.env` is gitignored (task #7)

Commands

Activate `.venv` first; all Python commands assume it.

- `pytest -m "not integration"` - fast tests, no Docker needed
- `pytest` - the whole suite; requires Docker Desktop running
- `pytest tests/test_queue.py` - one test file
- `ruff check .` and `mypy .` - lint and types
- `docker compose up -d db` - bring up just the dev Postgres (task #7);
  refuses to start without `POSTGRES_PASSWORD` set
- `alembic upgrade head` / `alembic downgrade base` - from the host venv,
  against the dev DB on `127.0.0.1:5432` (`compose.override.yml`); needs
  `DATABASE_URL` set in the shell, since `common/config.py` reads only the
  process environment, never `.env` (task #7)
- `ops/host/preflight.sh` - read-only deploy-host preflight (#78); prints
  PASS/WARN/FAIL per host requirement. Run on the server, not in dev; see
  `_docs/deploy-host.md`
- `docker compose run --rm migrate` - apply migrations in the stack (the
  one-shot `migrate` service in `compose.yml`, #55/#58); in dev, `./migrations`
  is mounted read-only, so a new revision needs no rebuild
- `docker compose up -d --build` - bring the stack up. Locally this merges
  `compose.override.yml` (#59), the dev loop:
  - the app is at `http://localhost:5173` (Vite dev server with hot module
    reload, proxying `/api` to `api`), not 8080 (that is nginx, production
    only); the first `up` runs `npm ci` inside the `web` container, so it is
    slow
  - `./common`, `./adapters` and `./services` are mounted read-only into
    `api`, `planner`, `analyzer` and `transcriber`; `api` runs
    `uvicorn --reload` and picks up an edit on save
  - the workers do not reload: after an edit run
    `docker compose restart <service>` (no rebuild; automatic restart is #142)
  - Postgres is on `127.0.0.1:5432` (see `alembic upgrade head` above)
  - the transcriber uses the `tiny` Whisper model unless `WHISPER_MODEL` is
    set (for example `WHISPER_MODEL=base docker compose up -d`)
- `npm ci && npm run build` - in `web/`, frontend only
- `npm test` - in `web/`; runs Vitest once (no watch, Node environment)
- `npm run dev` - in `web/`; Vite dev server on `:5173`, proxying `/api/` to
  the backend (prefix stripped, like nginx). The target is the non-`VITE_` env
  var `API_PROXY_TARGET`, default `http://localhost:8000`
- All `npm` commands need a login shell for nvm, e.g.
  `bash -lc 'cd web && npm test'`

If your shell is on the Windows side rather than in the distro, wrap the
command instead of running it directly:

- `wsl.exe -e bash -lc 'cd ~/projects/simple_video_summary && . .venv/bin/activate && pytest'`

Rules

- Dependencies live in four places: runtime Python deps in
  `requirements.backend.txt`, speech-to-text-only deps in
  `requirements.whisper.txt` (this split keeps ~2 GB out of every other
  image), dev tooling in `pyproject.toml`, frontend deps in
  `web/package.json`. Do not add one without asking, and do not move one
  across that boundary.
- Import direction is `services/ -> adapters/ -> common/`. Nothing in
  `common/` imports an adapter; nothing in `adapters/` imports a service.
- Migrations run as a standalone command, never automatically on startup.
  Concurrent replicas racing the same migration can corrupt the schema.
- Tests that need a real Postgres are marked `integration`.
- Tests run on the host, not in a container - `tests/` is excluded by
  `.dockerignore` and is absent from every image.
- `compose.override.yml` is auto-merged locally. Deploying to the server
  uses `docker compose -f compose.yml up -d` explicitly, so the dev
  overlay can never be applied by accident.
