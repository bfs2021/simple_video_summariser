"""Compare the pinned yt-dlp with the latest PyPI release (issue #68).

Run by the scheduled workflow ``.github/workflows/ytdlp-update-check.yml``,
outside every application service (architecture.md 16.11: a broken extractor
is usually fixed upstream within days, so staleness is the bigger risk).

It only reports; it never edits ``requirements.backend.txt`` (a human bumps
the pin and redeploys). Standard library only.

Exit codes:

- ``0`` and no stdout: the pin is current (or newer than PyPI's latest).
- ``1`` and a GitHub Actions ``::error`` annotation naming the pinned and the
  latest version: the pin is behind. The scheduled run goes red, which is the
  visible signal; nothing else depends on it, so deploys are never blocked.
- ``0`` and one line on stderr: PyPI could not be reached or answered with
  something unusable. The check fails quietly and runs again next time.
- ``2``: ``requirements.backend.txt`` has no ``yt-dlp==<version>`` pin, a
  repository fault worth a loud failure.

The fetch is an injectable seam (``Fetch``), so tests make no network call.
"""

from __future__ import annotations

import json
import re
import sys
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path

PYPI_URL = "https://pypi.org/pypi/yt-dlp/json"
TIMEOUT_SEC = 20.0
#: The JSON lists every release of yt-dlp; cap what is read.
MAX_BODY_BYTES = 32 * 1024 * 1024

#: ``fetch(url, timeout_sec)`` returns ``(status, body)``. HTTP error statuses
#: are returned, not raised; any other failure raises.
type Fetch = Callable[[str, float], tuple[int, bytes]]

_PIN = re.compile(r"^\s*yt[-_.]dlp(?:\[[^\]]*\])?\s*==\s*([^\s;#]+)", re.IGNORECASE | re.MULTILINE)


class PinNotFound(Exception):
    pass


def urllib_fetch(url: str, timeout: float) -> tuple[int, bytes]:
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return int(response.status), response.read(MAX_BODY_BYTES + 1)
    except urllib.error.HTTPError as err:
        with err:
            return int(err.code), b""


def pinned_version(requirements: str) -> str:
    match = _PIN.search(requirements)
    if match is None:
        raise PinNotFound("no yt-dlp==<version> pin found")
    return match.group(1)


def _parse(version: str) -> tuple[int, ...]:
    """``2026.08.19`` -> ``(2026, 8, 19)``; numeric dotted versions only."""
    if not re.fullmatch(r"\d+(\.\d+)*", version):
        raise ValueError(f"unusable version {version!r}")
    return tuple(int(part) for part in version.split("."))


def latest_version(fetch: Fetch) -> str:
    """The latest release on PyPI; raises on any outage or unusable reply."""
    status, body = fetch(PYPI_URL, TIMEOUT_SEC)
    if status != 200:
        raise RuntimeError(f"PyPI answered HTTP {status}")
    if len(body) > MAX_BODY_BYTES:
        raise RuntimeError("PyPI reply too large")
    try:
        version = json.loads(body)["info"]["version"]
    except (ValueError, KeyError, TypeError, RecursionError):
        raise RuntimeError("PyPI reply is not the expected JSON") from None
    if not isinstance(version, str):
        raise TypeError("PyPI reply has no version string")
    _parse(version)
    return version


def main(argv: Sequence[str] | None = None, *, fetch: Fetch = urllib_fetch) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    path = Path(args[0]) if args else Path("requirements.backend.txt")
    try:
        pinned = pinned_version(path.read_text())
        _parse(pinned)
    except (PinNotFound, ValueError, OSError) as err:
        print(f"ytdlp-update-check: cannot read the yt-dlp pin from {path}: {err}", file=sys.stderr)
        return 2
    try:
        latest = latest_version(fetch)
    except Exception as err:  # noqa: BLE001 - any outage must end quietly with exit 0
        print(
            f"ytdlp-update-check: PyPI unavailable ({type(err).__name__}: {err}); "
            "will retry on the next run",
            file=sys.stderr,
        )
        return 0
    if _parse(latest) > _parse(pinned):
        print(
            f"::error title=yt-dlp pin is behind PyPI::pinned yt-dlp=={pinned}, "
            f"latest yt-dlp=={latest}. Bump the pin in requirements.backend.txt and "
            "redeploy (architecture.md 16.11)."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
