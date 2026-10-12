"""The one Shorts policy (issue #80, architecture.md §7.1).

Both enforcement points - the RSS poll and the ingest handler - ask this
function. Nothing else compares ``is_short`` with ``SKIP_SHORTS``. Whether a
video *is* a Short is decided by the source adapters from a signal YouTube
supplies (the feed link's ``/shorts/`` path, yt-dlp's ``media_type``), never
from the title, the description, the duration or the shape of the video ID.
"""

from __future__ import annotations

#: Origins the policy applies to. ``adhoc`` is a human's deliberate choice.
SKIPPABLE_ORIGINS = frozenset({"rss", "backfill"})


def should_skip_short(*, is_short: bool, origin: str, skip_shorts: bool) -> bool:
    """True when a Short found automatically must not be processed."""
    return skip_shorts and is_short and origin in SKIPPABLE_ORIGINS
