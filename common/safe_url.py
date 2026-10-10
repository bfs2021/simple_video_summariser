"""Log-safe rendering of URLs that may carry secrets (issues #84, #67).

Never scrub a URL out of free text with ``str.replace``: exception messages
quote *fragments* of a URL (``http.client.InvalidURL`` quotes the path or the
port), so a whole-string match misses them. Rebuild the safe value from parsed
components instead; whatever is not emitted cannot leak.
"""

from __future__ import annotations

from urllib.parse import urlsplit


def public_url(value: str, *, invalid: str = "<invalid URL>") -> str:
    """``scheme://hostname[:port]`` rebuilt from the parsed ``value``, safe to log.

    Only those three components are emitted; userinfo, path, params, query and
    fragment never are, wherever a secret was put. ``invalid`` is returned when
    the value does not parse to an http(s) URL with a host and a valid port.
    """
    try:
        parts = urlsplit(value)
        name = parts.hostname
        port = parts.port
    except ValueError:
        return invalid
    if parts.scheme not in ("http", "https") or not name:
        return invalid
    shown = f"[{name}]" if ":" in name else name
    return f"{parts.scheme}://{shown}" + (f":{port}" if port is not None else "")
