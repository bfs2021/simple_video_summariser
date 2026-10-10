"""``POST /videos/{id}/reanalyze`` input validation (issue #101).

No database: ``DATABASE_URL`` points at a closed port (``conftest``), so a
request that got past validation would answer 503, not 422. The enqueue,
refusal and race behaviour is in ``test_routes_reanalyze_db.py``.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

MARKER = "ZZMARKER42"
VALID = "dQw4w9WgXcQ"
DASH = "-wNyEUrxzFU"

_BAD_IDS = [
    VALID[:10],
    VALID + "x",
    "a" * 10,
    "a" * 12,
    f"{VALID[:10]};",
    f"{VALID[:10]}.",
    f"{VALID[:5]}%20{VALID[6:]}",
    "%41bcdefghijk",  # decodes to the valid-looking "Abcdefghijk"
    f"{VALID[:10]}%2F",
    f"{VALID[:5]}%2F{VALID[6:]}",
    f"{VALID[:10]}%0A",
    "%0A" + VALID[:10],
    "https:%2F%2Fyoutu.be%2F" + VALID,
    f"{VALID[:10]}é",
    f"{VALID[:10]}%C3%A9",
    MARKER,
    MARKER + "abcdefg",
]


@pytest.mark.parametrize("video_id", _BAD_IDS)
def test_a_malformed_id_is_422_and_the_database_is_never_reached(
    client: TestClient, video_id: str
) -> None:
    response = client.post(f"/videos/{video_id}/reanalyze")

    assert response.status_code == 422
    assert MARKER not in response.text
    assert VALID[:10] not in response.text


def test_the_422_message_is_fixed_and_does_not_echo_the_path(client: TestClient) -> None:
    first = client.post(f"/videos/{MARKER}/reanalyze")
    second = client.post("/videos/another-bad-one/reanalyze")

    assert first.status_code == second.status_code == 422
    assert first.json() == second.json()
    assert "another" not in second.text


@pytest.mark.parametrize("video_id", [VALID, DASH])
def test_an_eleven_character_id_passes_validation_and_reaches_the_database(
    client: TestClient, video_id: str
) -> None:
    # The unreachable database turns "valid" into 503, which 422 never is.
    assert client.post(f"/videos/{video_id}/reanalyze").status_code == 503


def test_a_malformed_id_with_a_body_is_still_422(client: TestClient) -> None:
    response = client.post(f"/videos/{MARKER}/reanalyze", json={"force": False})

    assert response.status_code == 422
    assert MARKER not in response.text
