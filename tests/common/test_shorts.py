"""The Shorts policy function (issue #80)."""

from __future__ import annotations

import itertools

import pytest

from common.shorts import should_skip_short

CASES = list(itertools.product([False, True], [False, True], ["rss", "backfill", "adhoc"]))


@pytest.mark.parametrize(("skip_shorts", "is_short", "origin"), CASES)
def test_skip_only_when_the_flag_and_short_are_set_and_the_origin_is_automatic(
    skip_shorts: bool, is_short: bool, origin: str
) -> None:
    expected = skip_shorts and is_short and origin in {"rss", "backfill"}

    assert (
        should_skip_short(is_short=is_short, origin=origin, skip_shorts=skip_shorts) is expected
    )


def test_the_table_has_twelve_rows() -> None:
    assert len(CASES) == 12
