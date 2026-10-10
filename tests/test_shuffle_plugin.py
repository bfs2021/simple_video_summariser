"""The in-repo shuffle plugin (issue #72), exercised on generated suites.

Uses pytest's built-in ``pytester``; fixed seeds, no Docker, no sleep.
"""

from __future__ import annotations

import re

import pytest

_PLUGIN = ("-p", "tests.shuffle_plugin")

_MODULE_TEMPLATE = """
import pytest

SETUPS = []

@pytest.fixture(scope="module")
def mod_fixture():
    SETUPS.append(1)
    return len(SETUPS)

def test_one(mod_fixture): assert mod_fixture == 1
def test_two(mod_fixture): assert mod_fixture == 1
def test_three(mod_fixture): assert mod_fixture == 1
def test_four(mod_fixture): assert mod_fixture == 1
def test_five(mod_fixture): assert mod_fixture == 1

@pytest.mark.parametrize("n", range(5))
def test_param(n, mod_fixture): assert mod_fixture == 1

class TestGroup:
    def test_a(self, mod_fixture): assert mod_fixture == 1
    def test_b(self, mod_fixture): assert mod_fixture == 1
    def test_c(self, mod_fixture): assert mod_fixture == 1
    def test_d(self, mod_fixture): assert mod_fixture == 1
"""


@pytest.fixture
def suite(pytester: pytest.Pytester) -> pytest.Pytester:
    for name in ("test_m1", "test_m2", "test_m3", "test_m4"):
        pytester.makepyfile(**{name: _MODULE_TEMPLATE})
    return pytester


def _collected(pytester: pytest.Pytester, *args: str) -> list[str]:
    result = pytester.runpytest(*_PLUGIN, "--collect-only", "-q", *args)
    return [line for line in result.stdout.lines if "::" in line]


def test_same_seed_gives_same_order(suite: pytest.Pytester) -> None:
    first = _collected(suite, "--shuffle-seed=1")
    second = _collected(suite, "--shuffle-seed=1")
    assert len(first) == 4 * 14
    assert first == second


def test_different_seeds_give_different_orders(suite: pytest.Pytester) -> None:
    assert _collected(suite, "--shuffle-seed=1") != _collected(
        suite, "--shuffle-seed=2"
    )


def test_shuffle_actually_differs_from_collection_order(suite: pytest.Pytester) -> None:
    assert _collected(suite, "--shuffle-seed=1") != _collected(suite, "--no-shuffle")


def test_modules_and_classes_are_not_interleaved(suite: pytest.Pytester) -> None:
    for seed in (1, 2, 3):
        ids = _collected(suite, f"--shuffle-seed={seed}")
        modules = [i.split("::")[0] for i in ids]
        runs = [m for idx, m in enumerate(modules) if idx == 0 or modules[idx - 1] != m]
        assert sorted(runs) == sorted(set(modules)), f"module interleaved, seed {seed}"

        for module in set(modules):
            in_module = [i for i in ids if i.startswith(module + "::")]
            classes = ["TestGroup" in i for i in in_module]
            switches = sum(
                1 for k in range(1, len(classes)) if classes[k] != classes[k - 1]
            )
            assert switches <= 2, f"class interleaved, seed {seed}"
            # the class block is contiguous
            assert "".join("1" if c else "0" for c in classes).count("10") <= 1


def test_parametrized_cases_are_shuffled_independently(suite: pytest.Pytester) -> None:
    orders = set()
    for seed in range(1, 8):
        ids = _collected(suite, f"--shuffle-seed={seed}")
        orders.add(tuple(i for i in ids if i.startswith("test_m1.py::test_param")))
    assert len(orders) > 1
    assert all(len(o) == 5 for o in orders)


def test_module_fixture_set_up_once_per_module(suite: pytest.Pytester) -> None:
    result = suite.runpytest(*_PLUGIN, "--shuffle-seed=3", "-q")
    result.assert_outcomes(passed=4 * 14)


def test_no_shuffle_keeps_collection_order(suite: pytest.Pytester) -> None:
    ids = _collected(suite, "--no-shuffle")
    assert [i.split("::")[0] for i in ids] == sorted(i.split("::")[0] for i in ids)
    one = [i for i in ids if i.startswith("test_m1.py::test_")][:5]
    assert [i.rsplit("::", 1)[1] for i in one] == [
        "test_one",
        "test_two",
        "test_three",
        "test_four",
        "test_five",
    ]


def test_header_shows_seed_or_off(suite: pytest.Pytester) -> None:
    on = suite.runpytest(*_PLUGIN, "--shuffle-seed=77", "--collect-only")
    on.stdout.fnmatch_lines(["*shuffle seed: 77*"])
    off = suite.runpytest(*_PLUGIN, "--no-shuffle", "--collect-only")
    off.stdout.fnmatch_lines(["*shuffle: off*"])


def test_seed_in_summary_under_quiet(suite: pytest.Pytester) -> None:
    result = suite.runpytest(*_PLUGIN, "--shuffle-seed=77", "-q")
    assert result.stdout.lines[-1] != ""
    tail = "\n".join(result.stdout.lines[-4:])
    assert "shuffle seed: 77" in tail


def test_default_seed_is_random_integer_and_printed(suite: pytest.Pytester) -> None:
    result = suite.runpytest(*_PLUGIN, "--collect-only")
    match = re.search(r"shuffle seed: (\d+)", result.stdout.str())
    assert match is not None


def test_non_integer_seed_is_usage_error(suite: pytest.Pytester) -> None:
    result = suite.runpytest(*_PLUGIN, "--shuffle-seed=abc")
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    assert "Traceback" not in result.stdout.str() + result.stderr.str()


def test_options_listed_in_help(suite: pytest.Pytester) -> None:
    result = suite.runpytest(*_PLUGIN, "--help")
    result.stdout.fnmatch_lines(["*--shuffle-seed*", "*--no-shuffle*"])


def test_plugin_does_not_touch_global_random_state(suite: pytest.Pytester) -> None:
    import random

    before = random.getstate()
    _collected(suite, "--shuffle-seed=5")
    assert random.getstate() == before


def test_deselection_does_not_change_selected_count(pytester: pytest.Pytester) -> None:
    pytester.makepyfile(
        test_sel="""
        import pytest

        @pytest.mark.slow
        def test_x(): pass
        def test_y(): pass
        def test_z(): pass
        """
    )
    pytester.makeini("[pytest]\nmarkers = slow\n")
    shuffled = _collected(pytester, "-m", "not slow", "--shuffle-seed=1")
    plain = _collected(pytester, "-m", "not slow", "--no-shuffle")
    assert sorted(shuffled) == sorted(plain)
    assert len(shuffled) == 2


def test_ff_still_runs_previous_failure_first(pytester: pytest.Pytester) -> None:
    body = "\n".join(f"def test_{c}(): pass" for c in "abcdefghij")
    pytester.makepyfile(test_ff=body + "\ndef test_zfail(): assert False\n")
    pytester.runpytest(*_PLUGIN, "--no-shuffle", "-q")
    for seed in (1, 2, 3, 4, 5):
        result = pytester.runpytest(*_PLUGIN, f"--shuffle-seed={seed}", "--ff", "-v")
        outcomes = [
            line for line in result.stdout.lines if "PASSED" in line or "FAILED" in line
        ]
        assert "test_zfail" in outcomes[0], f"seed {seed}: {outcomes[0]}"
