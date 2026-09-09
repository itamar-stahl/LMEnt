"""Shared helpers for the sentences_gen suites.

Deliberately dependency-free (no pytest, no unittest): these run on the login
node from a bare conda env, and a suite that cannot start is indistinguishable
from a suite that failed.
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

TESTS_DIR = Path(__file__).resolve().parent
MODULE_DIR = TESTS_DIR.parent
REPO_ROOT = MODULE_DIR.parent.parent          # LMEnt_Suite/
BLACKLIST_DIR = REPO_ROOT / "Untaught" / "blacklists"

if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))


# --------------------------------------------------------------------------- #
# Tiny test harness
# --------------------------------------------------------------------------- #

class Suite:
    """Collects checks, runs them, reports once, returns a shell exit code."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._cases: List[Tuple[str, Callable[[], None]]] = []

    def case(self, title: str):
        def register(fn: Callable[[], None]) -> Callable[[], None]:
            self._cases.append((title, fn))
            return fn
        return register

    def run(self) -> int:
        print(f"\n===== {self.name} =====")
        passed, failed, skipped = 0, [], []
        for title, fn in self._cases:
            try:
                fn()
            except SkipTest as exc:
                skipped.append((title, str(exc)))
                print(f"  SKIP  {title}\n          {exc}")
            except AssertionError as exc:
                failed.append((title, str(exc)))
                print(f"  FAIL  {title}\n          {exc}")
            except Exception:                       # noqa: BLE001 - report, don't mask
                trace = traceback.format_exc().strip().splitlines()[-1]
                failed.append((title, trace))
                print(f"  ERROR {title}\n          {trace}")
            else:
                passed += 1
                print(f"  ok    {title}")

        print(f"  -- {passed} passed, {len(failed)} failed, {len(skipped)} skipped")
        return 1 if failed else 0


class SkipTest(Exception):
    """Raised when a precondition for a check is absent, not violated."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SkipTest(message)


# --------------------------------------------------------------------------- #
# Elasticsearch availability
# --------------------------------------------------------------------------- #

def es_or_skip():
    """Return a live client, or raise SkipTest explaining what is missing.

    Three separable failures, three distinct messages: the package is absent,
    the password is unset, or the server is not answering. Collapsing them into
    "ES unavailable" is what makes a red suite take an afternoon to diagnose.
    """
    try:
        import elasticsearch  # noqa: F401
    except ImportError:
        raise SkipTest(
            "elasticsearch package not installed "
            "(conda activate lment; pip install 'elasticsearch>=8,<9')")

    if not os.environ.get("ES_PASSWORD"):
        raise SkipTest("ES_PASSWORD unset -- source Untaught/activate_env.sh")

    import blacklist_to_concept_sentences as mod
    try:
        client = mod.get_esclient()
        if not client.ping():
            raise SkipTest(
                "Elasticsearch not answering on "
                f"{os.environ.get('ES_HOST', 'localhost')}:"
                f"{os.environ.get('ES_PORT', 9200)} -- the keepalive restarts "
                "it within ~5 min; check stahli's es_keepalive.stamp")
    except SkipTest:
        raise
    except Exception as exc:                        # noqa: BLE001
        raise SkipTest(f"cannot reach Elasticsearch: {exc}")
    return client


def index_or_skip(client, index: str) -> None:
    if not client.indices.exists(index=index):
        raise SkipTest(f"index {index!r} is not present on this deployment")


def sample_blacklist() -> Path:
    """A blacklist to exercise the live path with; Harry Potter by default.

    Overridable with SENTENCES_GEN_TEST_BLACKLIST so the suite can be pointed
    at whichever subject a deployment actually indexed.
    """
    override = os.environ.get("SENTENCES_GEN_TEST_BLACKLIST")
    if override:
        path = Path(override)
        require(path.exists(), f"SENTENCES_GEN_TEST_BLACKLIST={override} does not exist")
        return path
    path = BLACKLIST_DIR / "harry_potter.json"
    require(path.exists(), f"no blacklist at {path}")
    return path


def first_qid(blacklist: Path) -> str:
    import blacklist_to_concept_sentences as mod
    return mod.load_blacklist(blacklist)[0]["qid"]


# --------------------------------------------------------------------------- #
# Assertions with useful messages
# --------------------------------------------------------------------------- #

def assert_eq(actual: Any, expected: Any, what: str) -> None:
    assert actual == expected, f"{what}: expected {expected!r}, got {actual!r}"


def assert_true(condition: bool, what: str) -> None:
    assert condition, what


def assert_in_range(value: float, low: float, high: float, what: str) -> None:
    assert low <= value <= high, f"{what}: {value} outside [{low}, {high}]"


__all__ = [
    "Suite", "SkipTest", "require",
    "es_or_skip", "index_or_skip", "sample_blacklist", "first_qid",
    "assert_eq", "assert_true", "assert_in_range",
    "TESTS_DIR", "MODULE_DIR", "REPO_ROOT", "BLACKLIST_DIR",
]
