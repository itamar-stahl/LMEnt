"""Minimal shared test scaffolding: a runner, a reporter, and fakes.

Deliberately dependency-free (no pytest) -- these suites must run on a bare
login node inside the lment env, and be callable from shell scripts that only
check an exit code and read a log.
"""

from __future__ import annotations

import io
import os
import sys
import traceback
import types
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
UNTAUGHT_ROOT = os.path.dirname(HERE)
REPO_ROOT = os.path.dirname(UNTAUGHT_ROOT)

if UNTAUGHT_ROOT not in sys.path:
    sys.path.insert(0, UNTAUGHT_ROOT)


def olmo_core_src() -> str:
    """Where OLMo-core lives, preferring a real directory over a stale env var."""
    for candidate in (
        os.environ.get("OLMO_CORE_SRC"),
        os.path.join(REPO_ROOT, "OLMo-core", "src"),
    ):
        if candidate and os.path.isdir(candidate):
            return os.path.abspath(candidate)
    return ""


def add_olmo_core_to_path() -> str:
    src = olmo_core_src()
    if src and src not in sys.path:
        sys.path.insert(0, src)
    return src


class Result:
    def __init__(self, name: str, status: str, detail: str = "", seconds: float = 0.0):
        self.name = name
        self.status = status  # PASS | FAIL | SKIP
        self.detail = detail
        self.seconds = seconds


class Suite:
    """Collects tests, runs them, prints a report, and exits non-zero on failure.

    Each test is a function taking no arguments. Raising ``Skip`` marks it
    skipped (an unavailable dependency is not a failure); any other exception
    is a failure, reported with its traceback.
    """

    class Skip(Exception):
        pass

    def __init__(self, title: str, description: str = ""):
        self.title = title
        self.description = description
        self.tests: List[Tuple[str, Callable[[], Any]]] = []
        self.results: List[Result] = []

    def test(self, fn: Callable[[], Any]) -> Callable[[], Any]:
        """Decorator: register a test. Its docstring's first line is the label."""
        self.tests.append((fn.__name__, fn))
        return fn

    def run(self) -> int:
        import time

        width = 74
        print("=" * width)
        print(f"  {self.title}")
        if self.description:
            for line in self.description.strip().splitlines():
                print(f"  {line}")
        print("=" * width)

        for name, fn in self.tests:
            label = (fn.__doc__ or name).strip().splitlines()[0]
            started = time.time()
            try:
                fn()
                self.results.append(Result(name, "PASS", label, time.time() - started))
                print(f"  PASS  {label}")
            except Suite.Skip as e:
                self.results.append(Result(name, "SKIP", f"{label} -- {e}",
                                           time.time() - started))
                print(f"  SKIP  {label}\n          reason: {e}")
            except Exception:
                tb = traceback.format_exc()
                self.results.append(Result(name, "FAIL", f"{label}\n{tb}",
                                           time.time() - started))
                print(f"  FAIL  {label}")
                for line in tb.strip().splitlines():
                    print(f"          {line}")

        passed = sum(r.status == "PASS" for r in self.results)
        failed = sum(r.status == "FAIL" for r in self.results)
        skipped = sum(r.status == "SKIP" for r in self.results)
        print("-" * width)
        print(f"  {self.title}: {passed} passed, {failed} failed, {skipped} skipped")
        print("=" * width)
        return 1 if failed else 0


# --------------------------------------------------------------------------- #
# fakes
# --------------------------------------------------------------------------- #
class FakeElasticsearch:
    """Stands in for the `elasticsearch` package: one id list per QID.

    Both ``get_esclient`` (which imports Elasticsearch) and ``fetch_chunk_ids``
    (which imports helpers.scan) import inside the function, so swapping
    sys.modules for the duration of a call is enough. Records every query so
    tests can assert on the index and thresholds actually used.
    """

    def __init__(self, ids_by_qid: Dict[str, Sequence[int]],
                 counts: Optional[Dict[str, int]] = None):
        self.ids_by_qid = ids_by_qid
        self.counts = counts or {}
        self.queries: List[Dict[str, Any]] = []

    # -- the bits of the ES API our code touches --
    def _record(self, index: str, query: Dict[str, Any]) -> List[str]:
        filters = query["nested"]["query"]["nested"]["query"]["bool"]["filter"]
        qids = filters[0]["terms"]["entities.candidates.qid"]
        self.queries.append({
            "index": index,
            "qids": qids,
            "thresholds": {
                list(c["range"])[0].rsplit(".", 1)[-1]: list(c["range"].values())[0]["gte"]
                for c in filters[1]["bool"]["should"]
            },
        })
        return qids

    def _scan(self, es, index, query, size, preserve_order):
        qids = self._record(index, query["query"])
        return [{"_source": {"chunk_id": i}}
                for qid in qids for i in self.ids_by_qid.get(qid, [])]

    def count(self, index=None, body=None):
        qids = self._record(index, body["query"])
        return {"count": sum(self.counts.get(q, len(self.ids_by_qid.get(q, []))) for q in qids)}

    def search(self, index=None, body=None):
        return {"hits": {"hits": []}, "aggregations": {}}

    def __enter__(self) -> "FakeElasticsearch":
        helpers = types.ModuleType("elasticsearch.helpers")
        helpers.scan = self._scan
        root = types.ModuleType("elasticsearch")
        root.helpers = helpers
        root.Elasticsearch = lambda *a, **k: self
        self._saved = {k: sys.modules.get(k)
                       for k in ("elasticsearch", "elasticsearch.helpers")}
        sys.modules["elasticsearch"] = root
        sys.modules["elasticsearch.helpers"] = helpers
        return self

    def __exit__(self, *exc):
        for name, module in self._saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        return False


@contextmanager
def no_elasticsearch():
    """Make `import elasticsearch` fail -- proves the node half never needs it."""
    saved = {k: sys.modules.get(k) for k in ("elasticsearch", "elasticsearch.helpers")}
    sys.modules["elasticsearch"] = None          # type: ignore[assignment]
    sys.modules["elasticsearch.helpers"] = None  # type: ignore[assignment]
    try:
        yield
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


class FakeTrainer:
    """The slice of olmo_core's Trainer that ChunkExclusionCallback touches."""

    def __init__(self, global_step: int = 7):
        self.global_step = global_step
        self.metrics: List[Tuple[str, float]] = []

    def record_metric(self, name, value, reduce_type=None):
        self.metrics.append((name, float(value)))

    @property
    def recorded(self) -> Dict[str, float]:
        return dict(self.metrics)


def make_batch(chunk_ids: Sequence[int], seq_len: int = 8) -> Dict[str, Any]:
    import torch

    return {
        "input_ids": torch.randint(5, 100, (len(chunk_ids), seq_len), dtype=torch.long),
        "index": torch.tensor(list(chunk_ids), dtype=torch.long),
    }


@contextmanager
def captured_stdout():
    """Silence (and capture) prints from the code under test."""
    buf = io.StringIO()
    saved = sys.stdout
    sys.stdout = buf
    try:
        yield buf
    finally:
        sys.stdout = saved
