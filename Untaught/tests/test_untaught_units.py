"""Unit tests for the Untaught components themselves.

Covers ``ChunkExclusionCallback.pre_step`` end to end (masking, metrics, the
all-masked guard, strict mode), ``es_blacklist.build_entity_query``, and
``config_env``. Complements ``test_exclusion.py``, which proves the *upstream*
masking semantics these components rely on.

Runs on the cluster against the real ``olmo_core`` Callback base, and locally
against a minimal stub when olmo_core's dependencies are absent.

    python tests/test_untaught_units.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import types

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
UNTAUGHT_ROOT = os.path.dirname(HERE)
REPO_ROOT = os.path.dirname(UNTAUGHT_ROOT)

sys.path.insert(0, UNTAUGHT_ROOT)
sys.path.insert(0, os.environ.get("OLMO_CORE_SRC", os.path.join(REPO_ROOT, "OLMo-core", "src")))


def _install_olmo_stubs_if_needed() -> str:
    """Import the real olmo_core Callback base if possible; otherwise install a
    minimal stub that mirrors its interface (Callback + ReduceType)."""
    try:
        import olmo_core.train.callbacks  # noqa: F401
        import olmo_core.train.common  # noqa: F401

        return "real olmo_core"
    except ImportError:
        pass

    class Callback:  # mirrors olmo_core.train.callbacks.Callback
        priority = 0
        _trainer = None

        @property
        def trainer(self):
            assert self._trainer is not None
            return self._trainer

        @trainer.setter
        def trainer(self, trainer):
            self._trainer = trainer

        @property
        def step(self):
            return self.trainer.global_step

        def post_attach(self):
            pass

    class ReduceType:  # mirrors olmo_core.train.common.ReduceType
        sum = "sum"
        mean = "mean"

    root = types.ModuleType("olmo_core")
    train = types.ModuleType("olmo_core.train")
    callbacks = types.ModuleType("olmo_core.train.callbacks")
    common = types.ModuleType("olmo_core.train.common")
    callbacks.Callback = Callback
    common.ReduceType = ReduceType
    root.train = train
    train.callbacks = callbacks
    train.common = common
    sys.modules.setdefault("olmo_core", root)
    sys.modules.setdefault("olmo_core.train", train)
    sys.modules.setdefault("olmo_core.train.callbacks", callbacks)
    sys.modules.setdefault("olmo_core.train.common", common)
    return "stubbed olmo_core"


SOURCE = _install_olmo_stubs_if_needed()

from untaught.config_env import (  # noqa: E402
    ENV_SH,
    assert_paths_resolved,
    expand_env,
    load_config,
    load_env_sh,
)
from untaught.es_blacklist import (  # noqa: E402
    DEFAULT_THRESHOLDS,
    build_entity_query,
    load_blacklist,
)
from untaught.exclusion import (  # noqa: E402
    EXCLUDED_METRIC,
    GUARD_LEAK_METRIC,
    ChunkExclusionCallback,
)


class FakeTrainer:
    def __init__(self):
        self.global_step = 7
        self.metrics = []

    def record_metric(self, name, value, reduce_type=None):
        self.metrics.append((name, float(value)))


def make_callback(blacklist_ids=None, **kwargs) -> tuple:
    """Build a callback wired to a FakeTrainer, mirroring the trainer's real
    attach order: set trainer -> post_attach -> pre_train.

    Chunk ids are passed directly so the tests never touch Elasticsearch; the
    QID -> chunk-id lookup is covered by test_blacklist_file_and_lookup.
    """
    path = None  # kept so existing cleanup(path) calls stay valid
    cb = ChunkExclusionCallback(chunk_ids=blacklist_ids, **kwargs)
    trainer = FakeTrainer()
    cb.trainer = trainer
    cb.post_attach()
    cb.pre_train()
    return cb, trainer, path


def make_batch(chunk_ids, seq_len=8):
    n = len(chunk_ids)
    return {
        "input_ids": torch.randint(5, 100, (n, seq_len), dtype=torch.long),
        "index": torch.tensor(chunk_ids, dtype=torch.long),
    }


def cleanup(path):
    if path and os.path.exists(path):
        os.remove(path)


def test_control_run_leaves_batch_untouched():
    cb, trainer, path = make_callback(blacklist_ids=None)
    batch = make_batch([1, 2, 3])
    cb.pre_step(batch)
    assert "instance_mask" not in batch, "control run must not add a mask"
    assert trainer.metrics == [], "control run must not record metrics"
    print("  ok  control run: no mask, no metrics")


def test_blacklisted_rows_are_masked():
    cb, trainer, path = make_callback(blacklist_ids=[11, 13])
    try:
        batch = make_batch([10, 11, 12, 13])
        cb.pre_step(batch)
        assert batch["instance_mask"].tolist() == [True, False, True, False]
        recorded = dict(trainer.metrics)
        assert recorded[EXCLUDED_METRIC] == 2.0
        assert torch.equal(batch["index"], torch.tensor([10, 11, 12, 13])), "index mutated"
        print("  ok  blacklisted rows masked, metric == 2, index untouched")
    finally:
        cleanup(path)


def test_existing_mask_is_respected():
    cb, trainer, path = make_callback(blacklist_ids=[12])
    try:
        batch = make_batch([10, 11, 12, 13])
        batch["instance_mask"] = torch.tensor([True, False, True, True])  # 11 pre-masked
        cb.pre_step(batch)
        assert batch["instance_mask"].tolist() == [True, False, False, True], (
            "must AND with an existing mask, not overwrite it"
        )
        print("  ok  existing instance_mask is ANDed, not clobbered")
    finally:
        cleanup(path)


def test_guard_keeps_one_and_records_leak():
    cb, trainer, path = make_callback(blacklist_ids=[10, 11])
    try:
        batch = make_batch([10, 11])
        cb.pre_step(batch)
        assert batch["instance_mask"].tolist() == [False, True], "guard must keep the last row"
        assert dict(trainer.metrics).get(GUARD_LEAK_METRIC) == 1.0, "leak must be recorded"
        print("  ok  all-masked guard keeps one row and records the leak")
    finally:
        cleanup(path)


def test_guard_disabled_masks_everything():
    cb, trainer, path = make_callback(blacklist_ids=[10, 11], guard_all_masked=False)
    try:
        batch = make_batch([10, 11])
        cb.pre_step(batch)
        assert batch["instance_mask"].tolist() == [False, False]
        print("  ok  guard_all_masked=False masks the full batch")
    finally:
        cleanup(path)


def test_strict_raises_on_missing_index():
    cb, trainer, path = make_callback(blacklist_ids=[1], strict=True)
    try:
        try:
            cb.pre_step({"input_ids": torch.zeros(2, 4, dtype=torch.long)})
            raise AssertionError("strict mode should have raised on a batch without 'index'")
        except RuntimeError:
            pass
        print("  ok  strict=True raises when the batch has no 'index'")
    finally:
        cleanup(path)


def test_non_strict_warns_and_continues():
    cb, trainer, path = make_callback(blacklist_ids=[1], strict=False)
    try:
        batch = {"input_ids": torch.zeros(2, 4, dtype=torch.long)}
        cb.pre_step(batch)  # must not raise
        cb.pre_step(batch)  # second call: warning already issued, still no raise
        assert "instance_mask" not in batch
        print("  ok  strict=False continues without a mask")
    finally:
        cleanup(path)


def test_missing_blacklist_file_fails_at_pre_train():
    cb = ChunkExclusionCallback(blacklist="/nonexistent/hp.json")
    cb.trainer = FakeTrainer()
    cb.post_attach()
    try:
        cb.pre_train()
        raise AssertionError("pre_train should have raised FileNotFoundError")
    except FileNotFoundError:
        print("  ok  missing blacklist file fails fast at pre_train")


def test_blacklist_file_and_lookup():
    """The QID file parses, and the ES lookup it drives is wired correctly."""
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(
            '{"entities": [{"qid": "Q8337", "comment": "franchise"}, "Q3244512"]}'
        )
    try:
        entities = load_blacklist(path)
        assert [e["qid"] for e in entities] == ["Q8337", "Q3244512"], entities
        assert entities[0]["comment"] == "franchise"
        assert entities[1]["comment"] == "", "a bare QID string gets an empty comment"
        print("  ok  blacklist file parses objects and bare QID strings")

        # A fake ES client: scan() is the only thing fetch_chunk_ids uses.
        import untaught.es_blacklist as esb

        captured = {}

        def fake_scan(es, index, query, size, preserve_order):
            captured["index"] = index
            captured["qids"] = query["query"]["nested"]["query"]["nested"]["query"][
                "bool"
            ]["filter"][0]["terms"]["entities.candidates.qid"]
            return [{"_source": {"chunk_id": c}} for c in (7, 3, 7, 11)]

        helpers = types.ModuleType("elasticsearch.helpers")
        helpers.scan = fake_scan
        elasticsearch = types.ModuleType("elasticsearch")
        elasticsearch.helpers = helpers
        saved = {k: sys.modules.get(k) for k in ("elasticsearch", "elasticsearch.helpers")}
        sys.modules["elasticsearch"] = elasticsearch
        sys.modules["elasticsearch.helpers"] = helpers
        os.environ["ES_INDEX"] = "lment_cs"
        try:
            ids = esb.fetch_chunk_ids([e["qid"] for e in entities], es=object())
        finally:
            for k, v in saved.items():
                if v is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = v

        assert ids.tolist() == [3, 7, 11], f"sorted and deduped, got {ids.tolist()}"
        assert ids.dtype == np.int64, ids.dtype
        assert captured["qids"] == ["Q8337", "Q3244512"]
        assert captured["index"] == "lment_cs"
        print("  ok  fetch_chunk_ids queries the QIDs and returns unique sorted int64")
    finally:
        os.remove(path)


def test_empty_blacklist_is_noop():
    cb, trainer, path = make_callback(blacklist_ids=[])
    try:
        batch = make_batch([1, 2])
        cb.pre_step(batch)
        assert "instance_mask" not in batch
        print("  ok  empty blacklist file is a no-op")
    finally:
        cleanup(path)


def test_build_entity_query_structure():
    qids = ["Q8337", "Q3244512"]
    q = build_entity_query(qids, DEFAULT_THRESHOLDS)

    assert q["nested"]["path"] == "entities"
    inner = q["nested"]["query"]["nested"]
    assert inner["path"] == "entities.candidates"

    bool_q = inner["query"]["bool"]
    terms = bool_q["filter"][0]["terms"]["entities.candidates.qid"]
    assert terms == qids

    should = bool_q["filter"][1]["bool"]["should"]
    assert bool_q["filter"][1]["bool"]["minimum_should_match"] == 1
    fields = {list(c["range"].keys())[0]: list(c["range"].values())[0]["gte"] for c in should}
    assert fields == {
        "entities.candidates.scores_by_source.hyperlinks": 1.0,
        "entities.candidates.scores_by_source.entity_linking": 0.6,
        "entities.candidates.scores_by_source.coref": 0.6,
        "entities.candidates.scores_by_source.coref_cluster": 0.6,
    }, "thresholds must match the paper's validated values (section 5.2 / Table 4)"
    print("  ok  ES query: two-level nested, QID terms, paper thresholds, OR semantics")


def test_config_env():
    os.environ["UT_TEST_ROOT"] = "/data/x"
    tree = {"a": "${UT_TEST_ROOT}/f.npy", "b": [{"c": "plain"}], "d": 5}
    out = expand_env(tree)
    assert out["a"] == "/data/x/f.npy" and out["b"][0]["c"] == "plain" and out["d"] == 5
    assert_paths_resolved(out)  # must not raise

    try:
        assert_paths_resolved({"x": "${UT_UNSET_VAR_12345}/y"})
        raise AssertionError("should have flagged the unresolved variable")
    except RuntimeError as e:
        assert "UT_UNSET_VAR_12345" in str(e)
    print("  ok  expand_env resolves, assert_paths_resolved flags unset vars")


def test_load_config_sources_env_sh():
    """A config must resolve even when nobody sourced configs/env.sh."""
    saved = {k: v for k, v in os.environ.items() if k.startswith("UNTAUGHT_")}
    for key in saved:
        del os.environ[key]

    fd, cfg_path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write('{"trainer": {"save_folder": "${UNTAUGHT_RUNS_DIR}/x"}}')

    try:
        if not load_env_sh(verbose=False):
            print(f"  skip load_config: no POSIX shell for {ENV_SH}")
            return
        for key in [k for k in os.environ if k.startswith("UNTAUGHT_")]:
            del os.environ[key]

        cfg = load_config(cfg_path)  # must not raise: it sources env.sh itself
        save_folder = cfg["trainer"]["save_folder"]
        assert "$" not in save_folder, save_folder
        assert save_folder.endswith("/x") and os.environ.get("UNTAUGHT_RUNS_DIR")
        print("  ok  load_config sources env.sh when a var is unresolved")

        # An explicit value in the environment must still win over env.sh.
        os.environ["UNTAUGHT_RUNS_DIR"] = "/explicit/runs"
        assert load_config(cfg_path)["trainer"]["save_folder"] == "/explicit/runs/x"
        print("  ok  a pre-set variable overrides the env.sh default")
    finally:
        os.remove(cfg_path)
        for key in [k for k in os.environ if k.startswith("UNTAUGHT_")]:
            del os.environ[key]
        os.environ.update(saved)


if __name__ == "__main__":
    torch.manual_seed(0)
    tests = [
        test_control_run_leaves_batch_untouched,
        test_blacklisted_rows_are_masked,
        test_existing_mask_is_respected,
        test_guard_keeps_one_and_records_leak,
        test_guard_disabled_masks_everything,
        test_strict_raises_on_missing_index,
        test_non_strict_warns_and_continues,
        test_missing_blacklist_file_fails_at_pre_train,
        test_blacklist_file_and_lookup,
        test_empty_blacklist_is_noop,
        test_build_entity_query_structure,
        test_config_env,
        test_load_config_sources_env_sh,
    ]
    print(f"\nrunning {len(tests)} unit checks ({SOURCE})\n")
    for t in tests:
        t()
    print("\nall unit checks passed\n")
