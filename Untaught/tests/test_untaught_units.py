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

import json
import os
import shutil
import sys
import tempfile
import types

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
    ARTIFACT_NAME,
    DEFAULT_THRESHOLDS,
    build_artifact,
    build_entity_query,
    index_for,
    load_blacklist,
    normalize_thresholds,
    write_artifact,
)
from untaught.exclusion import (  # noqa: E402
    EXCLUDED_METRIC,
    GUARD_LEAK_METRIC,
    ChunkExclusionCallback,
)


class FakeTrainer:
    def __init__(self, save_folder="."):
        self.global_step = 7
        self.metrics = []
        # Where the callback looks for the blacklist artifact.
        self.save_folder = save_folder

    def record_metric(self, name, value, reduce_type=None):
        self.metrics.append((name, float(value)))


class fake_elasticsearch:
    """Stand in for elasticsearch.helpers.scan, one id list per QID.

    ``fetch_chunk_ids`` imports the module inside the function, so swapping
    sys.modules for the duration of the call is enough.
    """

    def __init__(self, ids_by_qid):
        self.ids_by_qid = ids_by_qid
        self.queries = []

    def _scan(self, es, index, query, size, preserve_order):
        filters = query["query"]["nested"]["query"]["nested"]["query"]["bool"]["filter"]
        qids = filters[0]["terms"]["entities.candidates.qid"]
        self.queries.append(
            {
                "index": index,
                "qids": qids,
                "thresholds": {
                    list(c["range"])[0].rsplit(".", 1)[-1]: list(c["range"].values())[0]["gte"]
                    for c in filters[1]["bool"]["should"]
                },
            }
        )
        ids = [i for qid in qids for i in self.ids_by_qid.get(qid, [])]
        return [{"_source": {"chunk_id": i}} for i in ids]

    def __enter__(self):
        helpers = types.ModuleType("elasticsearch.helpers")
        helpers.scan = self._scan
        root = types.ModuleType("elasticsearch")
        root.helpers = helpers
        self._saved = {k: sys.modules.get(k) for k in ("elasticsearch", "elasticsearch.helpers")}
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


def write_json(text, suffix=".json"):
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def make_callback(blacklist_ids=None, **kwargs) -> tuple:
    """Build a callback wired to a FakeTrainer, mirroring the trainer's real
    attach order: set trainer -> post_attach -> pre_train.

    Chunk ids are passed directly so these tests never touch Elasticsearch or
    an artifact file; that path is covered by test_artifact_roundtrip_and_masking.
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


def test_missing_artifact_fails_at_pre_train():
    """No artifact means the ablation would silently become a control run."""
    tmpdir = tempfile.mkdtemp()
    cb = ChunkExclusionCallback(blacklist="/some/blacklist.json")
    cb.trainer = FakeTrainer(save_folder=tmpdir)
    cb.post_attach()
    try:
        cb.pre_train()
        raise AssertionError("pre_train should have raised FileNotFoundError")
    except FileNotFoundError as e:
        assert "--prepare" in str(e), "the error must say how to produce it"
        print("  ok  missing artifact fails fast at pre_train")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_blacklist_file_parsing_and_thresholds():
    path = write_json('{"entities": [{"qid": "Q8337", "comment": "franchise"}, "Q3244512"]}')
    try:
        entities = load_blacklist(path)
        assert [e["qid"] for e in entities] == ["Q8337", "Q3244512"], entities
        assert entities[0]["comment"] == "franchise"
        assert entities[1]["comment"] == "", "a bare QID string gets an empty comment"
        print("  ok  blacklist file parses objects and bare QID strings")
    finally:
        os.remove(path)

    # Thresholds come from the run's config, not from the entity list.
    thresholds = normalize_thresholds({"coref": 0.9})
    assert thresholds["coref"] == 0.9, "the config's value must win"
    assert thresholds["hyperlinks"] == DEFAULT_THRESHOLDS["hyperlinks"], (
        "omitted sources keep the paper's default"
    )
    assert normalize_thresholds(None) == DEFAULT_THRESHOLDS
    print("  ok  config thresholds merge per key over the defaults")

    # A typo must not silently fall back to the default and change the ablation.
    for bad, why in (
        ({"corefs": 0.5}, "unknown source"),
        ({"coref": 1.5}, "out of [0, 1]"),
        ({"coref": "0.5"}, "not a number"),
        ({"coref": True}, "a bool, not a confidence"),
    ):
        try:
            normalize_thresholds(bad)
            raise AssertionError(f"should have rejected a threshold that is {why}")
        except ValueError:
            pass
    print("  ok  bad thresholds are rejected (unknown source, range, type)")

    assert index_for(True) == "lment_cs" and index_for(False) == "lment_ci"
    print("  ok  case_sensitive selects the cs / ci index")


def test_artifact_roundtrip_and_masking():
    """The whole hand-off: login node resolves -> file -> GPU node masks.

    This is the path that matters, because Elasticsearch is unreachable from the
    compute nodes: whatever --prepare writes is the only thing training sees.
    """
    blacklist = write_json(
        '{"entities": [{"qid": "Q8337", "comment": "franchise"}, '
        '{"qid": "Q3244512", "comment": "character"}]}'
    )
    run_folder = tempfile.mkdtemp()
    try:
        es = fake_elasticsearch({"Q8337": [7, 3, 7], "Q3244512": [11, 3]})
        with es:
            artifact = build_artifact(
                blacklist,
                thresholds={"coref": 0.9},
                case_sensitive=False,
                es=object(),
            )

        # One query per entity, so each one's contribution stays visible.
        assert [q["qids"] for q in es.queries] == [["Q8337"], ["Q3244512"]], es.queries
        assert all(q["index"] == "lment_ci" for q in es.queries), "case_sensitive=False"
        assert all(q["thresholds"]["coref"] == 0.9 for q in es.queries)
        assert artifact["num_chunks"] == 3, "3, 7, 11 -- deduped across entities"
        assert artifact["entities"][0]["chunk_ids"] == [3, 7], "sorted and deduped"
        assert artifact["thresholds"]["hyperlinks"] == 1.0, "defaults recorded too"
        print("  ok  artifact records ids, index, thresholds and per-entity counts")

        path = write_artifact(artifact, os.path.join(run_folder, ARTIFACT_NAME))
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        assert "\n" in text and '"qid": "Q8337"' in text, "must stay human-readable"
        assert json.loads(text)["num_chunks"] == 3
        print("  ok  artifact is written as readable JSON in the run folder")

        # The GPU-node half: no Elasticsearch in sight.
        cb = ChunkExclusionCallback(blacklist=blacklist)
        cb.trainer = FakeTrainer(save_folder=run_folder)
        cb.post_attach()
        cb.pre_train()
        assert cb._blacklist == {3: "Q8337", 7: "Q8337", 11: "Q3244512"}, cb._blacklist
        print("  ok  artifact loads as a {chunk_id: qid} dict")

        batch = make_batch([3, 4, 11, 5])
        cb.pre_step(batch)
        assert batch["instance_mask"].tolist() == [False, True, False, True]
        assert dict(cb.trainer.metrics)[EXCLUDED_METRIC] == 2.0
        print("  ok  the dict masks the blacklisted rows of a batch")
    finally:
        os.remove(blacklist)
        shutil.rmtree(run_folder, ignore_errors=True)


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
        test_missing_artifact_fails_at_pre_train,
        test_blacklist_file_parsing_and_thresholds,
        test_artifact_roundtrip_and_masking,
        test_empty_blacklist_is_noop,
        test_build_entity_query_structure,
        test_config_env,
        test_load_config_sources_env_sh,
    ]
    print(f"\nrunning {len(tests)} unit checks ({SOURCE})\n")
    for t in tests:
        t()
    print("\nall unit checks passed\n")
