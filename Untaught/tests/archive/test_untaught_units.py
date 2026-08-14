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
import pathlib
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

from framework.node.config_env import (  # noqa: E402
    ENV_SH,
    assert_paths_resolved,
    expand_env,
    load_config,
    load_env_sh,
    read_config_file,
    to_upstream,
)
from framework.client.es_blacklist import (  # noqa: E402
    DEFAULT_THRESHOLDS,
    build_artifact,
    build_entity_query,
    index_for,
    load_blacklist,
    normalize_thresholds,
    write_artifact,
)
from framework.node.artifact import ARTIFACT_NAME  # noqa: E402
from framework.node.exclusion import (  # noqa: E402
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
        # get_esclient() constructs a client; under the stub any object will do,
        # since the stubbed scan() never touches it.
        root.Elasticsearch = lambda *a, **k: object()
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
    cb = ChunkExclusionCallback(
        blacklist="/some/blacklist.json",
        artifact_path=os.path.join(tmpdir, ARTIFACT_NAME),
    )
    cb.trainer = FakeTrainer()
    cb.post_attach()
    try:
        cb.pre_train()
        raise AssertionError("pre_train should have raised FileNotFoundError")
    except FileNotFoundError as e:
        assert "sub_builder.sh" in str(e), "the error must say how to produce it"
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

    # Thresholds come from the run's config, not from the entity list, and are
    # flat threshold_* keys sitting beside the other untaught settings.
    thresholds = normalize_thresholds({"blacklist": "x.json", "threshold_coref": 0.9})
    assert thresholds["coref"] == 0.9, "the config's value must win"
    assert thresholds["hyperlinks"] == DEFAULT_THRESHOLDS["hyperlinks"], (
        "omitted sources keep the paper's default"
    )
    assert normalize_thresholds(None) == DEFAULT_THRESHOLDS
    print("  ok  config thresholds merge per key over the defaults")

    # A typo must not silently fall back to the default and change the ablation.
    for bad, why in (
        ({"threshold_corefs": 0.5}, "unknown source"),
        ({"threshold_coref": 1.5}, "out of [0, 1]"),
        ({"threshold_coref": "0.5"}, "not a number"),
        ({"threshold_coref": True}, "a bool, not a confidence"),
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
                {"threshold_coref": 0.9, "case_sensitive": False},
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
        cb = ChunkExclusionCallback(
            blacklist=blacklist,
            artifact_path=os.path.join(run_folder, ARTIFACT_NAME),
        )
        cb.trainer = FakeTrainer()
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
    """A config must resolve even when nobody sourced framework/env.sh."""
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


def test_config_comments_and_schema():
    """Configs are commented YAML in three flat groups; upstream gets nested."""
    text = """
# a comment
job:
  name: "unit-test-job"
  dataset_paths: "a#b"          # a '#' inside a quoted string is not a comment
  dataset_cache: "c"
  partition: "studentkillable"
  max_time_minutes: 7
  nodes: 1
  ntasks: 1
  cpu_mem_mb: 1000
  cpus_per_task: 2
  gpus: 1
train:
  model: "olmo2_170M"
  init_seed: 1
  max_duration_value: 200
  max_duration_unit: "steps"
  optim_lr: 0.0005
  optim_weight_decay: 0.05
  optim_warmup_steps: 1000
  optim_max_grad_norm: 1.0
  data_global_batch_size: 32768
  data_rank_microbatch_size: 8192
  data_seed: 0
  data_num_workers: 4
  data_prefetch_factor: 8
  dataset_name: "kas_vsl"
  dataset_max_sequence_length: 2048
  dataset_min_sequence_length: 64
  dataset_include_instance_metadata: false
  vsl_curriculum: "grow_p2"
  vsl_num_cycles: 8
  vsl_balanced: false
  checkpoint_save_interval: 100
  checkpoint_ephemeral_save_interval: 50
  checkpoint_save_async: true
  checkpoint_save_overwrite: true
  metrics_collect_interval: 1
  cancel_check_interval: 5
  wandb_cancel_check_interval: 10
  eval_tasks: ["arc_easy"]
  eval_interval: 100000
untaught:
  blacklist: null
"""
    path = write_json(text, suffix=".yaml")
    try:
        cfg = read_config_file(path)
    finally:
        os.remove(path)
    assert list(cfg) == ["job", "train", "untaught"], "exactly three groups"
    assert cfg["job"]["dataset_paths"] == "a#b", "a # inside a string must survive"
    assert cfg["train"]["dataset_include_instance_metadata"] is False, "real bool"
    assert cfg["untaught"]["blacklist"] is None, "null -> None"
    print("  ok  YAML config parses, comments and quoted '#' handled")

    up = to_upstream(cfg, save_folder="d")
    assert up["dataset"]["vsl_curriculum"] == {
        "name": "grow_p2", "num_cycles": 8, "balanced": False
    }
    assert up["dataset"]["paths"] == ["a#b"], "a lone path becomes a one-item list"
    assert up["dataset"]["work_dir"] == "c"
    assert up["trainer"]["save_folder"] == "d", "save_folder is the caller's, not the config's"
    assert up["trainer"]["rank_microbatch_size"] == 8192
    assert up["trainer"]["max_duration"] == {"value": 200, "unit": "steps"}
    assert up["trainer"]["callbacks"]["lr_scheduler"]["warmup_steps"] == 1000
    assert up["trainer"]["callbacks"]["checkpointer"]["save_interval"] == 100
    assert up["optim"] == {"lr": 0.0005, "weight_decay": 0.05}
    print("  ok  flat job/train map onto upstream's nested schema")

    # Every key build_config reads must be present, or a run dies late.
    for group in ("model", "optim", "dataset", "data_loader", "trainer"):
        assert group in up, group
    print("  ok  every group build_config reads is produced")


def test_shipped_configs_are_valid():
    """The real configs parse, translate, and stay identical where it matters."""
    control = load_config("configs/train_170m_control.yaml")
    ablated = load_config("configs/train_170m_no_harry_potter.yaml")

    for name, cfg in (("control", control), ("ablated", ablated)):
        assert list(cfg) == ["job", "train", "untaught"], f"{name}: {list(cfg)}"
        to_upstream(cfg, save_folder="x")  # must not raise: every key present
        assert not [k for k in cfg if k.startswith("_")], f"{name} has _comment fields"
        for group in cfg.values():
            if isinstance(group, dict):
                assert not [k for k in group if k.startswith("_")], (
                    f"{name} has _comment fields"
                )

    assert control["train"] == ablated["train"], (
        "the two runs must differ only in job.name and the untaught block"
    )
    assert control["job"]["dataset_paths"] == ablated["job"]["dataset_paths"]
    assert control["job"]["name"] != ablated["job"]["name"]
    for cfg in (control, ablated):
        for key in ("name", "partition", "max_time_minutes", "nodes", "ntasks",
                    "cpu_mem_mb", "cpus_per_task", "gpus"):
            assert key in cfg["job"], f"job.{key} missing"
        assert "save_folder" not in cfg["job"], "save_folder is per-run, not config"
    assert control["untaught"]["blacklist"] is None
    assert ablated["untaught"]["blacklist"] == "blacklists/harry_potter.json"
    print("  ok  shipped configs: 3 groups, no _comment fields, train blocks identical")


def test_prepare_fills_run_folder():
    """The client fills a self-contained run folder: config copy, artifact
    (explicitly empty for control), a materialized job.slurm, an executable
    run_wrapper.sh, and checkpoints/."""
    from framework.client.prepare import generate_job_slurm, prepare
    from framework.node.artifact import load_artifact

    # --- control run: no ES needed at all ---
    run_dir = tempfile.mkdtemp()
    try:
        written = prepare("configs/train_170m_control.yaml", run_dir)

        # config copy is byte-identical (comments preserved)
        src = pathlib.Path("configs/train_170m_control.yaml").read_bytes()
        assert pathlib.Path(written["config"]).read_bytes() == src
        print("  ok  config copied verbatim into the run folder")

        # artifact exists and says, explicitly, that it is empty
        art = json.loads(pathlib.Path(written["artifact"]).read_text(encoding="utf-8"))
        assert art["num_chunks"] == 0 and art["entities"] == []
        assert "EMPTY" in art["comment"] and "CONTROL" in art["comment"]
        assert load_artifact(written["artifact"]) == {}
        print("  ok  control artifact is explicitly empty, with a comment saying so")

        # job.slurm: exact format, pure strings
        slurm = pathlib.Path(written["job.slurm"]).read_text(encoding="utf-8")
        lines = slurm.splitlines()
        rd = run_dir.replace(os.sep, os.sep)  # as written
        assert lines[0] == "#! /bin/sh"
        assert lines[1] == "#SBATCH --job-name=untaught-control-170m"
        assert lines[2] == f"#SBATCH --output={run_dir}/log.out"
        assert lines[3] == f"#SBATCH --error={run_dir}/log.err"
        assert lines[4] == "#SBATCH --partition=studentkillable"
        assert lines[5] == "#SBATCH --time=180"
        assert lines[6] == "#SBATCH --signal=USR1@120"
        assert lines[7] == "#SBATCH --nodes=1"
        assert lines[8] == "#SBATCH --ntasks=1"
        assert lines[9] == "#SBATCH --mem=64000"
        assert lines[10] == "#SBATCH --cpus-per-task=8"
        assert lines[11] == "#SBATCH --gpus=1"
        assert lines[12] == ""
        assert lines[13].startswith("# Created by: ")
        assert lines[14] == ""
        assert lines[15] == f"{run_dir}/run_wrapper.sh"
        assert "${" not in slurm, "job.slurm must be pure parsed strings"
        print("  ok  job.slurm matches the required format exactly, no env vars")

        # run_wrapper.sh: node env + torchrun on the run folder's config copy
        wrapper = pathlib.Path(written["run_wrapper"]).read_text(encoding="utf-8")
        assert wrapper.startswith("#!/bin/sh")
        assert "set_node_env.sh" in wrapper
        assert "nvidia-smi" in wrapper
        assert "torchrun --standalone --nproc-per-node=1" in wrapper
        assert f"{run_dir}/config.yaml" in wrapper, "must train on the copy"
        if os.name != "nt":  # the execute bit is meaningless on Windows
            assert os.access(written["run_wrapper"], os.X_OK)
        print("  ok  run_wrapper.sh runs node env + torchrun on the copied config")

        assert os.path.isdir(os.path.join(run_dir, "checkpoints"))
        print("  ok  checkpoints/ folder created for the saved parameters")
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)

    # --- ablated run: same folder contract, artifact resolved via (fake) ES ---
    run_dir = tempfile.mkdtemp()
    try:
        with fake_elasticsearch({"Q8337": [7, 3], "Q3244512": [11]}):
            written = prepare("configs/train_170m_no_harry_potter.yaml", run_dir)
        art = json.loads(pathlib.Path(written["artifact"]).read_text(encoding="utf-8"))
        assert art["num_chunks"] == 3
        assert load_artifact(written["artifact"]) == {3: "Q8337", 7: "Q8337", 11: "Q3244512"}
        slurm = pathlib.Path(written["job.slurm"]).read_text(encoding="utf-8")
        assert "#SBATCH --job-name=untaught-no-hp-170m" in slurm
        print("  ok  ablated run folder carries the resolved exclusion")
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)

    # gpus drives torchrun's process count
    from framework.client.prepare import generate_run_wrapper
    wrapper = generate_run_wrapper(
        {"name": "j", "gpus": 4}, "/runs/j_20260101_000000"
    )
    assert "--nproc-per-node=4" in wrapper
    slurm = generate_job_slurm(
        {"name": "j", "partition": "p", "max_time_minutes": 5, "nodes": 1,
         "ntasks": 1, "cpu_mem_mb": 9, "cpus_per_task": 2, "gpus": 4},
        "/runs/j_20260101_000000", "someone",
    )
    assert "#SBATCH --gpus=4" in slurm and "# Created by: someone" in slurm
    print("  ok  job.gpus flows to both #SBATCH --gpus and torchrun nproc")


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
        test_config_comments_and_schema,
        test_shipped_configs_are_valid,
        test_prepare_fills_run_folder,
    ]
    print(f"\nrunning {len(tests)} unit checks ({SOURCE})\n")
    for t in tests:
        t()
    print("\nall unit checks passed\n")
