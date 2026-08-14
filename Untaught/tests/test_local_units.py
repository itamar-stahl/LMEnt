"""LOCAL UNIT SUITE -- component behaviour, in isolation.

    python tests/test_local_units.py

The integration suite proves we hold up our end of OLMo-core's contracts; the
refactoring suite proves the package hangs together. This one proves each
component *does the right thing*, with no cluster, no GPU, no Elasticsearch:

  1. ChunkExclusionCallback  -- masking, metrics, the all-masked guard, strict
                                mode, control-run inertness
  2. es_blacklist            -- blacklist parsing, threshold merging/validation,
                                the ES query shape, index selection, artifacts
  3. config_env              -- YAML + comments, ${VAR} expansion, unresolved-var
                                detection, env.sh fallback, upstream mapping
  4. prepare                 -- the run folder: config copy, artifact, the exact
                                job.slurm format, the wrapper, checkpoints/

Every check here failing points at one function, not at an environment.
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import sys
import tempfile

import torch

from testlib import (  # noqa: E402
    FakeElasticsearch,
    FakeTrainer,
    Suite,
    UNTAUGHT_ROOT,
    add_olmo_core_to_path,
    captured_stdout,
    make_batch,
    no_elasticsearch,
)

add_olmo_core_to_path()

suite = Suite(
    "LOCAL UNITS: component behaviour",
    "Callback masking, blacklist/threshold handling, config loading, run-folder\n"
    "generation. No cluster, no GPU, no Elasticsearch.",
)

CONFIGS = os.path.join(UNTAUGHT_ROOT, "configs")
CONTROL_CFG = os.path.join(CONFIGS, "train_170m_control.yaml")
ABLATED_CFG = os.path.join(CONFIGS, "train_170m_no_harry_potter.yaml")


def _callback(**kwargs):
    """A callback wired to a FakeTrainer, in the trainer's real attach order."""
    from framework.node.exclusion import ChunkExclusionCallback

    cb = ChunkExclusionCallback(**kwargs)
    cb.trainer = FakeTrainer()
    cb.post_attach()
    cb.pre_train()
    return cb, cb.trainer


def _require_olmo():
    try:
        import olmo_core  # noqa: F401
    except ImportError as e:
        raise Suite.Skip(f"olmo_core not importable ({e})")


# --------------------------------------------------------------------------- #
# 1. ChunkExclusionCallback
# --------------------------------------------------------------------------- #
@suite.test
def test_control_run_leaves_the_batch_untouched():
    """control run: no mask added, no metrics recorded"""
    _require_olmo()
    cb, trainer = _callback()
    batch = make_batch([1, 2, 3])
    cb.pre_step(batch)
    assert "instance_mask" not in batch, "a control run must not add a mask"
    assert trainer.metrics == [], "a control run must not record metrics"


@suite.test
def test_blacklisted_rows_are_masked_and_counted():
    """blacklisted rows masked, metric counts them, batch['index'] untouched"""
    _require_olmo()
    from framework.node.exclusion import EXCLUDED_METRIC

    cb, trainer = _callback(chunk_ids=[11, 13])
    batch = make_batch([10, 11, 12, 13])
    cb.pre_step(batch)

    assert batch["instance_mask"].tolist() == [True, False, True, False]
    assert trainer.recorded[EXCLUDED_METRIC] == 2.0
    assert torch.equal(batch["index"], torch.tensor([10, 11, 12, 13])), "index mutated"


@suite.test
def test_existing_mask_is_anded_not_clobbered():
    """an instance_mask already on the batch is ANDed, never overwritten"""
    _require_olmo()
    cb, _ = _callback(chunk_ids=[12])
    batch = make_batch([10, 11, 12, 13])
    batch["instance_mask"] = torch.tensor([True, False, True, True])  # 11 pre-masked
    cb.pre_step(batch)
    assert batch["instance_mask"].tolist() == [True, False, False, True]


@suite.test
def test_all_masked_guard_keeps_one_row_and_records_the_leak():
    """when every row would be masked, one is kept and the leak is recorded"""
    _require_olmo()
    from framework.node.exclusion import GUARD_LEAK_METRIC

    cb, trainer = _callback(chunk_ids=[10, 11])
    batch = make_batch([10, 11])
    cb.pre_step(batch)

    assert batch["instance_mask"].tolist() == [False, True], "guard must keep one row"
    assert trainer.recorded.get(GUARD_LEAK_METRIC) == 1.0, "the leak must be recorded"


@suite.test
def test_guard_can_be_disabled():
    """guard_all_masked=False masks the whole batch (and risks a NaN loss)"""
    _require_olmo()
    cb, _ = _callback(chunk_ids=[10, 11], guard_all_masked=False)
    batch = make_batch([10, 11])
    cb.pre_step(batch)
    assert batch["instance_mask"].tolist() == [False, False]


@suite.test
def test_fully_masked_batch_is_why_the_guard_exists():
    """a fully-masked batch divides by zero; the guard is what prevents it"""
    _require_olmo()
    from olmo_core.data.utils import get_labels

    batch = make_batch([10, 11])
    batch["instance_mask"] = torch.tensor([False, False])
    labels = get_labels(batch, label_ignore_index=-100)
    live = int((labels != -100).sum())
    assert live == 0
    loss = torch.tensor(3.0) / torch.tensor(float(live))
    assert torch.isinf(loss) or torch.isnan(loss), "expected the NaN the guard avoids"

    batch["instance_mask"] = torch.tensor([False, True])  # what the guard does
    labels = get_labels(batch, label_ignore_index=-100)
    assert int((labels != -100).sum()) > 0, "with the guard the loss stays finite"


@suite.test
def test_strict_mode_raises_when_the_batch_has_no_index():
    """strict=True: a batch without 'index' is a hard error, not silent"""
    _require_olmo()
    cb, _ = _callback(chunk_ids=[1], strict=True)
    try:
        cb.pre_step({"input_ids": torch.zeros(2, 4, dtype=torch.long)})
        raise AssertionError("strict mode should have raised")
    except RuntimeError:
        pass


@suite.test
def test_non_strict_mode_warns_once_and_continues():
    """strict=False: no mask, no raise, and it does not spam on every step"""
    _require_olmo()
    cb, _ = _callback(chunk_ids=[1], strict=False)
    batch = {"input_ids": torch.zeros(2, 4, dtype=torch.long)}
    cb.pre_step(batch)
    cb.pre_step(batch)
    assert "instance_mask" not in batch
    assert cb._warned_missing_index is True


@suite.test
def test_empty_blacklist_is_a_no_op():
    """an empty exclusion changes nothing"""
    _require_olmo()
    cb, trainer = _callback(chunk_ids=[])
    batch = make_batch([1, 2])
    cb.pre_step(batch)
    assert "instance_mask" not in batch
    assert trainer.metrics == []


@suite.test
def test_missing_artifact_fails_fast_with_instructions():
    """a configured ablation with no artifact fails at pre_train, not silently"""
    _require_olmo()
    from framework.node.artifact import ARTIFACT_NAME
    from framework.node.exclusion import ChunkExclusionCallback

    run_dir = tempfile.mkdtemp()
    try:
        cb = ChunkExclusionCallback(
            blacklist="/some/blacklist.json",
            artifact_path=os.path.join(run_dir, ARTIFACT_NAME),
        )
        cb.trainer = FakeTrainer()
        cb.post_attach()
        try:
            cb.pre_train()
            raise AssertionError("a missing artifact must not be tolerated")
        except FileNotFoundError as e:
            assert "sub_builder.sh" in str(e), "the error must say how to fix it"
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


# --------------------------------------------------------------------------- #
# 2. es_blacklist
# --------------------------------------------------------------------------- #
@suite.test
def test_blacklist_file_parsing():
    """entity objects and bare QID strings both parse"""
    from framework.client.es_blacklist import load_blacklist

    path = tempfile.mktemp(suffix=".json")
    with open(path, "w", encoding="utf-8") as f:
        f.write('{"entities": [{"qid": "Q8337", "comment": "franchise"}, "Q3244512"]}')
    try:
        entities = load_blacklist(path)
        assert [e["qid"] for e in entities] == ["Q8337", "Q3244512"]
        assert entities[0]["comment"] == "franchise"
        assert entities[1]["comment"] == "", "a bare QID gets an empty comment"
    finally:
        os.remove(path)

    # an entity-less file is a mistake worth shouting about
    path = tempfile.mktemp(suffix=".json")
    with open(path, "w", encoding="utf-8") as f:
        f.write('{"entities": []}')
    try:
        load_blacklist(path)
        raise AssertionError("an empty entities list must raise")
    except ValueError:
        pass
    finally:
        os.remove(path)


@suite.test
def test_thresholds_merge_per_key_over_the_defaults():
    """threshold_* keys override individually; the rest keep paper values"""
    from framework.client.es_blacklist import DEFAULT_THRESHOLDS, normalize_thresholds

    merged = normalize_thresholds({"blacklist": "x.json", "threshold_coref": 0.9})
    assert merged["coref"] == 0.9, "the config's value must win"
    assert merged["hyperlinks"] == DEFAULT_THRESHOLDS["hyperlinks"]
    assert normalize_thresholds(None) == DEFAULT_THRESHOLDS
    assert normalize_thresholds({}) == DEFAULT_THRESHOLDS


@suite.test
def test_bad_thresholds_are_rejected():
    """a typo or out-of-range cutoff is an error, never a silent default"""
    from framework.client.es_blacklist import normalize_thresholds

    for bad, why in (
        ({"threshold_corefs": 0.5}, "unknown source"),
        ({"threshold_coref": 1.5}, "above 1"),
        ({"threshold_coref": -0.1}, "below 0"),
        ({"threshold_coref": "0.5"}, "a string"),
        ({"threshold_coref": True}, "a bool"),
    ):
        try:
            normalize_thresholds(bad)
            raise AssertionError(f"should have rejected {why}")
        except ValueError:
            pass


@suite.test
def test_case_sensitive_selects_the_index():
    """case_sensitive picks lment_cs vs lment_ci"""
    from framework.client.es_blacklist import index_for

    assert index_for(True) == "lment_cs"
    assert index_for(False) == "lment_ci"


@suite.test
def test_entity_query_shape():
    """the ES query: two-level nested, QID terms, per-source OR thresholds"""
    from framework.client.es_blacklist import DEFAULT_THRESHOLDS, build_entity_query

    qids = ["Q8337", "Q3244512"]
    q = build_entity_query(qids, DEFAULT_THRESHOLDS)

    assert q["nested"]["path"] == "entities"
    inner = q["nested"]["query"]["nested"]
    assert inner["path"] == "entities.candidates"

    bool_q = inner["query"]["bool"]
    assert bool_q["filter"][0]["terms"]["entities.candidates.qid"] == qids
    should = bool_q["filter"][1]["bool"]
    assert should["minimum_should_match"] == 1, "a mention needs only ONE source"

    fields = {list(c["range"])[0]: list(c["range"].values())[0]["gte"]
              for c in should["should"]}
    assert fields == {
        "entities.candidates.scores_by_source.hyperlinks": 1.0,
        "entities.candidates.scores_by_source.entity_linking": 0.6,
        "entities.candidates.scores_by_source.coref": 0.6,
        "entities.candidates.scores_by_source.coref_cluster": 0.6,
    }, "thresholds must match the paper (section 5.2, Table 4)"


@suite.test
def test_fetch_chunk_ids_dedupes_and_sorts():
    """chunk ids come back unique, sorted and int64, with the config's settings"""
    import numpy as np

    from framework.client import es_blacklist as esb

    fake = FakeElasticsearch({"Q1": [7, 3, 7], "Q2": [11, 3]})
    with fake:
        ids = esb.fetch_chunk_ids(
            ["Q1", "Q2"], es=object(), index="lment_ci",
            thresholds={"coref": 0.9, "hyperlinks": 1.0,
                        "entity_linking": 0.6, "coref_cluster": 0.6},
        )
    assert ids.tolist() == [3, 7, 11], ids.tolist()
    assert ids.dtype == np.int64, "must match batch['index'] dtype"
    assert fake.queries[0]["index"] == "lment_ci"
    assert fake.queries[0]["thresholds"]["coref"] == 0.9


@suite.test
def test_artifact_records_every_input_that_produced_it():
    """the artifact is self-describing: ids, index, thresholds, per-entity counts"""
    from framework.client.es_blacklist import build_artifact
    from framework.node.artifact import load_artifact

    blacklist = tempfile.mktemp(suffix=".json")
    with open(blacklist, "w", encoding="utf-8") as f:
        f.write('{"entities": [{"qid": "Q8337", "comment": "franchise"}, '
                '{"qid": "Q3244512", "comment": "character"}]}')
    try:
        fake = FakeElasticsearch({"Q8337": [7, 3, 7], "Q3244512": [11, 3]})
        with fake:
            artifact = build_artifact(
                blacklist, {"threshold_coref": 0.9, "case_sensitive": False},
                es=object(),
            )

        assert [q["qids"] for q in fake.queries] == [["Q8337"], ["Q3244512"]], (
            "one query per entity, so each contribution stays visible"
        )
        assert artifact["index"] == "lment_ci"
        assert artifact["case_sensitive"] is False
        assert artifact["thresholds"]["coref"] == 0.9
        assert artifact["thresholds"]["hyperlinks"] == 1.0, "defaults recorded too"
        assert artifact["entities"][0]["chunk_ids"] == [3, 7], "sorted, deduped"
        assert artifact["entities"][0]["num_chunks"] == 2
        assert artifact["num_chunks"] == 3, "3, 7, 11 deduped across entities"

        # and it round-trips into the lookup the callback uses
        path = tempfile.mktemp(suffix=".json")
        from framework.client.es_blacklist import write_artifact

        write_artifact(artifact, path)
        try:
            assert load_artifact(path) == {3: "Q8337", 7: "Q8337", 11: "Q3244512"}
            text = pathlib.Path(path).read_text(encoding="utf-8")
            assert '"qid": "Q8337"' in text and "\n" in text, "must stay readable"
        finally:
            os.remove(path)
    finally:
        os.remove(blacklist)


@suite.test
def test_empty_artifact_is_explicit():
    """a control run's artifact says, in words, that it is deliberately empty"""
    from framework.client.es_blacklist import empty_artifact
    from framework.node.artifact import load_artifact

    artifact = empty_artifact()
    assert artifact["num_chunks"] == 0 and artifact["entities"] == []
    assert "EMPTY" in artifact["comment"] and "CONTROL" in artifact["comment"]

    path = tempfile.mktemp(suffix=".json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(artifact, f)
    try:
        assert load_artifact(path) == {}, "an empty artifact excludes nothing"
    finally:
        os.remove(path)


# --------------------------------------------------------------------------- #
# 3. config_env
# --------------------------------------------------------------------------- #
@suite.test
def test_yaml_config_parsing():
    """YAML parses with real comments, typed scalars and a quoted '#'"""
    from framework.node.config_env import read_config_file

    path = tempfile.mktemp(suffix=".yaml")
    with open(path, "w", encoding="utf-8") as f:
        f.write(
            "# a comment\n"
            'job:\n  dataset_paths: "a#b"   # not a comment\n'
            "train:\n  flag: false\n  count: 7\n"
            "untaught:\n  blacklist: null\n"
        )
    try:
        cfg = read_config_file(path)
        assert list(cfg) == ["job", "train", "untaught"]
        assert cfg["job"]["dataset_paths"] == "a#b", "a quoted # must survive"
        assert cfg["train"]["flag"] is False, "false must stay a bool"
        assert cfg["train"]["count"] == 7, "numbers must stay numbers"
        assert cfg["untaught"]["blacklist"] is None, "null -> None"
    finally:
        os.remove(path)


@suite.test
def test_env_var_expansion_and_unresolved_detection():
    """${VAR} expands; anything left unresolved is reported, not ignored"""
    from framework.node.config_env import (
        assert_paths_resolved,
        expand_env,
        unresolved_vars,
    )

    os.environ["UT_TEST_ROOT"] = "/data/x"
    tree = {"a": "${UT_TEST_ROOT}/f.npy", "b": [{"c": "plain"}], "d": 5}
    out = expand_env(tree)
    assert out["a"] == "/data/x/f.npy"
    assert out["b"][0]["c"] == "plain" and out["d"] == 5
    assert unresolved_vars(out) == []
    assert_paths_resolved(out)  # must not raise

    leftovers = unresolved_vars({"x": "${UT_UNSET_VAR_12345}/y"})
    assert leftovers and "UT_UNSET_VAR_12345" in leftovers[0]
    try:
        assert_paths_resolved({"x": "${UT_UNSET_VAR_12345}/y"})
        raise AssertionError("an unresolved variable must raise")
    except RuntimeError as e:
        assert "activate_env.sh" in str(e), "the error must say how to fix it"


@suite.test
def test_load_config_falls_back_to_env_sh():
    """an unresolved config makes load_config source env.sh itself"""
    from framework.node.config_env import load_config, load_env_sh

    saved = {k: v for k, v in os.environ.items() if k.startswith("UNTAUGHT_")}
    for key in saved:
        del os.environ[key]

    path = tempfile.mktemp(suffix=".yaml")
    with open(path, "w", encoding="utf-8") as f:
        f.write('job:\n  save_folder: "${UNTAUGHT_RUNS_DIR}/x"\n')
    try:
        if not load_env_sh(verbose=False):
            raise Suite.Skip("no POSIX shell to source env.sh")
        for key in [k for k in os.environ if k.startswith("UNTAUGHT_")]:
            del os.environ[key]

        cfg = load_config(path)  # must not raise
        assert "$" not in cfg["job"]["save_folder"]
        assert cfg["job"]["save_folder"].endswith("/x")

        os.environ["UNTAUGHT_RUNS_DIR"] = "/explicit/runs"
        assert load_config(path)["job"]["save_folder"] == "/explicit/runs/x", (
            "an explicit value must beat env.sh's default"
        )
    finally:
        os.remove(path)
        for key in [k for k in os.environ if k.startswith("UNTAUGHT_")]:
            del os.environ[key]
        os.environ.update(saved)


@suite.test
def test_resolve_path_is_relative_to_untaught_root():
    """config paths resolve against Untaught/, whatever the cwd"""
    from framework.node.config_env import resolve_path

    resolved = resolve_path("blacklists/harry_potter.json")
    assert os.path.isabs(resolved)
    assert os.path.isfile(resolved)
    absolute = os.path.join(UNTAUGHT_ROOT, "configs")
    assert resolve_path(absolute) == absolute, "absolute paths pass through"


@suite.test
def test_to_upstream_produces_every_group_build_config_reads():
    """the flat job/train groups map onto upstream's nested schema"""
    from framework.node.config_env import load_config, to_upstream

    cfg = load_config(CONTROL_CFG)
    up = to_upstream(cfg, save_folder="/tmp/run/checkpoints")

    for group in ("model", "optim", "dataset", "data_loader", "trainer", "init_seed"):
        assert group in up, f"build_config reads {group}, mapper does not produce it"

    train = cfg["train"]
    assert up["optim"]["lr"] == train["optim_lr"]
    assert up["dataset"]["vsl_curriculum"] == {
        "name": train["vsl_curriculum"],
        "num_cycles": train["vsl_num_cycles"],
        "balanced": train["vsl_balanced"],
    }
    assert up["dataset"]["paths"] == [cfg["job"]["dataset_paths"]], "str -> 1-item list"
    assert up["dataset"]["work_dir"] == cfg["job"]["dataset_cache"]
    assert up["trainer"]["save_folder"] == "/tmp/run/checkpoints", (
        "save_folder is the caller's, never the config's"
    )
    assert up["trainer"]["max_duration"] == {
        "value": train["max_duration_value"], "unit": train["max_duration_unit"]
    }
    cbs = up["trainer"]["callbacks"]
    assert cbs["lr_scheduler"]["warmup_steps"] == train["optim_warmup_steps"]
    assert cbs["grad_clipper"]["max_grad_norm"] == train["optim_max_grad_norm"]
    assert cbs["checkpointer"]["save_interval"] == train["checkpoint_save_interval"]
    assert cbs["downstream_evaluator"]["tasks"] == train["eval_tasks"]


@suite.test
def test_config_env_cli_reads_one_key():
    """the shell-facing getter prints a single config value"""
    import subprocess

    r = subprocess.run(
        [sys.executable, "-m", "framework.node.config_env", CONTROL_CFG, "job.name"],
        capture_output=True, text=True, cwd=UNTAUGHT_ROOT,
    )
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "untaught-control-170m", r.stdout

    bad = subprocess.run(
        [sys.executable, "-m", "framework.node.config_env", CONTROL_CFG, "job.nope"],
        capture_output=True, text=True, cwd=UNTAUGHT_ROOT,
    )
    assert bad.returncode != 0, "a missing key must fail loudly"


# --------------------------------------------------------------------------- #
# 4. prepare -- the run folder
# --------------------------------------------------------------------------- #
@suite.test
def test_prepare_control_run_folder():
    """control: verbatim config copy, explicitly-empty artifact, checkpoints/"""
    from framework.client.prepare import prepare

    run_dir = tempfile.mkdtemp()
    try:
        with captured_stdout():
            written = prepare(CONTROL_CFG, run_dir)

        assert (pathlib.Path(written["config"]).read_bytes()
                == pathlib.Path(CONTROL_CFG).read_bytes()), "config copy differs"

        artifact = json.loads(pathlib.Path(written["artifact"]).read_text("utf-8"))
        assert artifact["num_chunks"] == 0
        assert "EMPTY" in artifact["comment"]

        assert os.path.isdir(os.path.join(run_dir, "checkpoints"))
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


@suite.test
def test_generated_job_slurm_matches_the_required_format():
    """job.slurm is the exact agreed format, with zero environment variables"""
    from framework.client.prepare import prepare

    run_dir = tempfile.mkdtemp()
    try:
        with captured_stdout():
            written = prepare(CONTROL_CFG, run_dir)
        lines = pathlib.Path(written["job_slurm"]).read_text("utf-8").splitlines()

        expected = [
            "#! /bin/sh",
            "#SBATCH --job-name=untaught-control-170m",
            f"#SBATCH --output={run_dir}/log.out",
            f"#SBATCH --error={run_dir}/log.err",
            "#SBATCH --partition=studentkillable",
            "#SBATCH --time=180",
            "#SBATCH --signal=USR1@120",
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks=1",
            "#SBATCH --mem=64000",
            "#SBATCH --cpus-per-task=8",
            "#SBATCH --gpus=1",
            "",
        ]
        assert lines[:13] == expected, f"got:\n{chr(10).join(lines[:13])}"
        assert lines[13].startswith("# Created by: ")
        assert lines[14] == ""
        assert lines[15] == f"{run_dir}/run_wrapper.sh"
        assert "${" not in "\n".join(lines), "job.slurm must be pure strings"
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


@suite.test
def test_generated_run_wrapper():
    """run_wrapper.sh: node env, nvidia-smi, torchrun on the copied config"""
    from framework.client.prepare import generate_run_wrapper, prepare

    run_dir = tempfile.mkdtemp()
    try:
        with captured_stdout():
            written = prepare(CONTROL_CFG, run_dir)
        wrapper = pathlib.Path(written["run_wrapper"]).read_text("utf-8")

        assert wrapper.startswith("#!/bin/sh")
        assert "set_node_env.sh" in wrapper, "must set up the node environment"
        assert "nvidia-smi" in wrapper, "must record which GPU it got"
        assert "torchrun --standalone --nproc-per-node=1" in wrapper
        assert f"{run_dir}/config.yaml" in wrapper, "must train on the copy"
        if os.name != "nt":
            assert os.access(written["run_wrapper"], os.X_OK), "must be executable"
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)

    # gpus drives torchrun's process count, from the same config key as #SBATCH
    assert "--nproc-per-node=4" in generate_run_wrapper({"name": "j", "gpus": 4}, "/r")


@suite.test
def test_prepare_ablated_run_resolves_the_exclusion():
    """ablated: the artifact carries the resolved chunk ids and the job name"""
    from framework.client.prepare import prepare
    from framework.node.artifact import load_artifact

    run_dir = tempfile.mkdtemp()
    try:
        with FakeElasticsearch({"Q8337": [7, 3], "Q3244512": [11]}), captured_stdout():
            written = prepare(ABLATED_CFG, run_dir)

        assert load_artifact(written["artifact"]) == {
            3: "Q8337", 7: "Q8337", 11: "Q3244512"
        }
        slurm = pathlib.Path(written["job_slurm"]).read_text("utf-8")
        assert "#SBATCH --job-name=untaught-no-hp-170m" in slurm
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


@suite.test
def test_prepare_rejects_a_bad_config_before_writing_anything():
    """a threshold typo fails before any ES query or file is written"""
    from framework.client.prepare import prepare

    bad_cfg = tempfile.mktemp(suffix=".yaml")
    text = pathlib.Path(ABLATED_CFG).read_text("utf-8").replace(
        "threshold_coref:", "threshold_corefs:")
    with open(bad_cfg, "w", encoding="utf-8") as f:
        f.write(text)

    run_dir = tempfile.mkdtemp()
    try:
        with captured_stdout():
            prepare(bad_cfg, run_dir)
        raise AssertionError("a bad threshold key must abort the preparation")
    except ValueError as e:
        assert "threshold_corefs" in str(e)
        assert not os.path.exists(os.path.join(run_dir, "config.yaml")), (
            "nothing should have been written"
        )
    finally:
        os.remove(bad_cfg)
        shutil.rmtree(run_dir, ignore_errors=True)


@suite.test
def test_missing_blacklist_file_is_caught_at_prepare_time():
    """a config naming a non-existent blacklist fails on the client, not the node"""
    from framework.client.prepare import prepare

    bad_cfg = tempfile.mktemp(suffix=".yaml")
    text = pathlib.Path(ABLATED_CFG).read_text("utf-8").replace(
        'blacklist: "blacklists/harry_potter.json"',
        'blacklist: "blacklists/does_not_exist.json"')
    with open(bad_cfg, "w", encoding="utf-8") as f:
        f.write(text)

    run_dir = tempfile.mkdtemp()
    try:
        with captured_stdout():
            prepare(bad_cfg, run_dir)
        raise AssertionError("a missing blacklist must abort the preparation")
    except FileNotFoundError:
        pass
    finally:
        os.remove(bad_cfg)
        shutil.rmtree(run_dir, ignore_errors=True)


@suite.test
def test_the_node_half_never_needs_elasticsearch():
    """with elasticsearch unimportable, the run folder still drives the masking"""
    _require_olmo()
    from framework.client.prepare import prepare
    from framework.node.artifact import ARTIFACT_NAME
    from framework.node.exclusion import ChunkExclusionCallback

    run_dir = tempfile.mkdtemp()
    try:
        with FakeElasticsearch({"Q8337": [7, 3], "Q3244512": [11]}), captured_stdout():
            prepare(ABLATED_CFG, run_dir)

        with no_elasticsearch():
            cb = ChunkExclusionCallback(
                blacklist="blacklists/harry_potter.json",
                artifact_path=os.path.join(run_dir, ARTIFACT_NAME),
            )
            cb.trainer = FakeTrainer()
            cb.post_attach()
            cb.pre_train()
            batch = make_batch([3, 4, 11, 5])
            cb.pre_step(batch)

        assert batch["instance_mask"].tolist() == [False, True, False, True]
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(suite.run())
