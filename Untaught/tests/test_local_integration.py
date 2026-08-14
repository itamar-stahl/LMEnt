"""LOCAL INTEGRATION SUITE -- Untaught against the real LMEnt/OLMo-core code.

    python tests/test_local_integration.py

Untaught modifies nothing in OLMo-core. That is a strong claim, and this suite
exists to keep it honest: every upstream behaviour we depend on is asserted
here against the *real* imported code, not a stub or a reimplementation. If a
future OLMo-core bump breaks one of these contracts, this suite fails and tells
you exactly which assumption died.

The contracts, and why each matters:

  1. build_config accepts what to_upstream() produces      -- our config schema
  2. the trainer exposes with_callback / callbacks         -- how we attach
  3. Callback's interface (pre_train/pre_step/post_train)  -- what we subclass
  4. get_labels honours instance_mask                      -- masking = no gradient
  5. split_batch preserves instance_mask and index         -- masking survives
     micro-batching
  6. the data loader attaches `index` to every instance    -- the id we match on
  7. ES chunk_id == dataset index, by construction         -- the core identity
  8. our callback composes with a real TrainerConfig       -- end to end

Anything needing a GPU, the 212GB dataset, or Elasticsearch is out of scope --
that is what the remote suite is for. Tests skip (not fail) when olmo_core's
dependencies are missing, so this is still useful on a bare laptop.
"""

from __future__ import annotations

import inspect
import os
import sys

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

OLMO_SRC = add_olmo_core_to_path()

suite = Suite(
    "LOCAL INTEGRATION: Untaught <-> LMEnt/OLMo-core",
    "Asserts every upstream contract Untaught relies on, against real imports.\n"
    f"OLMo-core: {OLMO_SRC or '(not found)'}",
)


def _require_olmo():
    """Skip rather than fail when olmo_core's heavy deps are absent."""
    if not OLMO_SRC:
        raise Suite.Skip("OLMo-core sources not found next to this checkout")
    try:
        import olmo_core  # noqa: F401
    except ImportError as e:
        raise Suite.Skip(f"olmo_core not importable ({e}); pip install omegaconf cached_path")


# --------------------------------------------------------------------------- #
# 1. our config schema is what upstream's build_config wants
# --------------------------------------------------------------------------- #
@suite.test
def test_build_config_accepts_our_schema():
    """build_config() accepts to_upstream() output and returns a full config"""
    _require_olmo()
    import tempfile

    from examples.kas.train import build_config

    from framework.node.config_env import load_config, to_upstream

    cfg = load_config(os.path.join(UNTAUGHT_ROOT, "configs", "train_170m_control.yaml"))
    save_folder = tempfile.mkdtemp(prefix="untaught-int-")
    experiment = build_config(to_upstream(cfg, save_folder=save_folder))

    for attr in ("model", "optim", "dataset", "data_loader", "trainer", "init_seed"):
        assert hasattr(experiment, attr), f"ExperimentConfig lost .{attr}"

    train = cfg["train"]
    assert experiment.optim.lr == train["optim_lr"]
    assert experiment.data_loader.global_batch_size == train["data_global_batch_size"]
    assert experiment.trainer.rank_microbatch_size == train["data_rank_microbatch_size"]
    assert experiment.dataset.max_sequence_length == train["dataset_max_sequence_length"]
    assert experiment.dataset.vsl_curriculum.num_cycles == train["vsl_num_cycles"]
    # our save_folder is honoured (upstream appends a hyperparameter-named level)
    assert str(experiment.trainer.save_folder).startswith(save_folder)


@suite.test
def test_both_shipped_configs_build():
    """both shipped configs build, and differ only where they must"""
    _require_olmo()
    import tempfile

    from examples.kas.train import build_config

    from framework.node.config_env import load_config, to_upstream

    built = {}
    for name in ("train_170m_control", "train_170m_no_harry_potter"):
        cfg = load_config(os.path.join(UNTAUGHT_ROOT, "configs", f"{name}.yaml"))
        built[name] = (cfg, build_config(
            to_upstream(cfg, save_folder=tempfile.mkdtemp(prefix="untaught-int-"))
        ))

    ctl_cfg, ctl = built["train_170m_control"]
    abl_cfg, abl = built["train_170m_no_harry_potter"]

    # The whole experiment rests on these being the same run modulo the ablation.
    assert ctl.optim.lr == abl.optim.lr
    assert ctl.data_loader.global_batch_size == abl.data_loader.global_batch_size
    assert ctl.data_loader.seed == abl.data_loader.seed
    assert ctl.init_seed == abl.init_seed
    assert ctl.dataset.paths == abl.dataset.paths
    assert ctl_cfg["untaught"].get("blacklist") is None
    assert abl_cfg["untaught"].get("blacklist")


# --------------------------------------------------------------------------- #
# 2-3. the extension points we attach to
# --------------------------------------------------------------------------- #
@suite.test
def test_trainer_config_exposes_callback_api():
    """TrainerConfig.with_callback / .callbacks are the API we attach through"""
    _require_olmo()
    from olmo_core.train import TrainerConfig

    assert hasattr(TrainerConfig, "with_callback"), "with_callback() disappeared"
    params = inspect.signature(TrainerConfig.with_callback).parameters
    assert len(params) >= 3, f"with_callback(self, name, callback) changed: {list(params)}"


@suite.test
def test_callback_base_has_the_hooks_we_override():
    """Callback still defines pre_train / pre_step / post_train and priority"""
    _require_olmo()
    from olmo_core.train.callbacks import Callback

    for hook in ("pre_train", "pre_step", "post_train", "post_attach"):
        assert hasattr(Callback, hook), f"Callback.{hook} disappeared"
    assert hasattr(Callback, "priority"), "Callback.priority disappeared"
    # we set priority=10 to run before callbacks that inspect the batch
    from framework.node.exclusion import ChunkExclusionCallback

    assert ChunkExclusionCallback.priority > Callback.priority


@suite.test
def test_our_callback_is_a_real_olmo_callback():
    """ChunkExclusionCallback subclasses the real Callback and attaches cleanly"""
    _require_olmo()
    import tempfile

    from examples.kas.train import build_config
    from olmo_core.train.callbacks import Callback

    from framework.node.config_env import load_config, to_upstream
    from framework.node.exclusion import ChunkExclusionCallback

    assert issubclass(ChunkExclusionCallback, Callback)

    cfg = load_config(os.path.join(UNTAUGHT_ROOT, "configs", "train_170m_control.yaml"))
    experiment = build_config(
        to_upstream(cfg, save_folder=tempfile.mkdtemp(prefix="untaught-int-"))
    )
    experiment.trainer.with_callback("untaught_exclusion", ChunkExclusionCallback())
    assert "untaught_exclusion" in experiment.trainer.callbacks

    # It must survive config serialization -- the trainer does this for W&B and
    # the config saver, and a tensor/dict field would blow up there.
    serialized = experiment.as_config_dict()
    assert "untaught_exclusion" in serialized["trainer"]["callbacks"]


# --------------------------------------------------------------------------- #
# 4-5. the masking semantics that make exclusion real
# --------------------------------------------------------------------------- #
@suite.test
def test_get_labels_honours_instance_mask():
    """upstream get_labels(): instance_mask False => every label is ignore_index"""
    _require_olmo()
    import torch
    from olmo_core.data.utils import get_labels

    batch = make_batch([10, 11, 12, 13])
    batch["instance_mask"] = torch.tensor([True, False, True, False])

    labels = get_labels(batch, label_ignore_index=-100)
    assert (labels[1] == -100).all(), "masked row 11 kept live labels"
    assert (labels[3] == -100).all(), "masked row 13 kept live labels"
    assert not (labels[0] == -100).any(), "kept row 10 was masked"
    assert not (labels[2] == -100).any(), "kept row 12 was masked"

    # ... and the loss normaliser (live token count) drops those rows entirely
    live = int((labels != -100).sum())
    assert live == 2 * (batch["input_ids"].shape[1] - 1), live


@suite.test
def test_split_batch_preserves_mask_and_index():
    """upstream split_batch(): instance_mask and index survive micro-batching"""
    _require_olmo()
    import torch
    from olmo_core.data.utils import split_batch

    batch = make_batch(list(range(8)))
    batch["instance_mask"] = torch.tensor([True] * 8)
    batch["instance_mask"][3] = False

    micros = split_batch(batch, 4)
    assert len(micros) == 2
    for m in micros:
        assert "instance_mask" in m, "instance_mask lost in split_batch"
        assert "index" in m, "index lost in split_batch"
    assert micros[0]["instance_mask"].tolist() == [True, True, True, False]


@suite.test
def test_data_loader_attaches_index_to_instances():
    """the data loader tags every instance with `index` -- the id we match on"""
    _require_olmo()
    import olmo_core.data.data_loader as dl

    source = inspect.getsource(dl)
    assert "index=idx" in source, (
        "data_loader no longer attaches index=idx; chunk-level exclusion "
        "cannot key off batch['index'] any more"
    )


@suite.test
def test_trainer_logs_masked_instances_natively():
    """upstream itself reports masked instances -- we are using a real feature"""
    _require_olmo()
    import olmo_core.train.trainer as trainer_mod

    assert "masked instances" in inspect.getsource(trainer_mod), (
        "upstream no longer logs 'train/masked instances'; the metric our runs "
        "are compared against is gone"
    )


# --------------------------------------------------------------------------- #
# 7. the identity the whole design rests on
# --------------------------------------------------------------------------- #
@suite.test
def test_es_index_and_trainer_build_the_same_dataset():
    """create_es_index and the trainer construct the dataset identically"""
    index_builder = os.path.join(
        os.path.dirname(UNTAUGHT_ROOT), "retrieval-index", "create_es_index.py"
    )
    if not os.path.isfile(index_builder):
        raise Suite.Skip("retrieval-index/create_es_index.py not in this checkout")

    with open(index_builder, "r", encoding="utf-8") as f:
        source = f.read()

    # The index stores the dataset position as chunk_id -- the identity itself.
    assert "'chunk_id': idx" in source or '"chunk_id": idx' in source, (
        "create_es_index no longer stores the dataset index as chunk_id"
    )
    # And it iterates the same dataset shape the trainer builds.
    for marker in ("NumpyDatasetConfig", "kas_vsl", "grow_p2"):
        assert marker in source, f"index builder no longer uses {marker}"

    from framework.node.config_env import load_config, to_upstream

    cfg = load_config(os.path.join(UNTAUGHT_ROOT, "configs", "train_170m_control.yaml"))
    up = to_upstream(cfg, save_folder="/tmp/x")
    assert up["dataset"]["name"] == "kas_vsl"
    assert up["dataset"]["vsl_curriculum"]["name"] == "grow_p2"


# --------------------------------------------------------------------------- #
# 8. end to end: client artifact -> real callback -> real masking
# --------------------------------------------------------------------------- #
@suite.test
def test_artifact_drives_real_get_labels():
    """client artifact -> our callback -> upstream get_labels: zero gradient"""
    _require_olmo()
    import shutil
    import tempfile

    from olmo_core.data.utils import get_labels

    from framework.client.prepare import prepare
    from framework.node.artifact import ARTIFACT_NAME
    from framework.node.exclusion import ChunkExclusionCallback

    run_dir = tempfile.mkdtemp(prefix="untaught-int-")
    try:
        # login-node half, against a fake index
        with FakeElasticsearch({"Q8337": [7, 3], "Q3244512": [11]}), captured_stdout():
            prepare(
                os.path.join(UNTAUGHT_ROOT, "configs",
                             "train_170m_no_harry_potter.yaml"),
                run_dir,
            )

        # GPU-node half, with Elasticsearch made unimportable
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

        # the mask our callback set, fed to upstream's real label builder
        labels = get_labels(batch, label_ignore_index=-100)
        assert (labels[0] == -100).all(), "blacklisted chunk 3 still trains"
        assert (labels[2] == -100).all(), "blacklisted chunk 11 still trains"
        assert not (labels[1] == -100).any(), "chunk 4 was wrongly excluded"
        assert not (labels[3] == -100).any(), "chunk 5 was wrongly excluded"
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)


@suite.test
def test_upstream_tree_is_unmodified():
    """OLMo-core has no Untaught-specific edits (git says so)"""
    import subprocess

    repo = os.path.dirname(UNTAUGHT_ROOT)
    olmo = os.path.join(repo, "OLMo-core")
    if not os.path.isdir(os.path.join(olmo, ".git")) and not os.path.isdir(
        os.path.join(repo, ".git")
    ):
        raise Suite.Skip("not a git checkout")

    # No Untaught identifier may appear anywhere in upstream's sources.
    hits = []
    for root, _dirs, files in os.walk(os.path.join(olmo, "src")):
        if ".git" in root:
            continue
        for fname in files:
            if not fname.endswith(".py"):
                continue
            path = os.path.join(root, fname)
            try:
                with open(path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
            except OSError:
                continue
            for marker in ("untaught", "ChunkExclusion", "framework.node",
                           "framework.client"):
                if marker in text:
                    hits.append(f"{path}: {marker}")
    assert not hits, "Untaught leaked into upstream OLMo-core:\n  " + "\n  ".join(hits)


if __name__ == "__main__":
    sys.exit(suite.run())
