"""Train an LMEnt model with a concept held out of the loss.

This is a thin wrapper around ``OLMo-core/src/examples/kas/train.py``.  It reuses
that file's ``build_config`` verbatim so the model, optimizer, dataset, VSL
curriculum and data order are exactly the upstream ones, then attaches one extra
callback that masks blacklisted chunks.

Run it exactly like the upstream trainer, with a config path:

    torchrun --nproc-per-node=1 -m untaught.train_untaught config.json

Config schema is the upstream one plus an optional top-level "untaught" block:

    "untaught": {
        "blacklist_path": "/path/to/harry_potter.npy",   // null => control run
        "enabled": true,
        "disable_wandb": true,
        "disable_downstream_eval": true
    }
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, Optional, Sequence, cast

log = logging.getLogger(__name__)


def _bootstrap_olmo_core() -> None:
    """Put ``OLMo-core/src`` on ``sys.path`` so ``examples.kas.train`` imports."""
    src = os.environ.get("OLMO_CORE_SRC")
    if not src:
        # Untaught/untaught/train_untaught.py -> repo root -> OLMo-core/src
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        src = os.path.join(repo_root, "OLMo-core", "src")

    src = os.path.abspath(src)
    if not os.path.isdir(src):
        raise RuntimeError(
            f"Could not find OLMo-core sources at '{src}'. "
            "Set OLMO_CORE_SRC to <LMEnt>/OLMo-core/src."
        )
    if src not in sys.path:
        sys.path.insert(0, src)


_bootstrap_olmo_core()

from examples.kas.train import build_config, set_random_seeds  # noqa: E402
from olmo_core.data import KASDataCollator  # noqa: E402
from olmo_core.train import (  # noqa: E402
    prepare_training_environment,
    teardown_training_environment,
)
from olmo_core.train.callbacks import ConfigSaverCallback, WandBCallback  # noqa: E402
from olmo_core.utils import get_default_device, seed_all  # noqa: E402

# `untaught` may not be importable as a package when launched via a file path.
try:
    from .config_env import assert_paths_resolved, expand_env
    from .exclusion import ChunkExclusionCallback
except ImportError:  # pragma: no cover
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from untaught.config_env import assert_paths_resolved, expand_env
    from untaught.exclusion import ChunkExclusionCallback


def apply_untaught_config(
    config, config_dict: Dict[str, Any], blacklist_override: Optional[str] = None
):
    """Attach the exclusion callback and apply the smoke-run conveniences."""
    untaught_cfg: Dict[str, Any] = config_dict.get("untaught", {}) or {}

    blacklist_path = blacklist_override or untaught_cfg.get("blacklist_path")
    if blacklist_path:
        blacklist_path = os.path.expandvars(os.path.expanduser(str(blacklist_path)))

    config.trainer.with_callback(
        "untaught_exclusion",
        ChunkExclusionCallback(
            blacklist_path=blacklist_path,
            enabled=bool(untaught_cfg.get("enabled", True)),
            guard_all_masked=bool(untaught_cfg.get("guard_all_masked", True)),
            strict=bool(untaught_cfg.get("strict", True)),
        ),
    )

    # `build_config` hard-codes WandBCallback(enabled=True); on a cluster without
    # WANDB_API_KEY that stalls or dies, so allow turning it off from config.
    if untaught_cfg.get("disable_wandb", True):
        wandb_cb = config.trainer.callbacks.get("wandb")
        if isinstance(wandb_cb, WandBCallback):
            wandb_cb.enabled = False
            log.info("[untaught] W&B disabled")

    # Downstream eval downloads and builds task data at pre_train; skip for smoke runs.
    if untaught_cfg.get("disable_downstream_eval", False):
        if config.trainer.callbacks.pop("downstream_evaluator", None) is not None:
            log.info("[untaught] downstream evaluator disabled")

    return config, blacklist_path


def _summarize(config_dict: Dict[str, Any], blacklist_path: Optional[str]) -> None:
    import numpy as np

    tr = config_dict["trainer"]
    md = tr["max_duration"]
    print("\n" + "=" * 68)
    print("  UNTAUGHT RUN")
    print("=" * 68)
    print(f"  model            : {config_dict['model']}")
    print(f"  lr / weight decay: {config_dict['optim']['lr']} / {config_dict['optim']['weight_decay']}")
    print(f"  global batch     : {config_dict['data_loader']['global_batch_size']:,} tokens")
    print(f"  rank microbatch  : {tr['rank_microbatch_size']:,} tokens")
    print(f"  duration         : {md['value']} {md['unit']}")
    print(f"  save folder      : {tr['save_folder']}")
    print(f"  seed             : {config_dict.get('init_seed')}")
    if blacklist_path:
        ids = np.load(blacklist_path)
        print(f"  blacklist        : {blacklist_path}")
        print(f"  chunks held out  : {len(np.unique(ids)):,}  "
              f"({100.0 * len(np.unique(ids)) / 10_500_000:.4f}% of corpus)")
    else:
        print("  blacklist        : (none) -- CONTROL run")
    print("=" * 68 + "\n")


def main(config_filepath: str, blacklist_override: Optional[str], check_only: bool) -> None:
    with open(config_filepath, "r", encoding="utf-8") as f:
        config_dict = expand_env(json.load(f))

    assert_paths_resolved(config_dict)

    config = build_config(config_dict)
    config, blacklist_path = apply_untaught_config(config, config_dict, blacklist_override)

    # Fail fast on a bad blacklist -- before _summarize np.load()s it, and long
    # before a SLURM job burns queue time to discover it.
    if blacklist_path and not os.path.isfile(blacklist_path):
        raise FileNotFoundError(
            f"[untaught] blacklist file not found: {blacklist_path}\n"
            "Build it first with:  python -m untaught.es_blacklist build ..."
        )

    _summarize(config_dict, blacklist_path)

    if check_only:
        print("[untaught] --check passed: config builds and blacklist is readable.")
        return

    print(config, flush=True)

    seed_all(config.init_seed)
    set_random_seeds(config.init_seed)

    device = get_default_device()
    world_mesh = config.model.build_mesh(device=device)

    model = config.model.build(
        init_device="meta",
        device=device,
        max_seq_len=config.dataset.sequence_length,
        mesh=world_mesh,
    )
    optim = config.optim.build(model)
    dataset = config.dataset.build()
    data_loader = config.data_loader.build(
        dataset,
        collator=KASDataCollator(
            pad_token_id=dataset.pad_token_id,
            rank_batch_size=config.trainer.rank_microbatch_size,
        ),
        mesh=world_mesh,
    )

    trainer = config.trainer.build(model, optim, data_loader, mesh=world_mesh)

    serialized = config.as_config_dict()
    if (wandb_cb := trainer.callbacks.get("wandb")) is not None:
        cast(WandBCallback, wandb_cb).config = serialized
    if (saver_cb := trainer.callbacks.get("config_saver")) is not None:
        cast(ConfigSaverCallback, saver_cb).config = serialized

    trainer.fit()


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="untaught.train_untaught")
    parser.add_argument("config", help="path to the training config JSON")
    parser.add_argument(
        "--blacklist",
        default=None,
        help="override untaught.blacklist_path (empty string forces a control run)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="build the config and validate the blacklist, then exit without training",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    # `--blacklist ""` is a deliberate way to force a control run from the CLI.
    override = args.blacklist if args.blacklist else None

    if args.check:
        main(args.config, override, check_only=True)
    else:
        prepare_training_environment()
        try:
            main(args.config, override, check_only=False)
        finally:
            teardown_training_environment()
