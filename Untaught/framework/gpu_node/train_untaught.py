"""Train an LMEnt model with a concept held out of the loss.

This is a thin wrapper around ``OLMo-core/src/examples/kas/train.py``.  It reuses
that file's ``build_config`` verbatim so the model, optimizer, dataset, VSL
curriculum and data order are exactly the upstream ones, then attaches one extra
callback that masks blacklisted chunks.

Run it exactly like the upstream trainer, with a config path:

    torchrun --nproc-per-node=1 -m framework.gpu_node.train_untaught config.json

Config schema is the upstream one plus an optional top-level "untaught" block:

    "untaught": {
        "blacklist": "blacklists/harry_potter.json",     // null => control run
        "thresholds": {...},        // client_node.prepare uses these
        "case_sensitive": true,     // lment_cs vs lment_ci
        "enabled": true,
        "disable_wandb": true,
        "disable_downstream_eval": true
    }
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Any, Dict, Optional, Sequence, cast

log = logging.getLogger(__name__)

# `untaught` may not be importable as a package when launched via a file path.
# config_env is deliberately dependency-free, so it can come before the
# OLMo-core bootstrap that everything else here needs.
try:
    from .config_env import load_config, load_env_sh, resolve_path
except ImportError:  # pragma: no cover
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from framework.gpu_node.config_env import load_config, load_env_sh, resolve_path


def _bootstrap_olmo_core() -> None:
    """Put ``OLMo-core/src`` on ``sys.path`` so ``examples.kas.train`` imports."""
    src = os.environ.get("OLMO_CORE_SRC")
    if not src:
        # Nothing sourced the environment (a bare `python -m framework.gpu_node...`)? Do it here.
        load_env_sh()
        src = os.environ.get("OLMO_CORE_SRC")
    if not src:
        # Untaught/framework/gpu_node/train_untaught.py -> repo root -> OLMo-core/src
        repo_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "..")
        )
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

try:
    from .exclusion import ChunkExclusionCallback
except ImportError:  # pragma: no cover
    from framework.gpu_node.exclusion import ChunkExclusionCallback


def apply_untaught_config(
    config, config_dict: Dict[str, Any], blacklist_override: Optional[str] = None
):
    """Attach the exclusion callback and apply the smoke-run conveniences."""
    untaught_cfg: Dict[str, Any] = config_dict.get("untaught", {}) or {}

    # Paths in the config are relative to Untaught/, so a run works the same
    # from any cwd without an environment variable standing in for the root.
    blacklist_path = blacklist_override or untaught_cfg.get("blacklist")
    if blacklist_path:
        blacklist_path = resolve_path(blacklist_path)

    config.trainer.with_callback(
        "untaught_exclusion",
        ChunkExclusionCallback(
            blacklist=blacklist_path,
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

    if untaught_cfg.get("adapt_to_gpu", True):
        adapt_to_gpu(config, untaught_cfg)

    return config, blacklist_path


def adapt_to_gpu(config, untaught_cfg: Dict[str, Any]) -> None:
    """Make the run survive whatever card SLURM handed us.

    ``build_config`` hard-codes ``compile=True`` and bf16 FSDP params, which
    assume an Ampere-or-newer GPU with plenty of memory. On a preemptible
    student partition you do not get to choose, so rather than crash after the
    queue wait, downgrade the two settings that are pure implementation detail:

    * ``torch.compile`` needs CUDA capability >= 7.0 (triton). Below that every
      block logs a multi-page "WON'T CONVERT" traceback and falls back to eager
      anyway -- turning it off removes the noise and the wasted compile time.
    * ``rank_microbatch_size`` is a *memory* knob only. The trainer accumulates
      gradients until ``global_batch_size`` tokens are reached, so shrinking it
      leaves the optimizer math, the step count and the data order untouched --
      unlike touching the batch size, which would break the comparison with the
      control run.
    """
    import torch

    if not torch.cuda.is_available():
        return

    major, minor = torch.cuda.get_device_capability()
    name = torch.cuda.get_device_name()
    total_gib = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    log.info("[untaught] GPU: %s (capability %d.%d, %.1f GiB)", name, major, minor, total_gib)

    if major < 7 and getattr(config.model, "compile", False):
        config.model.compile = False
        log.warning(
            "[untaught] %s is capability %d.%d; torch.compile needs >= 7.0. "
            "Compilation disabled (training runs eager).", name, major, minor
        )
    if major < 8:
        log.warning(
            "[untaught] %s has no native bf16; FSDP still runs param_dtype=bfloat16 "
            "but emulated and slow. Prefer an Ampere-or-newer card for real runs.",
            name,
        )

    # 12 GiB with a 100K-token vocab cannot hold 8,192-token microbatch logits.
    # Calibrated against the observed failure: 8,192 tokens OOMed on an 11.9 GiB
    # TITAN Xp, so allow ~1K microbatch tokens per 4 GiB and round down to a
    # power of two (which also keeps it dividing the 32,768-token global batch).
    budget = float(untaught_cfg.get("microbatch_gib_per_1k_tokens", 4.0))
    max_micro = int(untaught_cfg.get("max_rank_microbatch", 0)) or int(
        max(1024, (total_gib / budget) * 1024)
    )
    current = config.trainer.rank_microbatch_size
    if current > max_micro:
        # Keep it a power of two and a multiple of the longest sequence.
        adjusted = 1 << (max_micro.bit_length() - 1)
        config.trainer.rank_microbatch_size = adjusted
        log.warning(
            "[untaught] rank_microbatch_size %d -> %d to fit %.1f GiB. Gradient "
            "accumulation keeps the %d-token global batch, so the optimizer math "
            "is unchanged; only memory and speed differ.",
            current, adjusted, total_gib, config.data_loader.global_batch_size,
        )


def _summarize(config, config_dict: Dict[str, Any], blacklist_path: Optional[str]) -> None:
    tr = config_dict["trainer"]
    md = tr["max_duration"]
    print("\n" + "=" * 68)
    print("  UNTAUGHT RUN")
    print("=" * 68)
    print(f"  model            : {config_dict['model']}")
    print(f"  lr / weight decay: {config_dict['optim']['lr']} / {config_dict['optim']['weight_decay']}")
    print(f"  global batch     : {config_dict['data_loader']['global_batch_size']:,} tokens")
    # The effective value, which adapt_to_gpu may have lowered from the config.
    micro = config.trainer.rank_microbatch_size
    micro_note = "" if micro == tr["rank_microbatch_size"] else f"  (config: {tr['rank_microbatch_size']:,})"
    print(f"  rank microbatch  : {micro:,} tokens{micro_note}")
    print(f"  duration         : {md['value']} {md['unit']}")
    print(f"  save folder      : {tr['save_folder']}")
    print(f"  seed             : {config_dict.get('init_seed')}")
    if blacklist_path:
        # The entities and thresholds behind it are client_node's business; this
        # side only consumes the artifact they were resolved into.
        print(f"  blacklist        : {blacklist_path}")
    else:
        print("  blacklist        : (none) -- CONTROL run")
    print("=" * 68 + "\n")


def main(
    config_filepath: str, blacklist_override: Optional[str], check_only: bool
) -> None:
    config_dict = load_config(config_filepath)

    config = build_config(config_dict)
    config, blacklist_path = apply_untaught_config(config, config_dict, blacklist_override)

    # Fail fast on a missing blacklist, long before a SLURM job burns queue time
    # to discover it.
    if blacklist_path and not os.path.isfile(blacklist_path):
        raise FileNotFoundError(f"[untaught] blacklist file not found: {blacklist_path}")

    _summarize(config, config_dict, blacklist_path)

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
    parser = argparse.ArgumentParser(prog="framework.gpu_node.train_untaught")
    parser.add_argument("config", help="path to the training config JSON")
    parser.add_argument(
        "--blacklist",
        default=None,
        help="override untaught.blacklist (empty string forces a control run)",
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
