"""Train an LMEnt model with a concept held out of the loss.

This is a thin wrapper around ``OLMo-core/src/examples/kas/train.py``.  It reuses
that file's ``build_config`` verbatim so the model, optimizer, dataset, VSL
curriculum and data order are exactly the upstream ones, then attaches one extra
callback that masks blacklisted chunks.

Normally launched by a run folder's ``run_wrapper.sh`` on the config copy
inside that folder:

    torchrun --standalone --nproc-per-node=1 \
        framework/node/train_untaught.py <run_dir>/config.yaml

The run folder (= the config copy's directory) is where everything lives: the
blacklist artifact this process reads, and the checkpoints/ subfolder it saves
parameters into. Config schema is three flat groups -- "job", "train",
"untaught"; see configs/*.yaml. ``config_env.to_upstream`` maps them onto the
nested dict ``build_config`` expects.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import re
import sys
from datetime import datetime
from typing import Any, Dict, Optional, Sequence, cast

log = logging.getLogger(__name__)

# Launched via a file path (run_wrapper.sh does), relative imports fail and
# `framework` must be found on sys.path. config_env is deliberately free of
# heavy dependencies, so it can come before the OLMo-core bootstrap that
# everything else here needs.
try:
    from .config_env import load_config, load_env_sh, resolve_path, to_upstream
except ImportError:  # pragma: no cover
    # Untaught/framework/node/train_untaught.py -> Untaught/ (framework's parent)
    sys.path.insert(
        0,
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    )
    from framework.node.config_env import (
        load_config,
        load_env_sh,
        resolve_path,
        to_upstream,
    )


def _bootstrap_olmo_core() -> None:
    """Put ``OLMo-core/src`` on ``sys.path`` so ``examples.kas.train`` imports.

    Tries the environment first, then this checkout's own sibling directory, and
    takes the first that actually exists -- an ``OLMO_CORE_SRC`` pointing at
    another machine's layout must not stop a working local checkout from
    importing.
    """
    if not os.environ.get("OLMO_CORE_SRC"):
        # Nothing sourced the environment (a bare `python -m framework.node...`)? Do it here.
        load_env_sh()

    # Untaught/framework/node/train_untaught.py -> repo root -> OLMo-core/src
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..")
    )
    candidates = [
        os.environ.get("OLMO_CORE_SRC"),
        os.path.join(repo_root, "OLMo-core", "src"),
    ]

    for candidate in candidates:
        if candidate and os.path.isdir(candidate):
            src = os.path.abspath(candidate)
            if src not in sys.path:
                sys.path.insert(0, src)
            return

    raise RuntimeError(
        "Could not find OLMo-core sources. Tried: "
        + ", ".join(repr(c) for c in candidates if c)
        + ". Set OLMO_CORE_SRC to <LMEnt>/OLMo-core/src."
    )


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
    from .run_folder import (
        CHECKPOINTS_DIR,
        artifact_path,
        checkpoints_path,
        config_path,
        environment_path,
        resolve_run_dir,
        runs_dir,
    )
except ImportError:  # pragma: no cover
    from framework.node.exclusion import ChunkExclusionCallback
    from framework.node.run_folder import (
        CHECKPOINTS_DIR,
        artifact_path,
        checkpoints_path,
        config_path,
        environment_path,
        resolve_run_dir,
        runs_dir,
    )


# Fields that may differ between two submissions of the *same* experiment:
# how it was scheduled, how often it checkpoints, and how often it logs. Change
# any of these between resubmissions and it is still the same training run.
# Everything else -- the model, the optimizer, the data, the seeds, the whole
# untaught block -- must match, or the runs are different experiments and
# continuing one from the other would be silent corruption of both.
RESUME_IGNORED_FIELDS: Dict[str, set] = {
    "job": {
        "partition",
        "resume_from_previous_run",
        "max_time_minutes",
        "nodes",
        "ntasks",
        "cpu_mem_mb",
        "cpus_per_task",
        "gpus",
    },
    "train": {
        "checkpoint_save_interval",
        "checkpoint_ephemeral_save_interval",
        "checkpoint_save_async",
        "checkpoint_save_overwrite",
        "metrics_collect_interval",
        "cancel_check_interval",
        "wandb_cancel_check_interval",
        "eval_tasks",
        "eval_interval",
    },
}


# Fields where an empty value means "no preference", and so is compatible with
# any value the other run had. job.constraint is the case: two runs that both
# demanded a card and demanded *different* cards are not the same experiment,
# but a run that demanded nothing can continue, or be continued by, either.
RESUME_EMPTY_MATCHES_ANY = {("job", "constraint")}


def resume_identity(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """The part of a config that defines *which experiment* this is.

    Two runs may continue one another only if these are equal, field for field.
    """
    identity: Dict[str, Any] = {}
    for group, values in config_dict.items():
        if isinstance(values, dict):
            ignored = RESUME_IGNORED_FIELDS.get(group, set())
            identity[group] = {k: v for k, v in values.items() if k not in ignored}
        else:
            identity[group] = values
    return identity


def _identity_differences(ours: Dict[str, Any], theirs: Dict[str, Any]) -> list:
    """Which fields disqualify a candidate -- so the log can say why."""
    differences = []
    for group in sorted(set(ours) | set(theirs)):
        a, b = ours.get(group, {}), theirs.get(group, {})
        if not isinstance(a, dict) or not isinstance(b, dict):
            if a != b:
                differences.append(group)
            continue
        for key in sorted(set(a) | set(b)):
            empty_matches_any = (group, key) in RESUME_EMPTY_MATCHES_ANY
            # A key an older config never had reads as "no preference" for those
            # fields, and as a genuine difference for every other field.
            missing = "" if empty_matches_any else "<missing>"
            ours_value, theirs_value = a.get(key, missing), b.get(key, missing)
            if ours_value == theirs_value:
                continue
            if empty_matches_any and (not ours_value or not theirs_value):
                continue
            differences.append(f"{group}.{key}")
    return differences


def latest_checkpoint_dir(run_dir: str) -> Optional[str]:
    """The folder of ``stepN`` checkpoints in a run, if it has any.

    Returns the *parent* of the step directories, which is what upstream's
    ``maybe_load_checkpoint`` wants -- it picks the newest step inside.
    """
    best: Optional[str] = None
    best_step = -1
    patterns = (
        os.path.join(run_dir, CHECKPOINTS_DIR, "*", "step*"),
        os.path.join(run_dir, CHECKPOINTS_DIR, "step*"),
    )
    for pattern in patterns:
        for step_dir in glob.glob(pattern):
            if not os.path.isdir(step_dir):
                continue
            match = re.fullmatch(r"step(\d+)", os.path.basename(step_dir))
            if match and int(match.group(1)) > best_step:
                best_step = int(match.group(1))
                best = os.path.dirname(step_dir)
    return best


def find_previous_checkpoint(
    run_dir: str, config_dict: Dict[str, Any]
) -> Optional[str]:
    """The newest earlier run of this experiment that left a checkpoint.

    Upstream already resumes from the save folder if a checkpoint is there, but
    every submission gets a *fresh* run folder -- so after a preemption, or when
    the 1-day studentkillable limit ends a job mid-epoch, our own checkpoints/
    is empty and there is nothing for it to find. This looks one folder back.

    A candidate qualifies only if its folder name starts with this config's
    ``job.name`` **and** its saved config.yaml matches ours field for field,
    ignoring the scheduling and logging knobs in ``RESUME_IGNORED_FIELDS``. The
    comparison is against the config each run actually trained on -- never
    against a directory name, which encodes only a handful of hyperparameters
    (upstream's is model/lr/batch/wd/duration-*value*, so it cannot even tell
    "1 epoch" from "1 step", let alone a different seed or a different
    blacklist).
    """
    job_name = config_dict["job"]["name"]
    ours = resume_identity(config_dict)

    candidates = sorted(
        (
            d for d in glob.glob(os.path.join(runs_dir(), f"{job_name}_*"))
            if os.path.isdir(d) and os.path.normpath(d) != os.path.normpath(run_dir)
        ),
        reverse=True,  # timestamps sort lexicographically: newest first
    )

    for candidate in candidates:
        previous_config = config_path(candidate)
        if not os.path.isfile(previous_config):
            continue

        checkpoints = latest_checkpoint_dir(candidate)
        if checkpoints is None:
            continue

        try:
            theirs = resume_identity(load_config(previous_config))
        except Exception as e:  # a corrupt or half-written config is not a match
            log.warning("[untaught] cannot read %s: %s", previous_config, e)
            continue

        differences = _identity_differences(ours, theirs)
        if not differences:
            return checkpoints

        log.info(
            "[untaught] not resuming from %s: config differs in %s",
            os.path.basename(candidate),
            ", ".join(differences[:5]) + ("..." if len(differences) > 5 else ""),
        )

    return None


def write_run_environment(run_dir: str, config, resumed_from: Optional[str]) -> str:
    """Record the machine this run actually trained on.

    adapt_to_gpu quietly adjusts to whatever card SLURM hands out -- eager
    instead of compiled below capability 7.0, a smaller microbatch on a small
    card, emulated bf16 on pre-Ampere. None of that changes the optimizer math,
    but the control and ablated halves of a pair can land on *different* GPU
    types, and comparing two runs is only sound if you can see that they did.
    So each run states its own conditions, next to its config and its
    checkpoints.
    """
    import platform

    import torch

    environment: Dict[str, Any] = {
        "recorded": datetime.now().astimezone().isoformat(timespec="seconds"),
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "gpu": None,
        "compile": bool(getattr(config.model, "compile", False)),
        "rank_microbatch_size": config.trainer.rank_microbatch_size,
        "global_batch_size": config.data_loader.global_batch_size,
        "param_dtype": str(getattr(config.model.dp_config, "param_dtype", None)),
        "resumed_from": resumed_from,
    }
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability()
        environment["gpu"] = {
            "name": torch.cuda.get_device_name(),
            "capability": f"{major}.{minor}",
            "memory_gib": round(
                torch.cuda.get_device_properties(0).total_memory / (1024 ** 3), 1
            ),
            "native_bf16": major >= 8,
            "triton_compilable": major >= 7,
        }

    path = environment_path(run_dir)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(environment, f, indent=2)
        f.write("\n")
    log.info("[untaught] recorded run environment in %s", path)
    return path


def apply_untaught_config(config, config_dict: Dict[str, Any], run_dir: str):
    """Attach the exclusion callback and apply the smoke-run conveniences."""
    untaught_cfg: Dict[str, Any] = config_dict.get("untaught", {}) or {}

    # Paths in the config are relative to Untaught/, so a run works the same
    # from any cwd without an environment variable standing in for the root.
    blacklist_path = untaught_cfg.get("blacklist")
    if blacklist_path:
        blacklist_path = resolve_path(blacklist_path)

    config.trainer.with_callback(
        "untaught_exclusion",
        ChunkExclusionCallback(
            blacklist=blacklist_path,
            # The resolved exclusion sits in the run folder, beside the config.
            artifact_path=artifact_path(run_dir),
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


def _summarize(
    config,
    config_dict: Dict[str, Any],
    run_dir: str,
    blacklist_path: Optional[str],
    resumed_from: Optional[str] = None,
) -> None:
    train = config_dict["train"]
    print("\n" + "=" * 68)
    print("  UNTAUGHT RUN")
    print("=" * 68)
    print(f"  model            : {train['model']}")
    print(f"  lr / weight decay: {train['optim_lr']} / {train['optim_weight_decay']}")
    print(f"  global batch     : {train['data_global_batch_size']:,} tokens")
    # The effective value, which adapt_to_gpu may have lowered from the config.
    micro = config.trainer.rank_microbatch_size
    configured = train["data_rank_microbatch_size"]
    micro_note = "" if micro == configured else f"  (config: {configured:,})"
    print(f"  rank microbatch  : {micro:,} tokens{micro_note}")
    print(f"  duration         : {train['max_duration_value']} {train['max_duration_unit']}")
    print(f"  run folder       : {run_dir}")
    print(f"  checkpoints      : {config.trainer.save_folder}")
    print(f"  seed             : {config.init_seed}")
    if resumed_from:
        print(f"  resuming from    : {resumed_from}")
    else:
        print("  resuming from    : (nothing found) -- random init")
    if blacklist_path:
        # The entities and thresholds behind it are framework.client's business; this
        # side only consumes the artifact they were resolved into.
        print(f"  blacklist        : {blacklist_path}")
    else:
        print("  blacklist        : (none) -- CONTROL run")
    print("=" * 68 + "\n")


def main(target: str, check_only: bool) -> None:
    """Train the run named by ``target``.

    Normally that is a run folder -- its name, or its path -- holding the
    config.yaml to train on and the blacklist artifact to mask by. For
    ``--check`` a raw ``configs/*.yaml`` is also accepted, so a config can be
    validated before any run folder exists; it gets a throwaway folder rather
    than littering the repo.
    """
    if target.endswith((".yaml", ".yml")):
        if not check_only:
            raise SystemExit(
                "[untaught] expected a run folder, not a config file. A run is "
                "trained from its own folder, which sub_builder.sh creates:\n"
                "  . ./framework/client/sub_builder.sh <config.yaml>"
            )
        import tempfile

        run_dir = tempfile.mkdtemp(prefix="untaught-check-")
        config_file = os.path.abspath(target)
    else:
        run_dir = resolve_run_dir(target)
        config_file = config_path(run_dir)
        if not os.path.isfile(config_file):
            raise FileNotFoundError(
                f"[untaught] no config.yaml in {run_dir}. Run folders are built by:"
                "\n  . ./framework/client/sub_builder.sh <config.yaml>"
            )

    config_dict = load_config(config_file)
    config = build_config(
        to_upstream(config_dict, save_folder=checkpoints_path(run_dir))
    )

    # Resume. Upstream loads from the save folder when a checkpoint is there
    # (load_strategy="if_available"), which covers a job restarted in place; the
    # load_path below covers the usual case on a preemptible partition, where
    # this is a new run folder continuing the last one.
    resumed_from = None
    if config_dict["job"].get("resume_from_previous_run", True) and not glob.glob(
        os.path.join(str(config.trainer.save_folder), "step*")
    ):
        resumed_from = find_previous_checkpoint(run_dir, config_dict)
        if resumed_from:
            config.trainer.load_path = resumed_from
    # build_config ignores init_seed (ExperimentConfig defaults it), so apply the
    # config's value here -- the control and ablated runs must share it.
    config.init_seed = config_dict["train"]["init_seed"]
    config, blacklist_path = apply_untaught_config(config, config_dict, run_dir)

    # Fail fast on a missing blacklist, long before a SLURM job burns queue time
    # to discover it.
    if blacklist_path and not os.path.isfile(blacklist_path):
        raise FileNotFoundError(f"[untaught] blacklist file not found: {blacklist_path}")

    _summarize(config, config_dict, run_dir, blacklist_path, resumed_from)

    if check_only:
        print("[untaught] --check passed: config builds and blacklist is readable.")
        return

    write_run_environment(run_dir, config, resumed_from)

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
    parser = argparse.ArgumentParser(prog="framework.node.train_untaught")
    parser.add_argument(
        "run",
        help="a run folder: its name under runs/, or a path "
             "(with --check, a configs/*.yaml is also accepted)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="build the config and validate the blacklist, then exit without training",
    )
    return parser.parse_args(argv)


def route_logs_to_stdout() -> None:
    """Send INFO logging to stdout, leaving stderr for actual problems.

    Python's logging defaults to stderr, and ``prepare_training_environment()``
    configures olmo_core's logger the same way -- so without this, the entire
    training narrative (every step's loss, every checkpoint) lands in log.err
    while log.out holds almost nothing. Nothing there is an error, but a log.err
    full of routine INFO lines is a file nobody can skim for trouble.

    Warnings and tracebacks still go to stderr: torch and the C libraries write
    there directly, and logging's own WARNING+ records are re-pointed to stderr
    below. So log.err ends up holding exactly what deserves attention.
    """
    root = logging.getLogger()
    for handler in root.handlers:
        if isinstance(handler, logging.StreamHandler) and handler.stream is sys.stderr:
            handler.setStream(sys.stdout)
            handler.addFilter(lambda record: record.levelno < logging.WARNING)

    problems = logging.StreamHandler(sys.stderr)
    problems.setLevel(logging.WARNING)
    problems.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    )
    root.addHandler(problems)


if __name__ == "__main__":
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        stream=sys.stdout,
    )

    if args.check:
        main(args.run, check_only=True)
    else:
        prepare_training_environment()
        # After olmo_core has configured logging, not before.
        route_logs_to_stdout()
        try:
            main(args.run, check_only=False)
        finally:
            teardown_training_environment()
