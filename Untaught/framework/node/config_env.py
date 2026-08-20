"""Everything about reading a run config: YAML parsing, ``${VAR}`` expansion,
translation to upstream's nested schema, and path resolution.

No heavy imports on purpose -- these run on the login node, inside SLURM jobs,
and in local unit tests, before torch or olmo_core are anywhere in sight.

Also a tiny CLI, so shell scripts can read one value without parsing YAML:

    python -m framework.node.config_env <config> <dotted.key>
"""

from __future__ import annotations

import os
import subprocess
import sys
from typing import Any, Dict, List, Optional

# Untaught/framework/node/config_env.py -> Untaught/
UNTAUGHT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
ENV_SH = os.path.join(UNTAUGHT_ROOT, "framework", "env.sh")


def read_config_file(path: str) -> Dict[str, Any]:
    """Parse a run config. YAML, so the file can carry real ``#`` comments.

    YAML is a superset of JSON, so a ``.json`` config still parses -- handy
    while both formats are around.
    """
    try:
        import yaml
    except ImportError as e:  # pragma: no cover - PyYAML ships with transformers
        raise RuntimeError(
            "[untaught] PyYAML is required to read run configs. "
            "It is part of the lment env: `. ./activate_env.sh` first."
        ) from e

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError(f"[untaught] {path} did not parse to a mapping")
    return config


def to_upstream(config: Dict[str, Any], save_folder: str) -> Dict[str, Any]:
    """Translate our flat ``job``/``train`` groups into upstream's nested schema.

    ``examples.kas.train.build_config`` wants ``config["dataset"]["vsl_curriculum"]
    ["num_cycles"]`` and friends. That shape is upstream's business, not a
    structure worth reproducing by hand in every config file, so the files stay
    flat and grouped by prefix and this function does the mapping.

    ``save_folder`` is a parameter, not a config field: every submission gets its
    own run folder (``runs/<job_name>_<date>_<time>``), so where checkpoints land
    is decided per run -- the trainer derives it from where its config copy
    lives, never from the config's contents.
    """
    job = config["job"]
    train = config["train"]

    paths = job["dataset_paths"]
    if isinstance(paths, str):
        paths = [paths]

    return {
        "model": train["model"],
        "init_seed": train["init_seed"],
        "optim": {
            "lr": train["optim_lr"],
            "weight_decay": train["optim_weight_decay"],
        },
        "dataset": {
            "name": train["dataset_name"],
            "paths": paths,
            "work_dir": job["dataset_cache"],
            "max_sequence_length": train["dataset_max_sequence_length"],
            "min_sequence_length": train["dataset_min_sequence_length"],
            "include_instance_metadata": train["dataset_include_instance_metadata"],
            "vsl_curriculum": {
                "name": train["vsl_curriculum"],
                "num_cycles": train["vsl_num_cycles"],
                "balanced": train["vsl_balanced"],
            },
        },
        "data_loader": {
            "global_batch_size": train["data_global_batch_size"],
            "seed": train["data_seed"],
            "num_workers": train["data_num_workers"],
            "prefetch_factor": train["data_prefetch_factor"],
        },
        "trainer": {
            "save_folder": save_folder,
            "rank_microbatch_size": train["data_rank_microbatch_size"],
            "save_overwrite": train["checkpoint_save_overwrite"],
            "metrics_collect_interval": train["metrics_collect_interval"],
            "cancel_check_interval": train["cancel_check_interval"],
            "max_duration": {
                "value": train["max_duration_value"],
                "unit": train["max_duration_unit"],
            },
            "callbacks": {
                "lr_scheduler": {"warmup_steps": train["optim_warmup_steps"]},
                "grad_clipper": {"max_grad_norm": train["optim_max_grad_norm"]},
                "checkpointer": {
                    "save_interval": train["checkpoint_save_interval"],
                    "ephemeral_save_interval": train["checkpoint_ephemeral_save_interval"],
                    "save_async": train["checkpoint_save_async"],
                },
                "wandb": {"cancel_check_interval": train["wandb_cancel_check_interval"]},
                "downstream_evaluator": {
                    "tasks": train["eval_tasks"],
                    "eval_interval": train["eval_interval"],
                },
            },
        },
    }


def expand_env(obj: Any) -> Any:
    """Recursively expand ``$VAR`` / ``~`` in every string of a config tree.

    Lets the configs refer to ``${LMENT_DATASET}`` instead of hard-coding one
    cluster's paths, without touching upstream ``build_config``.
    """
    if isinstance(obj, str):
        return os.path.expanduser(os.path.expandvars(obj))
    if isinstance(obj, dict):
        return {k: expand_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [expand_env(v) for v in obj]
    return obj


def resolve_path(path: str) -> str:
    """Absolute path for a config entry, taking relative ones from ``Untaught/``.

    Lets a config say ``"blacklists/harry_potter.json"`` and mean the same file
    whatever directory the job was launched from -- no environment variable
    needed to name the project root.
    """
    expanded = os.path.expanduser(os.path.expandvars(str(path)))
    if os.path.isabs(expanded):
        return expanded
    return os.path.normpath(os.path.join(UNTAUGHT_ROOT, expanded))


def unresolved_vars(config_dict: Any) -> List[str]:
    """Return ``path = value`` strings for every config entry still holding a
    ``$VAR`` reference. ``os.path.expandvars`` leaves ``${FOO}`` untouched when
    ``FOO`` is unset, and a glob over a literal ``${LMENT_DATASET}/...`` matches
    nothing -- which surfaces much later as a confusing empty-dataset error.
    """
    leftovers: List[str] = []

    def walk(node: Any, path: str) -> None:
        if isinstance(node, str):
            if "${" in node or (node.startswith("$") and len(node) > 1):
                leftovers.append(f"{path} = {node}")
        elif isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{path}.{k}")
        elif isinstance(node, list):
            for i, v in enumerate(node):
                walk(v, f"{path}[{i}]")

    walk(config_dict, "config")
    return leftovers


def assert_paths_resolved(config_dict: Any) -> None:
    """Catch unset environment variables before they become silent bad paths."""
    leftovers = unresolved_vars(config_dict)
    if leftovers:
        raise RuntimeError(
            f"Unresolved environment variables in the config. Sourcing {ENV_SH} "
            "was attempted automatically and did not define them -- source it "
            "yourself and check for typos:\n"
            f"  cd {UNTAUGHT_ROOT} && . ./activate_env.sh\n  "
            + "\n  ".join(leftovers)
        )


def _exported_names(env_sh_text: str) -> List[str]:
    """The variable names ``env.sh`` exports, read off its own ``export`` lines.

    Only these are copied out of the helper shell, so nothing else about that
    shell's environment (PWD, SHLVL, ...) leaks into this process.
    """
    names: List[str] = []
    for line in env_sh_text.splitlines():
        line = line.strip()
        if not line.startswith("export "):
            continue
        for token in line[len("export "):].split():
            name = token.split("=", 1)[0]
            if name.isidentifier() and name not in names:
                names.append(name)
    return names


def load_env_sh(path: Optional[str] = None, verbose: bool = True) -> List[str]:
    """Source ``framework/env.sh`` in a helper shell and copy its exports here.

    So a bare ``python tests/verify_chunk_alignment.py`` works the same as one
    run after ``. ./activate_env.sh``. Variables already set in this process win,
    exactly like the ``: "${FOO:=default}"`` defaults inside env.sh, so an
    explicit ``LMENT_DATASET=/other python ...`` still overrides.

    Returns the names it set; empty if the file or a POSIX shell is missing.
    """
    path = path or ENV_SH
    if not os.path.isfile(path):
        return []

    try:
        with open(path, "r", encoding="utf-8") as f:
            wanted = _exported_names(f.read())
    except OSError:
        return []
    if not wanted:
        return []

    child_env = dict(os.environ)
    # env.sh takes UNTAUGHT_ROOT from the environment; we know the real answer
    # from this file's location, so hand it over rather than let it default.
    child_env.setdefault("UNTAUGHT_ROOT", UNTAUGHT_ROOT)

    try:
        proc = subprocess.run(
            ["sh", "-c", '. "$1" >&2 || exit 1; env -0', "sh", path],
            capture_output=True,
            env=child_env,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        return []  # no POSIX shell (e.g. a Windows checkout) -- caller still errors out
    if proc.returncode != 0:
        return []

    child_vars: Dict[str, str] = {}
    for entry in proc.stdout.decode("utf-8", "replace").split("\0"):
        name, sep, value = entry.partition("=")
        if sep and name in wanted:
            child_vars[name] = value

    applied = [n for n in wanted if n in child_vars and n not in os.environ]
    for name in applied:
        os.environ[name] = child_vars[name]

    if applied and verbose:
        print(
            f"[untaught] environment was not sourced; loaded {len(applied)} "
            f"variable(s) from {path}",
            file=sys.stderr,
        )
    return applied


def load_config(config_path: str) -> Any:
    """Read a run config, expand ``$VAR``/``~`` in it, and validate.

    The one entry point every script should use: it retries once through
    ``load_env_sh`` when something is unresolved, so forgetting to source the
    environment is no longer a failure mode. The shell-out only happens on that
    retry -- an already-sourced environment costs nothing.
    """
    raw = read_config_file(config_path)

    config_dict = expand_env(raw)
    if unresolved_vars(config_dict) and load_env_sh():
        config_dict = expand_env(raw)

    assert_paths_resolved(config_dict)
    return config_dict


if __name__ == "__main__":
    # Tiny getter so shell scripts can read one config value without parsing
    # YAML themselves:  python -m framework.node.config_env <config> <dotted.key>
    if len(sys.argv) != 3:
        print("usage: python -m framework.node.config_env <config> <dotted.key>",
              file=sys.stderr)
        raise SystemExit(2)
    node: Any = load_config(sys.argv[1])
    for part in sys.argv[2].split("."):
        if not isinstance(node, dict) or part not in node:
            print(f"[untaught] no key '{sys.argv[2]}' in {sys.argv[1]}", file=sys.stderr)
            raise SystemExit(1)
        node = node[part]
    print(node)
