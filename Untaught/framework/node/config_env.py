"""Config-tree environment expansion. No heavy imports on purpose --
these run on the login node, inside SLURM jobs, and in local unit tests.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from typing import Any, Dict, List, Optional

# Untaught/framework/node/config_env.py -> Untaught/
UNTAUGHT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
ENV_SH = os.path.join(UNTAUGHT_ROOT, "framework", "env.sh")


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
    """Read a training config JSON, expand ``$VAR``/``~``, and validate.

    The one entry point every script should use: it retries once through
    ``load_env_sh`` when something is unresolved, so forgetting to source the
    environment is no longer a failure mode. The shell-out only happens on that
    retry -- an already-sourced environment costs nothing.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    config_dict = expand_env(raw)
    if unresolved_vars(config_dict) and load_env_sh():
        config_dict = expand_env(raw)

    assert_paths_resolved(config_dict)
    return config_dict
