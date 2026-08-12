"""Config-tree environment expansion. No heavy imports on purpose --
these run on the login node, inside SLURM jobs, and in local unit tests.
"""

from __future__ import annotations

import os
from typing import Any, List


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


def assert_paths_resolved(config_dict: Any) -> None:
    """Catch unset environment variables before they become silent bad paths.

    ``os.path.expandvars`` leaves ``${FOO}`` untouched when ``FOO`` is unset,
    and a glob over a literal ``${LMENT_DATASET}/...`` matches nothing -- which
    surfaces much later as a confusing empty-dataset error.
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
    if leftovers:
        raise RuntimeError(
            "Unresolved environment variables in the config -- did you source "
            "configs/env.sh?\n  " + "\n  ".join(leftovers)
        )
