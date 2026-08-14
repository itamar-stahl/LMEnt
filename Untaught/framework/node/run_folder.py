"""The run folder: the contract between the client and the compute node.

One folder per submission, ``runs/<job_name>_<date>_<time>/``, holding
everything that run used and produced:

    config.yaml               the exact config it trains on
    untaught_blacklist.json   the exact exclusion (explicitly empty for control)
    job.slurm                 what was submitted -- literal strings, no variables
    run_wrapper.sh            what the node executed
    client.log                the submission log
    log.out / log.err         the job's output
    run_environment.json      the machine it actually trained on
    checkpoints/              saved parameters, by step

The names live here, on the node side, because both halves need them and
imports only ever go client -> node. This module is deliberately
dependency-free: the compute node needs nothing but json to read it all.

``client.es_blacklist`` writes the artifact; everything here reads.
"""

from __future__ import annotations

import json
import os
from typing import Dict

# --- layout -----------------------------------------------------------------
CONFIG_NAME = "config.yaml"
ARTIFACT_NAME = "untaught_blacklist.json"
JOB_SLURM = "job.slurm"
RUN_WRAPPER = "run_wrapper.sh"
CHECKPOINTS_DIR = "checkpoints"
CLIENT_LOG = "client.log"
RUN_ENVIRONMENT = "run_environment.json"

# Untaught/framework/node/run_folder.py -> Untaught/
UNTAUGHT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


def runs_dir() -> str:
    """Where run folders live: ``$UNTAUGHT_RUNS_DIR``, else ``Untaught/runs``."""
    return os.environ.get("UNTAUGHT_RUNS_DIR") or os.path.join(UNTAUGHT_ROOT, "runs")


def resolve_run_dir(run: str) -> str:
    """Absolute path of a run folder, from a bare name or a path.

    Lets the generated wrapper name a run the way a human does -- just
    ``untaught-control-170m_20260814_212936`` -- while a path still works for
    ad-hoc use.
    """
    if os.path.isabs(run) or "/" in run or os.sep in run:
        return os.path.abspath(run)
    return os.path.join(runs_dir(), run)


def config_path(run_dir: str) -> str:
    """The config a run trains on. Always the copy inside its own folder."""
    return os.path.join(run_dir, CONFIG_NAME)


def artifact_path(run_dir: str) -> str:
    """The resolved exclusion for a run."""
    return os.path.join(run_dir, ARTIFACT_NAME)


def checkpoints_path(run_dir: str) -> str:
    """Where the trainer saves parameters, by step."""
    return os.path.join(run_dir, CHECKPOINTS_DIR)


def environment_path(run_dir: str) -> str:
    """Where the node records what it actually ran on."""
    return os.path.join(run_dir, RUN_ENVIRONMENT)


def load_artifact(path: str) -> Dict[int, str]:
    """Read an artifact back as the ``{chunk_id: qid}`` lookup used per batch.

    A chunk mentioning several blacklisted entities is credited to the first
    one that matched -- it is excluded either way.
    """
    with open(path, "r", encoding="utf-8") as f:
        artifact = json.load(f)

    blacklist: Dict[int, str] = {}
    for entity in artifact.get("entities", []):
        for chunk_id in entity.get("chunk_ids", []):
            blacklist.setdefault(int(chunk_id), entity["qid"])
    return blacklist
