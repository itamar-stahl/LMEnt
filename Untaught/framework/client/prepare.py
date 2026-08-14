"""Resolve a run's blacklist against Elasticsearch, into the run's folder.

Run this on the login node before submitting -- the GPU nodes cannot reach the
index, so this is where the QIDs in ``untaught.blacklist`` become the concrete
chunk ids the trainer will mask:

    python -m framework.client.prepare --config configs/train_170m_no_harry_potter.yaml

The artifact lands in ``trainer.save_folder``, beside the checkpoints and
config.json, so a run and the exact exclusion it was trained with stay
together. The training job reads it and never talks to Elasticsearch.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Sequence

from framework.node.artifact import ARTIFACT_NAME
from framework.node.config_env import load_config, resolve_path, to_upstream

from .es_blacklist import build_artifact, normalize_thresholds, write_artifact


def prepare(config_path: str) -> Optional[str]:
    """Write the artifact for ``config_path``; ``None`` for a control run."""
    config_dict = load_config(config_path)
    untaught_cfg = config_dict.get("untaught", {}) or {}

    blacklist = untaught_cfg.get("blacklist")
    if not blacklist:
        print(f"[untaught] {config_path} has no blacklist -- control run, nothing to do.")
        return None

    blacklist = resolve_path(blacklist)
    if not os.path.isfile(blacklist):
        raise FileNotFoundError(f"[untaught] blacklist file not found: {blacklist}")

    # The save folder is built from the hyperparameters, so ask the upstream
    # builder rather than reimplementing the name -- it has to match exactly the
    # folder the training job will look in.
    from examples.kas.train import build_config

    save_folder = build_config(to_upstream(config_dict)).trainer.save_folder

    artifact = build_artifact(blacklist, untaught_cfg)
    path = write_artifact(artifact, os.path.join(save_folder, ARTIFACT_NAME))

    print(f"[untaught] blacklist  : {blacklist}")
    print(f"[untaught] index      : {artifact['index']} "
          f"(case_sensitive={artifact['case_sensitive']})")
    print(f"[untaught] thresholds : {artifact['thresholds']}")
    for entity in artifact["entities"]:
        print(f"[untaught]   {entity['qid']:<12} {entity['num_chunks']:>9,} chunks  "
              f"{entity['comment']}")
    print(f"[untaught] total      : {artifact['num_chunks']:,} unique chunks "
          f"({100.0 * artifact['num_chunks'] / 10_500_000:.4f}% of the corpus)")
    print(f"[untaught] wrote      : {path}")
    return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="framework.client.prepare", description=__doc__)
    parser.add_argument("--config", required=True, help="a training run config (YAML)")
    args = parser.parse_args(argv)

    # Validate the thresholds before the queries, so a typo fails in a second.
    cfg = load_config(args.config).get("untaught", {}) or {}
    normalize_thresholds(cfg)

    prepare(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
