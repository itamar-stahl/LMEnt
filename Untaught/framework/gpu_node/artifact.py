"""The blacklist artifact: the file that carries chunk ids across the fence.

Elasticsearch runs on the login node and is unreachable from the GPU nodes, so
``client_node`` resolves the entity QIDs into chunk ids there and writes them
into the run's ``save_folder``. This module is the *reading* half, and it is
deliberately dependency-free: the compute node needs nothing but json.

The writing half is ``client_node.es_blacklist.build_artifact``.
"""

from __future__ import annotations

import json
from typing import Dict

# Written next to the checkpoints, read back by the exclusion callback.
ARTIFACT_NAME = "untaught_blacklist.json"


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
