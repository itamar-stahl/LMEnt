"""Turn a set of Wikidata QIDs into the chunk ids to hold out.

This is the "entity-based retrieval" of LMEnt paper section 3.2 used for a new
purpose: instead of retrieving chunks that mention an entity so we can read
them, we retrieve them so we can refuse to train on them.

``fetch_chunk_ids`` returns those ids as a numpy array and nothing is written to
disk -- the trainer calls it once at ``pre_train`` and keeps the array in
memory. Because ``chunk_id`` is the dataset instance index (see
``exclusion.py``), that array is all the trainer needs; the tokenized dataset
and its caches are never touched.

The commands here are for inspection only.

Usage
-----
    # what does the corpus think "Harry Potter" is?
    python -m framework.es_blacklist resolve --name "Harry Potter"

    # how much would this run remove, and what does it look like?
    python -m framework.es_blacklist count \
        --config configs/train_170m_no_harry_potter.json --preview 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# Paper section 5.2 / Table 4: the thresholds validated on a 60-entity dev set.
DEFAULT_THRESHOLDS: Dict[str, float] = {
    "hyperlinks": 1.0,
    "entity_linking": 0.6,
    "coref": 0.6,
    "coref_cluster": 0.6,
}

DEFAULT_INDEX = "lment_cs"

# The two indexes the suite's README restores:
#   enwiki_case_sensitive -> lment_cs      enwiki -> lment_ci
CASE_SENSITIVE_INDEX = "lment_cs"
CASE_INSENSITIVE_INDEX = "lment_ci"

# Written next to the checkpoints, read back by the exclusion callback.
ARTIFACT_NAME = "untaught_blacklist.json"


def index_for(case_sensitive: bool) -> str:
    """Which ES index a run should retrieve from."""
    return CASE_SENSITIVE_INDEX if case_sensitive else CASE_INSENSITIVE_INDEX


# --------------------------------------------------------------------------- #
# Elasticsearch
# --------------------------------------------------------------------------- #
def get_esclient(
    scheme: Optional[str] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
    password: Optional[str] = None,
):
    """Mirrors ``retrieval-index/create_es_index.py::get_esclient``."""
    from elasticsearch import Elasticsearch

    scheme = scheme or os.environ.get("ES_SCHEME", "https")
    host = host or os.environ.get("ES_HOST", "localhost")
    port = int(port or os.environ.get("ES_PORT", 9200))
    password = password if password is not None else os.environ.get("ES_PASSWORD", "")

    if not password:
        print(
            "[untaught] warning: no ES password. Set ES_PASSWORD or pass --es-password.",
            file=sys.stderr,
        )

    warnings.filterwarnings("ignore", message=".*verify_certs.*")
    return Elasticsearch(
        f"{scheme}://{host}:{port}",
        basic_auth=("elastic", password),
        request_timeout=300,
        max_retries=10,
        retry_on_timeout=True,
        verify_certs=False,
        ssl_show_warn=False,
    )


def load_blacklist(path: str) -> List[Dict[str, str]]:
    """Read a blacklist file: the static list of entities to hold out.

        {"entities": [{"qid": "Q8337", "comment": "the franchise"}, ...]}

    A bare ``"Q8337"`` in place of the entity object is accepted too. Returns
    the normalised entity list; ``[e["qid"] for e in ...]`` gives the QIDs.

    Retrieval *confidence* is not here -- that is a property of the training
    run, so it lives in the run's config under ``framework.thresholds``.
    """
    with open(path, "r", encoding="utf-8") as f:
        spec = json.load(f)

    entities: List[Dict[str, str]] = []
    for entry in spec.get("entities", []):
        if isinstance(entry, str):
            entities.append({"qid": entry, "comment": ""})
        else:
            entities.append({"qid": str(entry["qid"]), "comment": entry.get("comment", "")})

    if not entities:
        raise ValueError(f"[untaught] no 'entities' in blacklist file: {path}")
    return entities


def normalize_thresholds(thresholds: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """Validate a config's ``framework.thresholds`` and fill in the defaults.

    Merged **per key**, so naming one source leaves the other three at the
    paper's values. Raises on an unknown source name, a non-number or a value
    outside [0, 1]: a typo that silently fell back to the default would quietly
    change which chunks the run holds out.
    """
    merged = dict(DEFAULT_THRESHOLDS)
    for source, value in (thresholds or {}).items():
        if source not in DEFAULT_THRESHOLDS:
            raise ValueError(
                f"[untaught] unknown threshold source '{source}' in untaught.thresholds. "
                f"Known sources: {', '.join(sorted(DEFAULT_THRESHOLDS))}"
            )
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"[untaught] threshold '{source}' must be a number, got {value!r}")
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(
                f"[untaught] threshold '{source}' = {value} is outside [0, 1]; "
                "mention scores are confidences"
            )
        merged[source] = float(value)
    return merged


def fetch_chunk_ids(
    qids: Sequence[str],
    es=None,
    index: Optional[str] = None,
    thresholds: Optional[Dict[str, float]] = None,
    scroll_size: int = 5000,
) -> np.ndarray:
    """Every chunk id that mentions one of ``qids``, as a sorted unique array.

    This is the whole blacklist: a few thousand int64s for a single entity, so
    the trainer calls it at ``pre_train`` and keeps the result in memory rather
    than staging a file. Needs Elasticsearch to be reachable from wherever the
    caller runs -- on a cluster that means the compute node, not just the login
    node.
    """
    from elasticsearch.helpers import scan

    es = es if es is not None else get_esclient()
    index = index or os.environ.get("ES_INDEX", DEFAULT_INDEX)
    query = build_entity_query(qids, thresholds or DEFAULT_THRESHOLDS)

    chunk_ids: List[int] = []
    for hit in scan(
        es,
        index=index,
        query={"query": query, "_source": ["chunk_id"]},
        size=scroll_size,
        preserve_order=False,
    ):
        cid = hit["_source"].get("chunk_id")
        if cid is not None:
            chunk_ids.append(int(cid))

    return np.unique(np.asarray(chunk_ids, dtype=np.int64))


def build_entity_query(qids: Sequence[str], thresholds: Dict[str, float]) -> Dict[str, Any]:
    """Match chunks with >=1 mention whose candidate list contains one of ``qids``
    at or above the per-source score thresholds.

    ``entities`` is a nested field and ``entities.candidates`` is nested inside
    it, so this needs two levels of ``nested`` query. The threshold clauses are
    a ``should`` with ``minimum_should_match: 1`` -- matching the paper, where a
    mention counts if it satisfies *at least one* of the source thresholds.
    """
    score_clauses = [
        {"range": {f"entities.candidates.scores_by_source.{source}": {"gte": value}}}
        for source, value in thresholds.items()
    ]

    return {
        "nested": {
            "path": "entities",
            "query": {
                "nested": {
                    "path": "entities.candidates",
                    "query": {
                        "bool": {
                            "filter": [
                                {"terms": {"entities.candidates.qid": list(qids)}},
                                {"bool": {"should": score_clauses, "minimum_should_match": 1}},
                            ]
                        }
                    },
                }
            },
        }
    }


def build_artifact(
    blacklist_path: str,
    thresholds: Optional[Dict[str, Any]] = None,
    case_sensitive: bool = True,
    es=None,
) -> Dict[str, Any]:
    """Resolve a blacklist against Elasticsearch into a self-describing dict.

    Written on a machine that can reach ES (the login node) and read back on
    one that cannot (the GPU node), so it records not just the chunk ids but
    every input that produced them -- index, thresholds, per-entity counts --
    and stays readable in a text editor.

    Queries one QID at a time so each entity's contribution is visible, which
    is also what makes the loaded dict able to say *why* a chunk was excluded.
    """
    entities = load_blacklist(blacklist_path)
    thresholds = normalize_thresholds(thresholds)
    index = index_for(case_sensitive)
    es = es if es is not None else get_esclient()

    resolved: List[Dict[str, Any]] = []
    total: Dict[int, str] = {}
    for entity in entities:
        ids = fetch_chunk_ids([entity["qid"]], es=es, index=index, thresholds=thresholds)
        resolved.append(
            {
                "qid": entity["qid"],
                "comment": entity["comment"],
                "num_chunks": int(ids.size),
                "chunk_ids": [int(i) for i in ids],
            }
        )
        for chunk_id in ids:
            total.setdefault(int(chunk_id), entity["qid"])

    return {
        "generated": datetime.now().astimezone().isoformat(timespec="seconds"),
        "blacklist": blacklist_path,
        "index": index,
        "case_sensitive": case_sensitive,
        "thresholds": thresholds,
        "num_chunks": len(total),
        "entities": resolved,
    }


def write_artifact(artifact: Dict[str, Any], path: str) -> str:
    """Save the artifact as indented JSON, creating its folder if needed."""
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2)
        f.write("\n")
    return path


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


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
def cmd_resolve(args: argparse.Namespace) -> int:
    """Find which QIDs the corpus actually uses for a surface name.

    Never hard-code a QID you have not checked against this index -- the whole
    experiment rests on picking the right ones.
    """
    es = get_esclient(args.es_scheme, args.es_host, args.es_port, args.es_password)

    body = {
        "size": 0,
        "query": {
            "nested": {
                "path": "entities",
                "query": {
                    "nested": {
                        "path": "entities.candidates",
                        "query": {"match_phrase": {"entities.candidates.name": args.name}},
                    }
                },
            }
        },
        "aggs": {
            "ents": {
                "nested": {"path": "entities"},
                "aggs": {
                    "cands": {
                        "nested": {"path": "entities.candidates"},
                        "aggs": {
                            "matching": {
                                "filter": {
                                    "match_phrase": {"entities.candidates.name": args.name}
                                },
                                "aggs": {
                                    "qids": {
                                        "terms": {
                                            "field": "entities.candidates.qid",
                                            "size": args.top,
                                        }
                                    }
                                },
                            }
                        },
                    }
                },
            }
        },
    }

    resp = es.search(index=args.es_index, body=body)
    buckets = resp["aggregations"]["ents"]["cands"]["matching"]["qids"]["buckets"]

    if not buckets:
        print(f"No candidate entity named '{args.name}' found in index '{args.es_index}'.")
        return 1

    print(f"QIDs whose candidate name matches '{args.name}' (index '{args.es_index}'):\n")
    print(f"  {'QID':<14} {'mentions':>12}")
    print(f"  {'-' * 14} {'-' * 12}")
    for b in buckets:
        print(f"  {b['key']:<14} {b['doc_count']:>12,}")
    print(
        "\nPick the QIDs you mean and put them in an entities JSON file. "
        "A franchise, its characters and its individual works are all separate QIDs."
    )
    return 0


def cmd_count(args: argparse.Namespace) -> int:
    """Report what a training config would hold out, without training anything.

    Takes the config rather than the blacklist file so the count uses the same
    entities *and* the same thresholds the run will use.
    """
    try:
        from .config_env import load_config, resolve_path
    except ImportError:  # pragma: no cover - file-path launch
        from framework.config_env import load_config, resolve_path

    untaught_cfg = load_config(args.config).get("untaught", {}) or {}
    blacklist = untaught_cfg.get("blacklist")
    if not blacklist:
        print(f"[untaught] {args.config} has no blacklist -- it is a control run.")
        return 0

    blacklist = resolve_path(blacklist)
    entities = load_blacklist(blacklist)
    thresholds = normalize_thresholds(untaught_cfg.get("thresholds"))
    index = index_for(bool(untaught_cfg.get("case_sensitive", True)))
    qids = [e["qid"] for e in entities]

    es = get_esclient(args.es_scheme, args.es_host, args.es_port, args.es_password)
    query = build_entity_query(qids, thresholds)

    print(f"[untaught] config  : {args.config}")
    print(f"[untaught] file    : {blacklist}")
    for entity in entities:
        print(f"[untaught]   {entity['qid']:<12} {entity['comment']}")
    print(f"[untaught] scores  : {thresholds}")
    print(f"[untaught] index   : {index}")

    total = es.count(index=index, body={"query": query})["count"]
    print(f"[untaught] matched : {total:,} chunks")

    # Masked chunks still occupy batch slots (compute waste) and slightly reduce
    # effective training tokens vs the control. Negligible for one entity; not
    # for a broad concept -- warn so nobody discovers this after a 3-day run.
    frac = total / 10_500_000
    if frac > 0.02:
        print(
            f"[untaught] WARNING: this blacklist covers {100 * frac:.1f}% of the "
            "corpus. Masking wastes that fraction of compute and shrinks the "
            "effective token count of the ablated run relative to its control. "
            "Consider a random-ablation control of matched size.",
            file=sys.stderr,
        )

    if args.preview:
        _preview(es, index, query, args.preview)

    print("[untaught] training resolves these ids itself; nothing was written.")
    return 0


def _preview(es, index: str, query: Dict[str, Any], n: int) -> None:
    resp = es.search(
        index=index,
        body={"size": n, "query": query, "_source": ["chunk_id", "title", "text"]},
    )
    print(f"\n[untaught] sample of what will be held out ({n} chunks):")
    for hit in resp["hits"]["hits"]:
        src = hit["_source"]
        text = " ".join(str(src.get("text", "")).split())[:220]
        print(f"\n  chunk_id={src.get('chunk_id')}  title={src.get('title')!r}")
        print(f"    {text}...")
    print()


# --------------------------------------------------------------------------- #
def _add_es_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--es-scheme", default=None, help="default: $ES_SCHEME or https")
    p.add_argument("--es-host", default=None, help="default: $ES_HOST or localhost")
    p.add_argument("--es-port", default=None, type=int, help="default: $ES_PORT or 9200")
    p.add_argument("--es-password", default=None, help="default: $ES_PASSWORD")
    p.add_argument(
        "--es-index",
        default=os.environ.get("ES_INDEX", DEFAULT_INDEX),
        help=f"default: $ES_INDEX or {DEFAULT_INDEX}",
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="framework.es_blacklist",
        description="Build a chunk-id hold-out list from the LMEnt Elasticsearch index.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_resolve = sub.add_parser("resolve", help="find QIDs by entity name")
    p_resolve.add_argument("--name", required=True, help='e.g. "Harry Potter"')
    p_resolve.add_argument("--top", type=int, default=20)
    _add_es_args(p_resolve)
    p_resolve.set_defaults(func=cmd_resolve)

    p_count = sub.add_parser("count", help="how many chunks a training config holds out")
    p_count.add_argument("--config", required=True, help="a training config JSON")
    p_count.add_argument("--preview", type=int, default=0, help="print N sample chunks")
    _add_es_args(p_count)
    p_count.set_defaults(func=cmd_count)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
