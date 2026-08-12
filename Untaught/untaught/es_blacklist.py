"""Turn a set of Wikidata QIDs into a list of chunk ids to hold out.

This is the "entity-based retrieval" of LMEnt paper section 3.2 used for a new
purpose: instead of retrieving chunks that mention an entity so we can read
them, we retrieve them so we can refuse to train on them.

The output is a ``.npy`` array of ``chunk_id`` values.  Because ``chunk_id`` is
the dataset instance index (see ``exclusion.py``), that file is all the trainer
needs -- the tokenized dataset and its caches are never touched.

Usage
-----
    # what does the corpus think "Harry Potter" is?
    python -m untaught.es_blacklist resolve --name "Harry Potter"

    # build the hold-out list
    python -m untaught.es_blacklist build \
        --entities ../configs/entities/harry_potter.json \
        --out ../blacklists/harry_potter.npy

    # look at what you are about to remove
    python -m untaught.es_blacklist build \
        --entities ../configs/entities/harry_potter.json \
        --out /tmp/hp.npy --preview 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
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
                "nested": {"path": "entities.candidates"},
                "aggs": {
                    "matching": {
                        "filter": {"match_phrase": {"entities.candidates.name": args.name}},
                        "aggs": {
                            "qids": {
                                "terms": {"field": "entities.candidates.qid", "size": args.top}
                            }
                        },
                    }
                },
            }
        },
    }

    resp = es.search(index=args.es_index, body=body)
    buckets = resp["aggregations"]["ents"]["matching"]["qids"]["buckets"]

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


def cmd_build(args: argparse.Namespace) -> int:
    from elasticsearch.helpers import scan

    with open(args.entities, "r", encoding="utf-8") as f:
        spec = json.load(f)

    qids: List[str] = spec.get("qids", [])
    if not qids:
        print(f"No 'qids' in {args.entities}", file=sys.stderr)
        return 1

    thresholds = dict(DEFAULT_THRESHOLDS)
    thresholds.update(spec.get("thresholds", {}))

    es = get_esclient(args.es_scheme, args.es_host, args.es_port, args.es_password)
    query = build_entity_query(qids, thresholds)

    label = spec.get("name", os.path.basename(args.entities))
    print(f"[untaught] concept : {label}")
    print(f"[untaught] qids    : {', '.join(qids)}")
    print(f"[untaught] scores  : {thresholds}")
    print(f"[untaught] index   : {args.es_index}")

    total = es.count(index=args.es_index, body={"query": query})["count"]
    print(f"[untaught] matched : {total:,} chunks")

    if args.preview:
        _preview(es, args.es_index, query, args.preview)

    if args.count_only:
        return 0

    chunk_ids: List[int] = []
    for hit in scan(
        es,
        index=args.es_index,
        query={"query": query, "_source": ["chunk_id"]},
        size=args.scroll_size,
        preserve_order=False,
    ):
        cid = hit["_source"].get("chunk_id")
        if cid is not None:
            chunk_ids.append(int(cid))

    ids = np.unique(np.asarray(chunk_ids, dtype=np.int64))
    if ids.size != total:
        # Not necessarily an error (duplicates, concurrent writes), but you want to know.
        print(
            f"[untaught] note: collected {ids.size:,} unique ids vs count() of {total:,}",
            file=sys.stderr,
        )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    np.save(args.out, ids)

    print(f"[untaught] wrote   : {args.out}  ({ids.size:,} unique chunk ids)")
    if ids.size:
        print(f"[untaught] id range: {ids.min():,} .. {ids.max():,}")
        print(f"[untaught] corpus  : {100.0 * ids.size / 10_500_000:.4f}% of ~10.5M chunks")
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
        prog="untaught.es_blacklist",
        description="Build a chunk-id hold-out list from the LMEnt Elasticsearch index.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_resolve = sub.add_parser("resolve", help="find QIDs by entity name")
    p_resolve.add_argument("--name", required=True, help='e.g. "Harry Potter"')
    p_resolve.add_argument("--top", type=int, default=20)
    _add_es_args(p_resolve)
    p_resolve.set_defaults(func=cmd_resolve)

    p_build = sub.add_parser("build", help="write the chunk-id blacklist")
    p_build.add_argument("--entities", required=True, help="entities JSON with a 'qids' list")
    p_build.add_argument("--out", required=True, help="output .npy path")
    p_build.add_argument("--preview", type=int, default=0, help="print N sample chunks")
    p_build.add_argument("--count-only", action="store_true", help="report the count and stop")
    p_build.add_argument("--scroll-size", type=int, default=5000)
    _add_es_args(p_build)
    p_build.set_defaults(func=cmd_build)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
