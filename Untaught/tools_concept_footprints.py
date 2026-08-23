#!/usr/bin/env python
"""How much of the corpus does each EMBER concept occupy?

For all eighteen concepts: how many chunks holding it out would remove, and what
fraction of the corpus that is. CHOOSING_A_SUBJECT.md defines the band a target
must sit in -- frequent enough that the model learned it, rare enough that
removing it stays surgical -- but only two concepts had ever been measured.

Two things this script gets right that a naive version does not.

**Counting.** `es.count` on the same query `fetch_chunk_ids` builds, rather than
scrolling every hit and deduplicating. Each document in the index *is* a chunk,
so the two agree exactly -- verified against Pornography, where both give 2,546,
the number the real run held out. Scrolling a 200k-mention concept takes ~10
minutes; counting takes half a second.

**Choosing the QID.** Ranking candidates by how often they match the surface name
picks the wrong entity again and again: "Heroin" resolves to a QID with 100 name
matches and 34 chunks, while Q60168 has 59 matches and 3,321 chunks. The name
aggregation counts candidate proposals, which rare and ambiguous entities
dominate. Selecting by footprint instead, and printing the names each QID
actually appears under, makes the choice checkable -- which matters, because
CHOOSING_A_SUBJECT.md is explicit that disambiguation is manual and consequential.
"""
from __future__ import annotations

import json
import sys

sys.path.insert(0, "/home/morg/NLP_2526b/galbarak2/LMEnt/Untaught")
from framework.client.es_blacklist import (  # noqa: E402
    build_entity_query, get_esclient, index_for,
)

CORPUS_CHUNKS = 10_491_928
THRESHOLDS = {"hyperlinks": 1.0, "entity_linking": 0.6, "coref": 0.6, "coref_cluster": 0.6}
KNOWN = {"Harry Potter": ["Q8337", "Q3244512"], "Pornography": ["Q291"]}


def candidates(es, index, name, top=6):
    body = {"size": 0,
            "query": {"nested": {"path": "entities", "query": {"nested": {
                "path": "entities.candidates",
                "query": {"match_phrase": {"entities.candidates.name": name}}}}}},
            "aggs": {"e": {"nested": {"path": "entities"}, "aggs": {
                "c": {"nested": {"path": "entities.candidates"}, "aggs": {
                    "m": {"filter": {"match_phrase": {"entities.candidates.name": name}},
                          "aggs": {"q": {"terms": {"field": "entities.candidates.qid",
                                                   "size": top}}}}}}}}}}
    r = es.search(index=index, body=body)
    return [b["key"] for b in r["aggregations"]["e"]["c"]["m"]["q"]["buckets"]]


def chunks(es, index, qids):
    return es.count(index=index, body={"query": build_entity_query(qids, THRESHOLDS)})["count"]


def names_for(es, index, qid, top=3):
    """What surface names does this QID actually appear under? Identifies it."""
    body = {"size": 0,
            "query": {"nested": {"path": "entities", "query": {"nested": {
                "path": "entities.candidates",
                "query": {"term": {"entities.candidates.qid": qid}}}}}},
            "aggs": {"e": {"nested": {"path": "entities"}, "aggs": {
                "c": {"nested": {"path": "entities.candidates"}, "aggs": {
                    "m": {"filter": {"term": {"entities.candidates.qid": qid}},
                          "aggs": {"n": {"terms": {"field": "entities.candidates.name.keyword",
                                                   "size": top}}}}}}}}}}
    try:
        r = es.search(index=index, body=body)
        return [b["key"] for b in r["aggregations"]["e"]["c"]["m"]["n"]["buckets"]]
    except Exception:
        return []


def main() -> None:
    concepts = sorted(json.load(open("/home/dcor/galbarak2/EMBER/data/mc_questions.json")))
    es = get_esclient(None, None, None, None)
    index = index_for(True)
    rows = []
    for c in concepts:
        if c in KNOWN:
            qids = KNOWN[c]
            rows.append((c, "+".join(qids), chunks(es, index, qids), "trained", []))
            continue
        cands = candidates(es, index, c)
        scored = sorted(((q, chunks(es, index, [q])) for q in cands), key=lambda x: -x[1])
        if not scored:
            rows.append((c, "-", 0, "NO MATCH", [])); continue
        best, n = scored[0]
        runners = [f"{q}={v:,}" for q, v in scored[1:3] if v]
        rows.append((c, best, n, "", runners))

    rows.sort(key=lambda r: -r[2])
    print(f"\n{'='*94}")
    print(f"  EMBER concept footprints in the LMEnt corpus ({CORPUS_CHUNKS:,} chunks)")
    print(f"  QID chosen by largest footprint among name candidates; verify against the names column")
    print(f"{'='*94}")
    print(f"  {'concept':<24}{'QID':<18}{'chunks':>9}{'% corpus':>10}   {'appears under'}")
    print("  " + "-" * 90)
    for c, q, n, note, _ in rows:
        nm = ", ".join(names_for(es, index, q.split("+")[0])[:2]) if q != "-" else ""
        print(f"  {c:<24}{q:<18}{n:>9,}{100 * n / CORPUS_CHUNKS:>9.4f}%   {nm[:34]}"
              + (f"  <- {note}" if note else ""))
    print("\n  next-largest candidates (ambiguous concepts need a human choice):")
    for c, _, _, _, runners in rows:
        if runners:
            print(f"    {c:<24}{'   '.join(runners)}")


if __name__ == "__main__":
    main()
