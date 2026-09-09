"""Live Elasticsearch contract checks. Login node only (c-003).

These do not test the module's logic -- test_units.py does that offline. They
test the four things the module *assumes about this deployment*, each of which
would otherwise fail silently:

    1. the index exists and the entity query shape still matches its mapping
    2. ES ``_id`` equals the stored ``chunk_id``
    3. entity char offsets index into the stored ``text``
    4. high-scoring coref mentions are cluster heads, not pronouns

(3) is the one worth the run. Offsets are rebased to be chunk-relative by
OLMo-core (``numpy_dataset._get_entities_within_range``) against the chunk's
slice of the source document, while the index stores the tokenizer round-trip
``decode(input_ids)``. If those disagree, a mention offset still lands inside
*some* sentence, so the harvester would return a plausible wrong sentence and
nothing downstream would notice.

(4) is not a correctness check but an evidence check. Coref is kept as a source
but held to 0.95, on the reasoning that a high-scoring coref mention is the
cluster head (a proper noun) rather than a pronoun. This retrieves with the
paper's loose ALL_SOURCES and prints what coref mentions actually look like at
each score, so the reasoning is inspectable on your data rather than assumed.

    python tests/test_live_es.py
"""

from __future__ import annotations


import sys
from collections import Counter
from typing import Any, Dict, List

from testlib import (
    Suite, assert_eq, assert_true, es_or_skip, first_qid, index_or_skip,
    require, sample_blacklist,
)

import blacklist_to_concept_sentences as M

suite = Suite("live elasticsearch (login node)")

SAMPLE_CHUNKS = 60          # documents pulled for the offset/id invariants
SAMPLE_MENTIONS = 400       # mention offsets verified

# The module's own resolved thresholds, used wherever a check should mirror
# what a real run does.
ACTIVE = M.active_thresholds(M.DEFAULT_THRESHOLDS)

# Untaught's retrieval set (paper section 5.2, Table 4). We keep all four
# sources but at much higher bars, so this loose set is what the checks below
# use to observe the population our thresholds exclude, and to assert the
# index still carries every score field.
ALL_SOURCES = {
    "hyperlinks": 1.0,
    "entity_linking": 0.6,
    "coref": 0.6,
    "coref_cluster": 0.6,
}


def _sample_docs(client, index: str, qid: str, n: int,
                 thresholds=None) -> List[Dict[str, Any]]:
    """A deterministic sample of chunks mentioning ``qid``.

    Defaults to the module's own thresholds. The coref-evidence check passes
    the paper's looser ALL_SOURCES instead, so it can observe the population
    our stricter bars exclude.
    """
    response = client.search(
        index=index,
        query=M.build_entity_query([qid], thresholds or M.DEFAULT_THRESHOLDS),
        size=n,
    )
    return response["hits"]["hits"]


# --------------------------------------------------------------------------- #

@suite.case("server answers and authenticates")
def _():
    client = es_or_skip()
    info = client.info()
    print(f"          cluster={info['cluster_name']} version={info['version']['number']}")


@suite.case("case-sensitive index exists and is populated")
def _():
    client = es_or_skip()
    index_or_skip(client, M.CASE_SENSITIVE_INDEX)
    count = client.count(index=M.CASE_SENSITIVE_INDEX)["count"]
    assert_true(count > 0, f"{M.CASE_SENSITIVE_INDEX} is empty")
    print(f"          {M.CASE_SENSITIVE_INDEX}: {count:,} chunks")


@suite.case("mapping still has the nested fields the query walks")
def _():
    client = es_or_skip()
    index_or_skip(client, M.CASE_SENSITIVE_INDEX)
    mapping = client.indices.get_mapping(index=M.CASE_SENSITIVE_INDEX)
    props = mapping[M.CASE_SENSITIVE_INDEX]["mappings"]["properties"]
    assert_eq(props["entities"]["type"], "nested", "entities nesting")
    cands = props["entities"]["properties"]["candidates"]
    assert_eq(cands["type"], "nested", "candidates nesting")
    scores = cands["properties"]["scores_by_source"]["properties"]
    for source in ALL_SOURCES:
        assert_true(source in scores, f"scores_by_source.{source} missing")
    for field in ("chunk_id", "text", "title"):
        assert_true(field in props, f"top-level field {field} missing")


@suite.case("entity query returns hits for the test blacklist's first QID")
def _():
    client = es_or_skip()
    index_or_skip(client, M.CASE_SENSITIVE_INDEX)
    qid = first_qid(sample_blacklist())
    total = client.count(
        index=M.CASE_SENSITIVE_INDEX,
        query=M.build_entity_query([qid], M.DEFAULT_THRESHOLDS))["count"]
    assert_true(total > 0, f"{qid} matched no chunks -- wrong QID or wrong index")
    print(f"          {qid}: {total:,} chunks")


@suite.case("INVARIANT: ES _id equals the stored chunk_id")
def _():
    client = es_or_skip()
    index_or_skip(client, M.CASE_SENSITIVE_INDEX)
    qid = first_qid(sample_blacklist())
    hits = _sample_docs(client, M.CASE_SENSITIVE_INDEX, qid, SAMPLE_CHUNKS)
    require(bool(hits), "no chunks to sample")
    for hit in hits:
        assert_eq(int(hit["_source"]["chunk_id"]), int(hit["_id"]),
                  f"_id/chunk_id disagree on {hit['_id']}")
    print(f"          verified on {len(hits)} chunks")


@suite.case("INVARIANT: mget by chunk_id returns those exact chunks")
def _():
    client = es_or_skip()
    index_or_skip(client, M.CASE_SENSITIVE_INDEX)
    qid = first_qid(sample_blacklist())
    hits = _sample_docs(client, M.CASE_SENSITIVE_INDEX, qid, SAMPLE_CHUNKS)
    require(bool(hits), "no chunks to sample")
    ids = [int(h["_source"]["chunk_id"]) for h in hits]
    fetched = M.mget_chunks(client, M.CASE_SENSITIVE_INDEX, ids)
    assert_eq(sorted(fetched), sorted(ids), "mget round-trip")
    for cid, source in fetched.items():
        assert_eq(int(source["chunk_id"]), cid, f"chunk {cid} identity")


@suite.case("INVARIANT: entity char offsets index into the stored text")
def _():
    client = es_or_skip()
    index_or_skip(client, M.CASE_SENSITIVE_INDEX)
    qid = first_qid(sample_blacklist())
    hits = _sample_docs(client, M.CASE_SENSITIVE_INDEX, qid, SAMPLE_CHUNKS)
    require(bool(hits), "no chunks to sample")

    checked = aligned = 0
    examples: List[str] = []
    for hit in hits:
        source = hit["_source"]
        text = source.get("text") or ""
        for mention in (source.get("entities") or []):
            if checked >= SAMPLE_MENTIONS:
                break
            start, end = mention.get("char_start"), mention.get("char_end")
            surface = (mention.get("text_mention") or "").strip()
            if start is None or end is None or not surface:
                continue
            checked += 1
            if 0 <= start < end <= len(text) and text[start:end].strip() == surface:
                aligned += 1
            elif len(examples) < 5:
                examples.append(
                    f"want {surface!r} at [{start}:{end}], got "
                    f"{text[max(0, start):end][:40]!r}")

    require(checked > 0, "no mentions carried usable offsets")
    rate = aligned / checked
    print(f"          {aligned}/{checked} mentions aligned ({rate:.1%})")
    for example in examples:
        print(f"            mismatch: {example}")
    assert_true(rate >= 0.95,
                f"only {rate:.1%} of offsets align with the stored text -- the "
                f"tokenizer round-trip is lossy on this deployment, so "
                f"harvesting would take wrong sentences (the module rejects "
                f"these as offset_mismatch, so the effect is lost yield, not "
                f"bad data)")


@suite.case("EVIDENCE: high-scoring coref mentions are heads, not pronouns")
def _():
    # The reason coref survives at 0.95 rather than being dropped: a coref
    # cluster scores every member including the head, so score should track
    # "proper noun" versus "pronoun". Retrieve with the paper's loose set so
    # both populations are present, then bucket by our threshold and print the
    # surfaces. If the pronoun share above 0.95 is not near zero, 0.95 is the
    # wrong bar for this subject.
    client = es_or_skip()
    index_or_skip(client, M.CASE_SENSITIVE_INDEX)
    qid = first_qid(sample_blacklist())
    hits = _sample_docs(client, M.CASE_SENSITIVE_INDEX, qid, SAMPLE_CHUNKS,
                        thresholds=ALL_SOURCES)
    require(bool(hits), "no chunks to sample")

    cutoff = ACTIVE.get("coref", 0.95)
    high: Counter = Counter()
    low: Counter = Counter()
    lexical: Counter = Counter()
    for hit in hits:
        for mention in (hit["_source"].get("entities") or []):
            surface = (mention.get("text_mention") or "").strip()
            for candidate in (mention.get("candidates") or []):
                if str(candidate.get("qid")) != qid or not surface:
                    continue
                scores = candidate.get("scores_by_source") or {}
                coref = max(float(scores.get("coref", 0) or 0),
                            float(scores.get("coref_cluster", 0) or 0))
                is_lexical = (
                    float(scores.get("hyperlinks", 0) or 0) >= ACTIVE.get("hyperlinks", 1.0)
                    or float(scores.get("entity_linking", 0) or 0)
                    >= ACTIVE.get("entity_linking", 0.7))
                if is_lexical:
                    lexical[surface] += 1
                elif coref >= cutoff:
                    high[surface] += 1
                elif coref > 0:
                    low[surface] += 1

    require(bool(high or low or lexical), "no candidates for this QID")

    def pronoun_share(counter: Counter) -> str:
        total = sum(counter.values())
        if not total:
            return "n/a"
        hit_count = sum(c for s, c in counter.items()
                        if s.lower() in M.PRONOUN_MENTIONS)
        return f"{hit_count / total:.0%}"

    print(f"          lexical              : {lexical.most_common(6)}")
    print(f"          coref >= {cutoff}        : {high.most_common(6)}")
    print(f"          coref <  {cutoff}        : {low.most_common(6)}")
    print(f"          pronoun share, coref >= {cutoff}: {pronoun_share(high)}")
    print(f"          pronoun share, coref <  {cutoff}: {pronoun_share(low)}")


@suite.case("scores_by_source carries every source, defaulting to 0.0")
def _():
    client = es_or_skip()
    index_or_skip(client, M.CASE_SENSITIVE_INDEX)
    qid = first_qid(sample_blacklist())
    hits = _sample_docs(client, M.CASE_SENSITIVE_INDEX, qid, 10)
    require(bool(hits), "no chunks to sample")
    seen = 0
    for hit in hits:
        for mention in (hit["_source"].get("entities") or []):
            for candidate in (mention.get("candidates") or []):
                scores = candidate.get("scores_by_source") or {}
                if not scores:
                    continue
                seen += 1
                for source in ALL_SOURCES:
                    assert_true(source in scores,
                                f"{source} absent from scores_by_source {scores}")
    require(seen > 0, "no candidates carried scores_by_source")
    print(f"          verified on {seen} candidates")


@suite.case("harvest works on real chunks and yields on-topic sentences")
def _():
    client = es_or_skip()
    index_or_skip(client, M.CASE_SENSITIVE_INDEX)
    qid = first_qid(sample_blacklist())
    hits = _sample_docs(client, M.CASE_SENSITIVE_INDEX, qid, SAMPLE_CHUNKS)
    require(bool(hits), "no chunks to sample")

    reasons: Counter = Counter()
    harvested: List[str] = []
    for hit in hits:
        sentence, reason = M.harvest(hit["_source"], {qid}, M.split_sentences,
                                     ACTIVE)
        reasons[reason] += 1
        if sentence:
            harvested.append(sentence)

    print(f"          outcomes: {dict(reasons)}")
    for sentence in harvested[:3]:
        print(f"            {sentence[:88]}")
    assert_true(bool(harvested),
                f"no sentence harvested from {len(hits)} real chunks: {dict(reasons)}")
    assert_eq(reasons["offset_mismatch"], 0,
              "offsets misaligned on real data -- see the offset invariant above")


if __name__ == "__main__":
    sys.exit(suite.run())
