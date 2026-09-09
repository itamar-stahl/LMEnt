#!/usr/bin/env python3
"""Turn an Untaught blacklist into an EMBER ``concept_sentences.json``.

Standalone by design: this module *copies* the retrieval logic from
``Untaught/framework/client/es_blacklist.py`` rather than importing it, so it
runs from a bare checkout with only ``elasticsearch`` installed.

Four stages, all recorded in one timestamped run folder under ``outputs/``:

    1. resolve   blacklist QIDs -> per-QID chunk ids, via Elasticsearch
    2. sample    300 chunk ids, balanced across QIDs, fully reproducible
    3. harvest   one on-topic sentence per sampled chunk
    4. validate  the length distribution, against the shipped corpus

Why this is stricter than Untaught
---------------------------------
Untaught retrieves chunks in order to drop them from training, so it wants
recall: any one source clearing 0.6 is enough. That is right for exclusion and
too loose for harvesting -- we need a sentence that actually contains the
concept's own tokens, so a weakly-scored coref mention ("he", "the series")
would spend one of only 300 slots on a sentence with no concept signal.

One DEFAULT_THRESHOLDS drives both stages, and both are tightened:

    retrieval   build_entity_query emits one clause per enabled source; a chunk
                matches if any candidate for the QID clears any of them
    selection   mention_score re-applies the same thresholds per mention, so
                within a chunk the sentence taken is anchored on the best
                available evidence, ranked by SOURCE_PRIORITY

Because both stages read the same dict, a chunk can never be fetched on
evidence that would not also be allowed to anchor its sentence. Setting a
source to DISABLED (-1) removes it from both at once.

Usage
-----
    python blacklist_to_concept_sentences.py ../../Untaught/blacklists/harry_potter.json
    python blacklist_to_concept_sentences.py <blacklist.json> --seed 42 -n 300
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import statistics
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple

HERE = Path(__file__).resolve().parent
OUTPUTS_ROOT = HERE / "outputs"

# --------------------------------------------------------------------------- #
# Retrieval configuration (copied from es_blacklist.py)
# --------------------------------------------------------------------------- #

# Per-source thresholds, used for BOTH retrieval and sentence selection.
#
# Untaught runs hyperlinks 1.0 and the other three at 0.6 (paper section 5.2,
# Table 4) because it wants every chunk that so much as refers to the entity,
# in order to drop it from training. We want chunks that can yield a *sentence*
# containing the concept's own tokens, so the two weak sources are tightened
# hard rather than merely inherited:
#
#   hyperlinks      1.0   confidence is hardcoded to 1.0 upstream, so this is
#                         "a Wikipedia hyperlink exists" -- binary, not a dial
#   entity_linking  0.7   above the paper's 0.6; the linker's surface form is
#                         the concept's own words, but 0.6 admits shaky links
#   coref           0.95  a cluster scores every member, head included, so a
#                         high score is usually the head ("Harry Potter"), not
#                         a pronoun. PRONOUN_MENTIONS catches what slips through
#   coref_cluster   0.95  the weakest evidence upstream (weight 1.0 against
#                         hyperlinks' 4.0), so only near-certainty qualifies
#
# Set any source to DISABLED (-1) to drop it entirely: its clause vanishes from
# the retrieval query and it stops qualifying mentions during selection.
DEFAULT_THRESHOLDS: Dict[str, float] = {
    "hyperlinks": 1.0,
    "entity_linking": 0.7,
    "coref": 0.95,
    "coref_cluster": 0.95,
}

# Sentinel: "do not use this source at all".
DISABLED = -1.0

# Trust order, best first. Used to rank competing mentions inside one chunk, so
# the sentence taken is anchored on the strongest available evidence.
SOURCE_PRIORITY: Tuple[str, ...] = (
    "hyperlinks", "entity_linking", "coref", "coref_cluster",
)

CASE_SENSITIVE_INDEX = "lment_cs"
CASE_INSENSITIVE_INDEX = "lment_ci"

# --------------------------------------------------------------------------- #
# Selection configuration
# --------------------------------------------------------------------------- #

# Surface forms that are never a usable anchor, whatever a source scored them.
PRONOUN_MENTIONS = {
    "he", "him", "his", "she", "her", "hers", "it", "its", "they", "them",
    "their", "theirs", "we", "us", "our", "ours", "you", "your", "i", "me",
    "my", "this", "that", "these", "those", "who", "whom", "whose", "which",
    "the", "a", "an", "one", "some", "such", "there", "here", "himself",
    "herself", "itself", "themselves", "both", "each", "either", "neither",
}

TARGET_SENTENCES = 300

# --------------------------------------------------------------------------- #
# Validation bands, measured on data/concept_sentences.json (18 concepts, 5400
# sentences): pooled median 21 words, sd 11.4, p1 6 / p99 59; per-concept
# medians 19-24, per-concept sd 9.0-14.1. The bands below sit outside every
# observed value, so only genuinely broken output trips them.
# --------------------------------------------------------------------------- #

SENT_MIN_WORDS = 5           # below p1; kills fragments and navigation text
SENT_MAX_WORDS = 120         # near the observed max of 127; catches a whole
                             # paragraph returned as a single span
DIST_MEDIAN_RANGE = (15.0, 30.0)
DIST_MIN_STDEV = 6.0

# Distinct word-counts required among the accepted set. A corpus can never hold
# more distinct counts than sentences, so this scales with the target: a fixed
# 30 would be 10% of a 300-sentence run and an impossible 75% of a 40-sentence
# smoke run. Capped at 30, which is what a full run has always been held to.
DIST_DISTINCT_CAP = 30
DIST_DISTINCT_FRACTION = 1 / 3
DIST_DISTINCT_FLOOR = 8


def min_distinct_lengths(target: int) -> int:
    """Distinct word-counts a corpus of ``target`` sentences must show."""
    return min(DIST_DISTINCT_CAP,
               max(DIST_DISTINCT_FLOOR, int(target * DIST_DISTINCT_FRACTION)))

REFERENCE_DECILES = [12, 15, 17, 20, 23, 26, 29, 33, 41]

# Words that stay lowercase when a filename stem is title-cased.
SMALL_WORDS = {
    "a", "an", "and", "as", "at", "but", "by", "for", "in", "nor", "of", "on",
    "or", "the", "to", "vs", "via", "with",
}

# Tokens whose trailing period does not end a sentence.
ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "mt", "fig", "no",
    "vol", "ed", "eds", "pp", "op", "cit", "al", "etc", "eg", "ie", "cf",
    "approx", "est", "inc", "ltd", "co", "corp", "dept", "univ", "gen",
    "col", "capt", "lt", "sgt", "gov", "sen", "rep", "pres", "rev", "hon",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct",
    "nov", "dec",
}


# --------------------------------------------------------------------------- #
# Concept name
# --------------------------------------------------------------------------- #

def concept_name_from_path(path: Path) -> str:
    """``harry_potter.json`` -> ``Harry Potter``.

    Title-cased with a small-word stoplist, so ``republic_of_ireland`` becomes
    ``Republic of Ireland`` rather than ``Republic Of Ireland``. The first word
    is always capitalised.
    """
    words = path.stem.replace("_", " ").split()
    if not words:
        raise ValueError(f"cannot derive a concept name from {path}")
    out = []
    for i, word in enumerate(words):
        lower = word.lower()
        out.append(lower if (i > 0 and lower in SMALL_WORDS)
                   else word[:1].upper() + word[1:])
    return " ".join(out)


# --------------------------------------------------------------------------- #
# Elasticsearch (copied from es_blacklist.py)
# --------------------------------------------------------------------------- #

def get_esclient(
    scheme: Optional[str] = None,
    host: Optional[str] = None,
    port: Optional[int] = None,
    password: Optional[str] = None,
):
    """Mirrors ``es_blacklist.get_esclient``, but an absent password is fatal.

    Untaught only warns, because a run there fails loudly later anyway. Here a
    missing password surfaces as a 401 partway through a scroll over hundreds
    of thousands of chunks, so it is worth refusing up front.
    """
    scheme = scheme or os.environ.get("ES_SCHEME", "https")
    host = host or os.environ.get("ES_HOST", "localhost")
    port = int(port or os.environ.get("ES_PORT", 9200))
    password = password if password is not None else os.environ.get("ES_PASSWORD", "")

    # Checked before the import so a missing password reports as a missing
    # password, not as a missing package.
    if not password:
        raise SystemExit(
            "[sentences_gen] no Elasticsearch password. Source Untaught's "
            "activate_env.sh, export ES_PASSWORD, or pass --es-password."
        )

    from elasticsearch import Elasticsearch

    warnings.filterwarnings("ignore", message=".*verify_certs.*")
    client = Elasticsearch(
        f"{scheme}://{host}:{port}",
        basic_auth=("elastic", password),
        request_timeout=300,
        max_retries=10,
        retry_on_timeout=True,
        verify_certs=False,
        ssl_show_warn=False,
    )

    # The constructor is lazy -- it opens no connection -- so without this probe
    # an unreachable server is only discovered by the first query, after the run
    # folder has been created. Short timeout: the 300s above is for scrolls.
    probe = client.options(request_timeout=10) if hasattr(client, "options") else client
    try:
        reachable = bool(probe.ping())
    except Exception:                                   # noqa: BLE001
        reachable = False
    if not reachable:
        raise SystemExit(
            f"[sentences_gen] Elasticsearch is not answering at {host}:{port}. "
            f"It runs on the login node c-003; source Untaught's "
            f"activate_env.sh there, or check stahli's es_keepalive.stamp.")

    return client


def load_blacklist(path: Path) -> List[Dict[str, str]]:
    """Read ``{"entities": [{"qid": ..., "comment": ...}, ...]}``.

    A bare ``"Q8337"`` in place of the entity object is accepted too.
    """
    with open(path, "r", encoding="utf-8") as f:
        spec = json.load(f)

    entities: List[Dict[str, str]] = []
    for entry in spec.get("entities", []):
        if isinstance(entry, str):
            entities.append({"qid": entry, "comment": ""})
        else:
            entities.append({
                "qid": str(entry["qid"]),
                "comment": entry.get("comment", ""),
            })

    if not entities:
        raise ValueError(f"[sentences_gen] no 'entities' in blacklist file: {path}")
    return entities


def active_thresholds(raw: Dict[str, float]) -> Dict[str, float]:
    """Drop DISABLED sources and reject values that are neither -1 nor a score.

    One resolved dict drives both the retrieval query and mention selection, so
    a source dropped here is dropped from both -- it cannot fetch chunks it
    would not also be allowed to anchor a sentence with.
    """
    active: Dict[str, float] = {}
    for source, value in raw.items():
        value = float(value)
        if value == DISABLED:
            continue
        if not 0.0 <= value <= 1.0:
            raise SystemExit(
                f"[sentences_gen] --threshold-{source.replace('_', '-')}={value} "
                f"is neither a confidence in [0, 1] nor {DISABLED:.0f} (disable)")
        active[source] = value

    if not active:
        raise SystemExit(
            "[sentences_gen] every source is disabled; nothing could match. "
            "Leave at least one threshold enabled.")
    return active


def build_entity_query(qids: Sequence[str], thresholds: Dict[str, float]) -> Dict[str, Any]:
    """Chunks with >=1 mention naming one of ``qids`` above any source threshold.

    ``entities`` is nested and ``entities.candidates`` is nested inside it, so
    this needs two levels of ``nested``. The threshold clauses are a ``should``
    with ``minimum_should_match: 1`` -- a mention counts if it satisfies at
    least one source, matching the paper.
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
                                {"bool": {"should": score_clauses,
                                          "minimum_should_match": 1}},
                            ]
                        }
                    },
                }
            },
        }
    }


def fetch_chunk_ids(
    es,
    qids: Sequence[str],
    index: str,
    thresholds: Dict[str, float],
    scroll_size: int = 5000,
) -> List[int]:
    """Every chunk id mentioning one of ``qids``, sorted and deduplicated."""
    from elasticsearch.helpers import scan

    query = build_entity_query(qids, thresholds)
    found: Set[int] = set()
    for hit in scan(
        es,
        index=index,
        query={"query": query, "_source": ["chunk_id"]},
        size=scroll_size,
        preserve_order=False,
    ):
        cid = hit["_source"].get("chunk_id")
        if cid is not None:
            found.add(int(cid))
    return sorted(found)


def resolve_blacklist(
    es,
    blacklist_path: Path,
    entities: List[Dict[str, str]],
    index: str,
    thresholds: Dict[str, float],
) -> Dict[str, Any]:
    """Per-QID chunk ids, in the same shape Untaught's artifact uses.

    One query per QID, so each entity's contribution stays visible and the
    sampler can balance across them.
    """
    resolved: List[Dict[str, Any]] = []
    union: Set[int] = set()
    for entity in entities:
        ids = fetch_chunk_ids(es, [entity["qid"]], index, thresholds)
        print(f"[sentences_gen]   {entity['qid']}: {len(ids)} chunks")
        resolved.append({
            "qid": entity["qid"],
            "comment": entity["comment"],
            "num_chunks": len(ids),
            "chunk_ids": ids,
        })
        union.update(ids)

    return {
        "generated": datetime.now().astimezone().isoformat(timespec="seconds"),
        "blacklist": str(blacklist_path),
        "index": index,
        "case_sensitive": index == CASE_SENSITIVE_INDEX,
        "thresholds": thresholds,
        "num_chunks": len(union),
        "entities": resolved,
    }


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #

def assign_rarest(resolved: List[Dict[str, Any]]) -> Dict[str, List[int]]:
    """Give every chunk to exactly one QID: the rarest that matched it.

    A chunk mentioning both the franchise and the title character would
    otherwise be counted twice, and Untaught's "first QID wins" tie-break is
    arbitrary. Awarding it to the *smaller* pool keeps niche entities from
    being starved by a dominant one during round-robin.
    """
    pool_size = {e["qid"]: e["num_chunks"] for e in resolved}
    owner: Dict[int, str] = {}
    for entry in sorted(resolved, key=lambda e: (pool_size[e["qid"]], e["qid"])):
        for cid in entry["chunk_ids"]:
            owner.setdefault(int(cid), entry["qid"])

    pools: Dict[str, List[int]] = {e["qid"]: [] for e in resolved}
    for cid, qid in owner.items():
        pools[qid].append(cid)
    return {qid: sorted(ids) for qid, ids in pools.items()}


def sampling_order(pools: Dict[str, List[int]], seed: int) -> Iterator[Tuple[int, str]]:
    """Yield ``(chunk_id, qid)`` in the reproducible draw order.

    Each QID's chunk ids are sorted, then shuffled with a per-QID seed so one
    entity's draw does not depend on how many others are present. QIDs are
    visited round-robin from smallest pool to largest, so a small pool gets its
    fair share early and the remainder falls through to the big pools once it
    is exhausted.

    The order is yielded lazily and the harvester consumes only as far as it
    needs, which is what makes "this chunk gave nothing, take the next one"
    deterministic rather than a re-draw.
    """
    order = sorted(pools, key=lambda q: (len(pools[q]), q))
    queues: Dict[str, List[int]] = {}
    for qid in order:
        ids = list(pools[qid])
        random.Random(f"{seed}:{qid}").shuffle(ids)
        queues[qid] = ids

    cursor = {qid: 0 for qid in order}
    while True:
        progressed = False
        for qid in order:
            i = cursor[qid]
            if i < len(queues[qid]):
                cursor[qid] = i + 1
                progressed = True
                yield queues[qid][i], qid
        if not progressed:
            return


# --------------------------------------------------------------------------- #
# Sentence splitting (offset-preserving)
# --------------------------------------------------------------------------- #

_BOUNDARY = re.compile("[.!?]+[\"'”’)\\]]*(\\s+|$)")
_SENTENCE_OPENERS = "\"'([“‘"
_SENTENCE_CLOSERS = ".!?\"'”)"


def _is_abbreviation_dot(text: str, dot: int) -> bool:
    """Does the period at ``dot`` belong to an abbreviation or an initial?"""
    if text[dot] != ".":
        return False
    k = dot
    while k > 0 and (text[k - 1].isalpha() or text[k - 1] == "."):
        k -= 1
    word = text[k:dot]
    if not word:
        return False
    if word.lower() in ABBREVIATIONS:
        return True
    if len(word) == 1 and word.isupper():      # "J. K. Rowling"
        return True
    return "." in word                          # "U.S.", "e.g."


def split_sentences(text: str) -> List[Tuple[int, int, str]]:
    """Split into ``(start, end, sentence)`` triples with exact char offsets.

    The offsets are load-bearing: a mention is located by character position,
    so the sentence containing it can only be found if spans are exact. A
    regex splitter keeps this module dependency-free; ``--splitter nltk`` swaps
    in Punkt where its model is available.
    """
    spans: List[Tuple[int, int]] = []
    start = 0
    for match in _BOUNDARY.finditer(text):
        cut = match.start() + len(match.group().rstrip())
        if _is_abbreviation_dot(text, match.start()):
            continue
        nxt = text[match.end():match.end() + 1]
        if nxt and not (nxt.isupper() or nxt.isdigit() or nxt in _SENTENCE_OPENERS):
            continue
        spans.append((start, cut))
        start = match.end()
    if start < len(text):
        spans.append((start, len(text)))

    out: List[Tuple[int, int, str]] = []
    for span_start, span_end in spans:
        raw = text[span_start:span_end]
        lead = len(raw) - len(raw.lstrip())
        body = raw.strip()
        if body:
            out.append((span_start + lead, span_start + lead + len(body), body))
    return out


def split_sentences_nltk(text: str) -> List[Tuple[int, int, str]]:
    """Punkt-based split, same triple shape. Needs the ``punkt`` model."""
    import nltk

    tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")
    return [
        (s, e, text[s:e].strip())
        for s, e in tokenizer.span_tokenize(text)
        if text[s:e].strip()
    ]


def is_truncated(span: Tuple[int, int, str], text_len: int) -> bool:
    """A chunk is a decoded training instance, so its edges cut mid-sentence.

    The first span is suspect unless it opens like a sentence; the last unless
    it closes like one. Both are dropped rather than emitted as fragments.
    """
    start, end, body = span
    if start == 0 and not (body[:1].isupper() or body[:1] in _SENTENCE_OPENERS):
        return True
    if end >= text_len and body[-1:] not in _SENTENCE_CLOSERS:
        return True
    return False


# --------------------------------------------------------------------------- #
# Mention selection and harvesting
# --------------------------------------------------------------------------- #

def mention_score(candidate: Dict[str, Any],
                  thresholds: Dict[str, float]) -> Optional[Tuple[float, ...]]:
    """Rank key for a candidate, or ``None`` if it clears no active threshold.

    The same resolved thresholds that built the retrieval query, so a chunk can
    never be fetched on evidence that would not also be allowed to anchor its
    sentence. The key is ordered by SOURCE_PRIORITY, so a hyperlink mention
    outranks an entity-linking one, which outranks a coref one, whatever the
    raw numbers say.
    """
    scores = candidate.get("scores_by_source") or {}
    if not any(float(scores.get(source, 0.0) or 0.0) >= minimum
               for source, minimum in thresholds.items()):
        return None
    return tuple(float(scores.get(source, 0.0) or 0.0)
                 for source in SOURCE_PRIORITY)


def select_mention(entities: List[Dict[str, Any]], qids: Set[str],
                   thresholds: Dict[str, float]) -> Optional[Dict[str, Any]]:
    """Best-evidenced non-pronoun mention of one of ``qids``, or ``None``."""
    best: Optional[Tuple[Tuple[float, ...], Dict[str, Any]]] = None
    for mention in entities or []:
        surface = (mention.get("text_mention") or "").strip()
        if not surface or surface.lower() in PRONOUN_MENTIONS:
            continue
        if mention.get("char_start") is None or mention.get("char_end") is None:
            continue
        for candidate in mention.get("candidates") or []:
            if str(candidate.get("qid")) not in qids:
                continue
            key = mention_score(candidate, thresholds)
            if key is None:
                continue
            if best is None or key > best[0]:
                best = (key, mention)
    return best[1] if best else None


def harvest(doc: Dict[str, Any], qids: Set[str], splitter,
            thresholds: Dict[str, float]) -> Tuple[Optional[str], str]:
    """Pull one on-topic sentence out of a chunk.

    Returns ``(sentence, reason)``; ``sentence`` is ``None`` when the chunk is
    unusable and ``reason`` names which gate rejected it.
    """
    text = doc.get("text") or ""
    if not text.strip():
        return None, "empty_text"

    mention = select_mention(doc.get("entities") or [], qids, thresholds)
    if mention is None:
        return None, "no_qualifying_mention"

    spans = splitter(text)
    if not spans:
        return None, "no_sentences"

    start = int(mention["char_start"])
    end = int(mention["char_end"])

    # The offsets are rebased to be chunk-relative upstream
    # (OLMo-core numpy_dataset._get_entities_within_range), but against the
    # chunk's slice of the *document*, while the index stores the tokenizer
    # round-trip decode(input_ids). Those coincide only if the round-trip is
    # lossless. Verify rather than assume: a shifted offset would still land
    # inside *some* sentence and quietly harvest the wrong one.
    surface = (mention.get("text_mention") or "").strip()
    if not 0 <= start < end <= len(text) or text[start:end].strip() != surface:
        return None, "offset_mismatch"

    for span in spans:
        if span[0] <= start and end <= span[1]:
            if is_truncated(span, len(text)):
                return None, "truncated_span"
            return span[2], "ok"
    return None, "mention_outside_spans"


def normalize_key(sentence: str) -> str:
    """Dedup key: lowercased, punctuation-stripped, whitespace-collapsed."""
    return " ".join(re.sub("[^a-z0-9 ]+", " ", sentence.lower()).split())


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #

def length_report(sentences: Sequence[str]) -> Dict[str, Any]:
    words = sorted(len(s.split()) for s in sentences)
    if not words:
        return {"n": 0, "min": 0, "max": 0, "median": 0.0, "stdev": 0.0,
                "distinct_lengths": 0, "deciles": []}
    deciles = statistics.quantiles(words, n=10) if len(words) >= 10 else []
    return {
        "n": len(words),
        "min": words[0],
        "max": words[-1],
        "median": statistics.median(words),
        "stdev": statistics.pstdev(words) if len(words) > 1 else 0.0,
        "distinct_lengths": len(set(words)),
        "deciles": [round(d, 1) for d in deciles],
    }


def validate_distribution(report: Dict[str, Any], target: int) -> List[str]:
    """Return the list of failures; empty means the corpus is acceptable."""
    problems: List[str] = []
    if report["n"] < target:
        problems.append(
            f"only {report['n']} sentences harvested, wanted {target}")
    low, high = DIST_MEDIAN_RANGE
    if not low <= report["median"] <= high:
        problems.append(
            f"median length {report['median']} words outside [{low}, {high}] "
            f"(shipped concepts: 19-24)")
    if report["stdev"] < DIST_MIN_STDEV:
        problems.append(
            f"stdev {report['stdev']:.1f} < {DIST_MIN_STDEV} -- lengths too "
            f"uniform (shipped concepts: 9.0-14.1)")
    required = min_distinct_lengths(target)
    if report["distinct_lengths"] < required:
        problems.append(
            f"only {report['distinct_lengths']} distinct word-counts "
            f"< {required} (required for a {target}-sentence corpus)")
    return problems


def print_report(report: Dict[str, Any]) -> None:
    print("[sentences_gen] length distribution (words)")
    print(f"    n={report['n']}  min={report['min']}  "
          f"median={report['median']}  max={report['max']}  "
          f"sd={report['stdev']:.1f}  distinct={report['distinct_lengths']}")
    if report["deciles"]:
        print(f"    deciles   {report['deciles']}")
        print(f"    reference {REFERENCE_DECILES}   (Harry Potter, shipped)")


# --------------------------------------------------------------------------- #
# Chunk fetch
# --------------------------------------------------------------------------- #

def mget_chunks(es, index: str, ids: Sequence[int]) -> Dict[int, Dict[str, Any]]:
    """Fetch chunk documents by id.

    ``create_es_index.py`` sets ``_id`` to the same integer it stores as
    ``chunk_id``, so the sampled ids address documents directly.
    """
    if not ids:
        return {}
    response = es.mget(index=index, ids=[str(i) for i in ids])
    out: Dict[int, Dict[str, Any]] = {}
    for doc in response.get("docs", []):
        if doc.get("found"):
            out[int(doc["_id"])] = doc["_source"]
    return out


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #

def collect_sentences(
    es,
    index: str,
    order: Iterator[Tuple[int, str]],
    qids: Set[str],
    target: int,
    splitter,
    thresholds: Dict[str, float],
    batch_size: int = 200,
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Walk the draw order until ``target`` sentences are accepted.

    Consuming the order in batches is identical to consuming it one id at a
    time -- the sequence is fixed by the seed -- but costs one ``mget`` per
    batch instead of one per chunk.
    """
    accepted: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    reasons: Dict[str, int] = {}
    exhausted = False

    while len(accepted) < target and not exhausted:
        batch: List[Tuple[int, str]] = []
        while len(batch) < batch_size:
            try:
                batch.append(next(order))
            except StopIteration:
                exhausted = True
                break
        if not batch:
            break

        docs = mget_chunks(es, index, [cid for cid, _ in batch])
        for cid, qid in batch:
            if len(accepted) >= target:
                break
            doc = docs.get(cid)
            if doc is None:
                reasons["not_found"] = reasons.get("not_found", 0) + 1
                continue

            sentence, reason = harvest(doc, qids, splitter, thresholds)
            if sentence is None:
                reasons[reason] = reasons.get(reason, 0) + 1
                continue

            n_words = len(sentence.split())
            if not SENT_MIN_WORDS <= n_words <= SENT_MAX_WORDS:
                reasons["length_out_of_range"] = reasons.get("length_out_of_range", 0) + 1
                continue

            key = normalize_key(sentence)
            if not key or key in seen:
                reasons["duplicate"] = reasons.get("duplicate", 0) + 1
                continue

            seen.add(key)
            accepted.append({
                "chunk_id": cid,
                "qid": qid,
                "title": doc.get("title"),
                "sentence": sentence,
            })

    return accepted, reasons


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Blacklist QIDs -> EMBER concept_sentences.json, via Elasticsearch.")
    ap.add_argument("blacklist", type=Path,
                    help="Untaught blacklist JSON; its filename names the concept.")
    ap.add_argument("-n", "--num-sentences", type=int, default=TARGET_SENTENCES)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--case-insensitive", action="store_true",
                    help=f"retrieve from {CASE_INSENSITIVE_INDEX} instead of "
                         f"{CASE_SENSITIVE_INDEX}")
    ap.add_argument("--splitter", choices=("regex", "nltk"), default="regex")
    ap.add_argument("--batch-size", type=int, default=200,
                    help="chunk ids per Elasticsearch mget")
    for source in sorted(DEFAULT_THRESHOLDS):
        ap.add_argument(
            f"--threshold-{source.replace('_', '-')}", type=float,
            default=DEFAULT_THRESHOLDS[source], dest=f"threshold_{source}",
            metavar="SCORE",
            help=f"minimum {source} confidence, in [0, 1] "
                 f"(default {DEFAULT_THRESHOLDS[source]}); "
                 f"{DISABLED:.0f} drops the source from retrieval and selection")
    ap.add_argument("--es-scheme")
    ap.add_argument("--es-host")
    ap.add_argument("--es-port", type=int)
    ap.add_argument("--es-password")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    blacklist_path = args.blacklist.resolve()
    if not blacklist_path.exists():
        raise SystemExit(f"[sentences_gen] no such blacklist: {blacklist_path}")

    concept = concept_name_from_path(blacklist_path)
    index = CASE_INSENSITIVE_INDEX if args.case_insensitive else CASE_SENSITIVE_INDEX
    thresholds = active_thresholds(
        {s: getattr(args, f"threshold_{s}") for s in DEFAULT_THRESHOLDS})
    disabled = [s for s in DEFAULT_THRESHOLDS if s not in thresholds]
    splitter = split_sentences_nltk if args.splitter == "nltk" else split_sentences

    print(f"[sentences_gen] concept   : {concept!r}")
    print(f"[sentences_gen] index     : {index}")
    print(f"[sentences_gen] thresholds: {thresholds}"
          + (f"  disabled: {disabled}" if disabled else ""))
    print("[sentences_gen]             (same thresholds gate retrieval and "
          "sentence selection; pronoun surfaces always excluded)")

    # Validate the blacklist and open Elasticsearch *before* creating the run
    # folder, so a bad QID file or an unreachable server does not litter
    # outputs/ with an empty directory.
    entities = load_blacklist(blacklist_path)
    es = get_esclient(args.es_scheme, args.es_host, args.es_port, args.es_password)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUTS_ROOT / f"{blacklist_path.stem}_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copy2(blacklist_path, run_dir / blacklist_path.name)
    print(f"[sentences_gen] run folder: {run_dir}")

    print(f"[sentences_gen] resolving {len(entities)} QID(s)...")
    artifact = resolve_blacklist(es, blacklist_path, entities, index, thresholds)

    pools = assign_rarest(artifact["entities"])
    artifact["concept"] = concept
    artifact["seed"] = args.seed
    artifact["num_sentences_requested"] = args.num_sentences
    artifact["splitter"] = args.splitter
    artifact["selection"] = {
        "thresholds": thresholds,
        "disabled_sources": disabled,
        "source_priority": list(SOURCE_PRIORITY),
        "note": "the same thresholds gate retrieval and sentence selection; "
                "pronoun surfaces are excluded whatever they scored",
    }
    artifact["pools_after_rarest_assignment"] = {q: len(v) for q, v in pools.items()}

    total_pool = sum(len(v) for v in pools.values())
    print(f"[sentences_gen] pool: {total_pool} chunks across {len(pools)} QID(s)")
    if total_pool == 0:
        raise SystemExit("[sentences_gen] no chunks matched; check the QIDs")

    qids = {e["qid"] for e in artifact["entities"]}
    order = sampling_order(pools, args.seed)
    accepted, reasons = collect_sentences(
        es, index, order, qids, args.num_sentences, splitter, thresholds,
        args.batch_size)

    sentences = [a["sentence"] for a in accepted]
    report = length_report(sentences)
    artifact["harvest"] = {
        "accepted": len(accepted),
        "rejections": dict(sorted(reasons.items())),
        "per_qid": {q: sum(1 for a in accepted if a["qid"] == q) for q in sorted(qids)},
    }
    artifact["length_report"] = report

    with open(run_dir / "chunk_ids.json", "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2, ensure_ascii=False)
        f.write("\n")
    with open(run_dir / "harvested_sentences.json", "w", encoding="utf-8") as f:
        json.dump(accepted, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"[sentences_gen] harvested {len(accepted)}/{args.num_sentences}")
    if artifact["harvest"]["rejections"]:
        print(f"[sentences_gen] rejections: {artifact['harvest']['rejections']}")
    print(f"[sentences_gen] per QID   : {artifact['harvest']['per_qid']}")
    print_report(report)

    problems = validate_distribution(report, args.num_sentences)
    if problems:
        print("[sentences_gen] VALIDATION FAILED:", file=sys.stderr)
        for problem in problems:
            print(f"    - {problem}", file=sys.stderr)
        print(f"[sentences_gen] diagnostics kept in {run_dir}", file=sys.stderr)
        print("[sentences_gen] concept_sentences.json NOT written", file=sys.stderr)
        return 1

    out_path = run_dir / "concept_sentences.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([{"concept": concept, "sentences": sentences}], f,
                  indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"[sentences_gen] wrote {out_path}")
    print(f"[sentences_gen] use it with: --concept-json {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
