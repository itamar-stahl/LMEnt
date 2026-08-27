"""Offline checks: pure logic, no Elasticsearch, no network.

Everything here runs anywhere, including a laptop. If these fail on the login
node, the problem is the code, not the deployment.
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter
from pathlib import Path

from testlib import (
    MODULE_DIR, Suite, assert_eq, assert_in_range, assert_true,
)

import blacklist_to_concept_sentences as M

suite = Suite("units (offline)")

SHIPPED = MODULE_DIR.parent / "data" / "concept_sentences.json"
TH = M.active_thresholds(M.DEFAULT_THRESHOLDS)


def _cand(qid, hyperlinks=0.0, entity_linking=0.0, coref=0.0, coref_cluster=0.0):
    return {"qid": qid, "scores_by_source": {
        "hyperlinks": hyperlinks, "entity_linking": entity_linking,
        "coref": coref, "coref_cluster": coref_cluster}}


def _doc(text, mentions):
    return {"text": text, "title": "T", "entities": [
        {"char_start": s, "char_end": e, "text_mention": text[s:e],
         "candidates": cands} for s, e, cands in mentions]}


# --------------------------------------------------------------------------- #

@suite.case("concept name: underscores, small words, first word")
def _():
    assert_eq(M.concept_name_from_path(Path("harry_potter.json")),
              "Harry Potter", "harry_potter")
    assert_eq(M.concept_name_from_path(Path("pornography.json")),
              "Pornography", "pornography")
    assert_eq(M.concept_name_from_path(Path("republic_of_ireland.json")),
              "Republic of Ireland", "small word stays lowercase")
    assert_eq(M.concept_name_from_path(Path("of_mice.json")),
              "Of Mice", "first word capitalised even if a small word")


@suite.case("splitter: offsets are exact for every span")
def _():
    texts = [
        "Harry Potter is a series by J. K. Rowling. It sold well. Done.",
        "One sentence only",
        "Dr. Smith met Mr. Jones at 3 p.m. They talked. Then left!",
        'He said "stop." She did not. The end.',
    ]
    for text in texts:
        for start, end, body in M.split_sentences(text):
            assert_eq(text[start:end], body, f"span offset in {text[:24]!r}")


@suite.case("splitter: abbreviations and initials do not split")
def _():
    spans = M.split_sentences("Written by J. K. Rowling. Published by Prof. Bloom in 1997.")
    assert_eq(len(spans), 2, "sentence count")
    assert_true("J. K. Rowling" in spans[0][2], "initials kept together")
    assert_true("Prof. Bloom" in spans[1][2], "title abbreviation kept together")


@suite.case("splitter: never loses or duplicates text")
def _():
    text = "Alpha beta. Gamma delta! Epsilon? Zeta."
    spans = M.split_sentences(text)
    joined = " ".join(b for _, _, b in spans)
    assert_eq(joined.replace(" ", ""), text.replace(" ", ""), "content preserved")


@suite.case("mention: lexical evidence beats a coref pronoun in the same chunk")
def _():
    text = ("A neutral opening line about the weather here. "
            "He returned to the castle that evening and slept. "
            "Hogwarts is a school of witchcraft founded long ago. "
            "Trailing fragment cut")
    i_pron = text.index("He returned")
    i_lex = text.index("Hogwarts")
    doc = _doc(text, [
        (i_pron, i_pron + 2, [_cand("Q8337", coref=0.9, coref_cluster=0.9)]),
        (i_lex, i_lex + 8, [_cand("Q8337", hyperlinks=1.0, entity_linking=0.9)]),
    ])
    sentence, reason = M.harvest(doc, {"Q8337"}, M.split_sentences, TH)
    assert_eq(reason, "ok", "harvest reason")
    assert_true(sentence.startswith("Hogwarts"), f"picked {sentence!r}")


@suite.case("mention: a pronoun surface is refused even at coref 1.0")
def _():
    # The threshold alone cannot protect us here: coref 1.0 clears 0.95, so the
    # only thing standing between a pronoun and the corpus is PRONOUN_MENTIONS.
    text = "Opening line here about nothing. He went away quietly today. Tail cut"
    i = text.index("He went")
    doc = _doc(text, [(i, i + 2, [_cand("Q8337", coref=1.0, coref_cluster=1.0)])])
    assert_eq(M.harvest(doc, {"Q8337"}, M.split_sentences, TH)[1],
              "no_qualifying_mention", "pronoun surface must never anchor")


@suite.case("mention: a coref head at 0.95 qualifies (it is not a pronoun)")
def _():
    # Coref scores every cluster member including the head, so a high-scoring
    # coref mention is usually the proper noun itself. That is why the source
    # is retained at 0.95 rather than dropped.
    text = "An opening line of text. Hogwarts stood on a cliff above the lake. Tail"
    i = text.index("Hogwarts")
    doc = _doc(text, [(i, i + 8, [_cand("Q8337", coref=0.95)])])
    sentence, reason = M.harvest(doc, {"Q8337"}, M.split_sentences, TH)
    assert_eq(reason, "ok", "coref head at the threshold is accepted")
    assert_true(sentence.startswith("Hogwarts"), f"picked {sentence!r}")


@suite.case("mention: coref below 0.95 is refused")
def _():
    text = "An opening line of text. Hogwarts stood on a cliff above the lake. Tail"
    i = text.index("Hogwarts")
    doc = _doc(text, [(i, i + 8, [_cand("Q8337", coref=0.9, coref_cluster=0.9)])])
    assert_eq(M.harvest(doc, {"Q8337"}, M.split_sentences, TH)[1],
              "no_qualifying_mention", "0.9 < 0.95")


@suite.case("mention: entity_linking alone qualifies (abstract concepts need it)")
def _():
    text = "An opening line of text. Pornography was widely debated at the time. Tail"
    i = text.index("Pornography")
    doc = _doc(text, [(i, i + 11, [_cand("Q291", entity_linking=0.75)])])
    sentence, reason = M.harvest(doc, {"Q291"}, M.split_sentences, TH)
    assert_eq(reason, "ok", "EL-only accepted")
    assert_true("Pornography" in sentence, "sentence contains the mention")


@suite.case("mention: entity_linking below 0.7 is refused")
def _():
    text = "An opening line of text. Pornography was widely debated at the time. Tail"
    i = text.index("Pornography")
    doc = _doc(text, [(i, i + 11, [_cand("Q291", entity_linking=0.65)])])
    assert_eq(M.harvest(doc, {"Q291"}, M.split_sentences, TH)[1],
              "no_qualifying_mention", "0.65 is under our 0.7, over the paper's 0.6")


@suite.case("mention: ranked by source trust, not by raw score")
def _():
    # A coref mention scoring 1.0 must lose to a hyperlink mention scoring 1.0,
    # because SOURCE_PRIORITY puts hyperlinks first.
    text = ("A neutral opening line about the weather here. "
            "Azkaban held the prisoners on a rock in the sea. "
            "Hogwarts is a school of witchcraft founded long ago. "
            "Trailing fragment cut")
    i_coref = text.index("Azkaban")
    i_link = text.index("Hogwarts")
    doc = _doc(text, [
        (i_coref, i_coref + 7, [_cand("Q8337", coref=1.0, coref_cluster=1.0)]),
        (i_link, i_link + 8, [_cand("Q8337", hyperlinks=1.0)]),
    ])
    sentence, reason = M.harvest(doc, {"Q8337"}, M.split_sentences, TH)
    assert_eq(reason, "ok", "harvest reason")
    assert_true(sentence.startswith("Hogwarts"),
                f"hyperlink mention should win, picked {sentence!r}")


@suite.case("mention: offset mismatch is caught, never mis-harvested")
def _():
    text = "First sentence is here now. Hogwarts is a school of magic today. Tail"
    i = text.index("Hogwarts")
    doc = _doc(text, [(i, i + 8, [_cand("Q8337", hyperlinks=1.0)])])
    doc["entities"][0]["text_mention"] = "Hogwarts"
    doc["entities"][0]["char_start"] = 2          # deliberately wrong
    doc["entities"][0]["char_end"] = 10
    assert_eq(M.harvest(doc, {"Q8337"}, M.split_sentences, TH)[1],
              "offset_mismatch", "shifted offset must reject")


@suite.case("mention: truncated chunk edges are dropped")
def _():
    text = "Hogwarts is a school of witchcraft and this line runs off the chunk edge"
    doc = _doc(text, [(0, 8, [_cand("Q8337", hyperlinks=1.0)])])
    assert_eq(M.harvest(doc, {"Q8337"}, M.split_sentences, TH)[1],
              "truncated_span", "unterminated final span")


@suite.case("mention: a different QID's mention is ignored")
def _():
    text = "An opening line of text. Hogwarts is a school of magic today. Tail"
    i = text.index("Hogwarts")
    doc = _doc(text, [(i, i + 8, [_cand("Q999999", hyperlinks=1.0)])])
    assert_eq(M.harvest(doc, {"Q8337"}, M.split_sentences, TH)[1],
              "no_qualifying_mention", "foreign QID must not anchor")


@suite.case("sampling: rarest-QID assignment removes double counting")
def _():
    resolved = [
        {"qid": "Q8337", "comment": "", "num_chunks": 2200,
         "chunk_ids": list(range(2200))},
        {"qid": "Q3244512", "comment": "", "num_chunks": 600,
         "chunk_ids": list(range(1800, 2400))},
    ]
    pools = M.assign_rarest(resolved)
    assert_eq(sum(len(v) for v in pools.values()), 2400, "union size")
    assert_eq(len(pools["Q3244512"]), 600, "rare QID keeps all its chunks")
    assert_eq(len(pools["Q8337"]), 1800, "common QID yields the overlap")


@suite.case("sampling: round-robin balances across uneven pools")
def _():
    pools = {"Qbig": list(range(3000)), "Qsmall": list(range(9000, 9600))}
    drawn = [q for _, (_, q) in zip(range(300), M.sampling_order(pools, 42))]
    counts = Counter(drawn)
    assert_eq(counts["Qbig"], 150, "big pool share")
    assert_eq(counts["Qsmall"], 150, "small pool share")


@suite.case("sampling: exhausted small pool falls through to the big one")
def _():
    pools = {"Qbig": list(range(5000)), "Qtiny": list(range(9000, 9010))}
    drawn = [q for _, (_, q) in zip(range(60), M.sampling_order(pools, 42))]
    assert_eq(Counter(drawn)["Qtiny"], 10, "tiny pool capped at its size")


@suite.case("sampling: same seed identical, different seed different")
def _():
    pools = {"Qa": list(range(1000)), "Qb": list(range(5000, 5400))}
    first = [x for _, x in zip(range(200), M.sampling_order(pools, 42))]
    again = [x for _, x in zip(range(200), M.sampling_order(pools, 42))]
    other = [x for _, x in zip(range(200), M.sampling_order(pools, 7))]
    assert_eq(first, again, "seed 42 is reproducible")
    assert_true(first != other, "seed 7 differs from seed 42")


@suite.case("sampling: a QID's draw does not depend on its neighbours")
def _():
    alone = {"Qa": list(range(1000))}
    crowded = {"Qa": list(range(1000)), "Qb": list(range(5000, 5400))}
    solo = [cid for _, (cid, _) in zip(range(50), M.sampling_order(alone, 42))]
    mixed = [cid for _, (cid, qid) in zip(range(300), M.sampling_order(crowded, 42))
             if qid == "Qa"][:50]
    assert_eq(solo, mixed, "per-QID seeding is independent of the pool set")


@suite.case("dedup: normalisation collapses case and punctuation")
def _():
    assert_eq(M.normalize_key("Harry Potter, the boy!"),
              M.normalize_key("harry potter the boy"), "normalised keys")
    assert_true(M.normalize_key("A") != M.normalize_key("B"), "distinct stay distinct")


@suite.case("validation: all 18 shipped concepts pass the gate")
def _():
    records = json.loads(SHIPPED.read_text(encoding="utf-8"))
    for record in records:
        problems = M.validate_distribution(
            M.length_report(record["sentences"]), 300)
        assert_true(not problems, f"{record['concept']}: {problems}")


@suite.case("validation: degenerate corpora are rejected")
def _():
    cases = {
        "uniform lengths": [" ".join(["w"] * (20 + i % 3)) for i in range(300)],
        "identical lengths": [" ".join(["w"] * 21) for _ in range(300)],
        "too short": [" ".join(["w"] * (5 + i % 4)) for i in range(300)],
        "too long": [" ".join(["w"] * (55 + i % 11)) for i in range(300)],
        "short count": [" ".join(["w"] * (5 + i % 40)) for i in range(250)],
    }
    for name, sentences in cases.items():
        problems = M.validate_distribution(M.length_report(sentences), 300)
        assert_true(bool(problems), f"{name} should have failed the gate")


@suite.case("validation: the distinct-length bar scales with the target")
def _():
    # A corpus can never hold more distinct word-counts than sentences, so a
    # fixed bar calibrated on 300 is unreachable on a 40-sentence smoke run.
    assert_eq(M.min_distinct_lengths(300), 30, "full run keeps the old bar")
    assert_eq(M.min_distinct_lengths(40), 13, "reduced run scales down")
    assert_eq(M.min_distinct_lengths(10), 8, "floor holds for tiny runs")
    for target in (10, 40, 100, 300):
        assert_true(M.min_distinct_lengths(target) <= target,
                    f"bar for {target} must be reachable")


@suite.case("validation: 27-of-40 diversity passes, 5-of-40 does not")
def _():
    # The exact case that failed on c-003: 27 distinct counts in a 40-sentence
    # run is 67% diversity, better than the full run's 56/300.
    import random
    rng = random.Random(3)
    lengths = list(range(8, 35))[:27]
    while len(lengths) < 40:
        lengths.append(rng.choice(lengths))
    good = [" ".join(["w"] * n) for n in lengths]
    report = M.length_report(good)
    assert_eq(report["distinct_lengths"], 27, "fixture diversity")
    assert_true(not M.validate_distribution(report, 40),
                f"27/40 should pass: {M.validate_distribution(report, 40)}")

    poor = [" ".join(["w"] * (18 + i % 5)) for i in range(40)]
    assert_true(bool(M.validate_distribution(M.length_report(poor), 40)),
                "5 distinct counts in 40 sentences should fail")


@suite.case("validation: a realistic distribution passes")
def _():
    rng = random.Random(0)
    sentences = [" ".join(["w"] * max(5, min(120, int(rng.gauss(21, 11)))))
                 for _ in range(300)]
    report = M.length_report(sentences)
    assert_true(not M.validate_distribution(report, 300),
                f"healthy set rejected: {report}")
    assert_in_range(report["median"], 15, 30, "median")


@suite.case("query: two nested levels, QID terms, threshold should-clause")
def _():
    query = M.build_entity_query(["Q8337"], M.DEFAULT_THRESHOLDS)
    assert_eq(query["nested"]["path"], "entities", "outer nested path")
    inner = query["nested"]["query"]["nested"]
    assert_eq(inner["path"], "entities.candidates", "inner nested path")
    filters = inner["query"]["bool"]["filter"]
    assert_eq(filters[0]["terms"]["entities.candidates.qid"], ["Q8337"], "qid terms")
    assert_eq(filters[1]["bool"]["minimum_should_match"], 1, "any-source semantics")
    assert_eq(len(filters[1]["bool"]["should"]),
              len(M.DEFAULT_THRESHOLDS), "one clause per retrieved source")


@suite.case("query: defaults are stricter than the paper on every weak source")
def _():
    assert_eq(M.DEFAULT_THRESHOLDS,
              {"hyperlinks": 1.0, "entity_linking": 0.7,
               "coref": 0.95, "coref_cluster": 0.95}, "default thresholds")
    paper = {"entity_linking": 0.6, "coref": 0.6, "coref_cluster": 0.6}
    for source, loose in paper.items():
        assert_true(M.DEFAULT_THRESHOLDS[source] > loose,
                    f"{source} must be stricter than the paper's {loose}")


@suite.case("thresholds: DISABLED drops a source from the query entirely")
def _():
    raw = dict(M.DEFAULT_THRESHOLDS)
    raw["coref"] = M.DISABLED
    raw["coref_cluster"] = M.DISABLED
    active = M.active_thresholds(raw)
    assert_eq(sorted(active), ["entity_linking", "hyperlinks"], "surviving sources")

    query = M.build_entity_query(["Q8337"], active)
    clauses = query["nested"]["query"]["nested"]["query"]["bool"]["filter"][1]
    fields = {list(c["range"])[0] for c in clauses["bool"]["should"]}
    for source in ("coref", "coref_cluster"):
        assert_true(
            f"entities.candidates.scores_by_source.{source}" not in fields,
            f"disabled {source} must not appear in the query")
    assert_eq(len(fields), 2, "one clause per surviving source")


@suite.case("thresholds: DISABLED also stops the source qualifying a mention")
def _():
    # The same resolved dict gates both stages, so a disabled source cannot
    # fetch a chunk it would not be allowed to anchor.
    text = "An opening line of text. Hogwarts stood on a cliff above the lake. Tail"
    i = text.index("Hogwarts")
    doc = _doc(text, [(i, i + 8, [_cand("Q8337", coref=1.0)])])
    assert_eq(M.harvest(doc, {"Q8337"}, M.split_sentences, TH)[1],
              "ok", "enabled by default")
    without = M.active_thresholds({**M.DEFAULT_THRESHOLDS, "coref": M.DISABLED})
    assert_eq(M.harvest(doc, {"Q8337"}, M.split_sentences, without)[1],
              "no_qualifying_mention", "disabling coref must reject it")


@suite.case("thresholds: bad values and total disablement are refused")
def _():
    for bad in (1.5, -0.5, 2.0):
        try:
            M.active_thresholds({**M.DEFAULT_THRESHOLDS, "coref": bad})
        except SystemExit:
            pass
        else:
            raise AssertionError(f"threshold {bad} should have been rejected")
    try:
        M.active_thresholds({s: M.DISABLED for s in M.DEFAULT_THRESHOLDS})
    except SystemExit:
        pass
    else:
        raise AssertionError("disabling every source should have been refused")


@suite.case("thresholds: an empty should-clause can never be generated")
def _():
    # bool.should with minimum_should_match:1 and zero clauses matches nothing.
    # active_thresholds is what makes that state unreachable.
    query = M.build_entity_query(["Q8337"], TH)
    clauses = query["nested"]["query"]["nested"]["query"]["bool"]["filter"][1]
    assert_eq(len(clauses["bool"]["should"]), len(TH),
              "one clause per active source")
    assert_true(len(clauses["bool"]["should"]) > 0, "should-clause is non-empty")


@suite.case("blacklist: object and bare-string entities both load")
def _():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "x.json"
        path.write_text(json.dumps(
            {"entities": [{"qid": "Q1", "comment": "c"}, "Q2"]}), encoding="utf-8")
        entities = M.load_blacklist(path)
        assert_eq([e["qid"] for e in entities], ["Q1", "Q2"], "qids")
        assert_eq(entities[1]["comment"], "", "bare string gets empty comment")


if __name__ == "__main__":
    sys.exit(suite.run())
