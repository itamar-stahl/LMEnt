#!/usr/bin/env python
"""Build the three question sets (target / neighbour / unrelated) for one topic.

Everything comes from `ember_eval/completion_eval/data/completion_questions.json`
-- EMBER's questions rewritten as declarative stems, all 18 topics. No new
questions are written; the only choice made here is WHICH existing questions
form the unrelated set, and that is a fixed-seed draw recorded in the output.

    set        selection (HP choice)            test (reported)
    target     <topic> QA_train (50)            <topic> QA_test (50)
    neighbour  <topic> SimdomQA_train (50)      <topic> SimdomQA_test (50)
    unrelated  50 from other topics' QA_train   50 from other topics' QA_test

"Other topics" excludes the target and any EMBER topic that IS its neighbouring
domain (Culture of Greece for Rome, Golf for Baseball), so an unrelated
question really is unrelated. Selection and test never share an item; the
script asserts it.

Each item carries the exact string the scorer conditions on (`stem`) and the
exact string it scores (`completion`, WITH the leading space -- the tokenizer
treats " Augustus" and "Augustus" differently and the leading-space form is
what a continuation of the stem looks like).

    python ember_eval/nll_kl/build_sets.py --topic "Ancient Rome" --out sets/rome.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
QUESTIONS = HERE.parents[0] / "completion_eval" / "data" / "completion_questions.json"

SEED = 20260920
N_UNRELATED = 50

# Topics whose whole domain is the target's neighbouring domain and therefore
# cannot supply "unrelated" questions. The SimdomQA sets already cover these.
NEIGHBOUR_TOPICS: Dict[str, List[str]] = {
    "Ancient Rome": ["Culture of Greece"],
    "Baseball": ["Golf"],
    "Artificial intelligence": [],
}


def item_id(topic: str, split: str, question: str) -> str:
    return hashlib.sha1(f"{topic}\x1f{split}\x1f{question}".encode("utf-8")).hexdigest()[:12]


def make_item(topic: str, split: str, raw: Dict[str, Any], role: str, phase: str) -> Dict[str, Any]:
    stem = raw["stem"].strip()
    answer = raw["correct_answer"].strip()
    if not stem or not answer:
        raise SystemExit(f"empty stem or answer in {topic}/{split}: {raw.get('q')!r}")
    return {
        "id": item_id(topic, split, raw["q"].strip()),
        "role": role,            # target | neighbour | unrelated
        "phase": phase,          # selection | test
        "source_topic": topic,
        "source_split": split,
        "question": raw["q"].strip(),
        "stem": stem,
        "completion": " " + answer,
    }


def build(topic: str, data: Dict[str, Any]) -> Dict[str, Any]:
    if topic not in data:
        raise SystemExit(f"{topic!r} not in {QUESTIONS}; have {sorted(data)}")
    excluded = [topic] + NEIGHBOUR_TOPICS.get(topic, [])
    others = sorted(t for t in data if t not in excluded)

    sets: Dict[str, List[Dict[str, Any]]] = {}
    sets["target_selection"] = [make_item(topic, "QA_train", r, "target", "selection")
                                for r in data[topic]["QA_train"]]
    sets["target_test"] = [make_item(topic, "QA_test", r, "target", "test")
                           for r in data[topic]["QA_test"]]
    sets["neighbour_selection"] = [make_item(topic, "SimdomQA_train", r, "neighbour", "selection")
                                   for r in data[topic]["SimdomQA_train"]]
    sets["neighbour_test"] = [make_item(topic, "SimdomQA_test", r, "neighbour", "test")
                              for r in data[topic]["SimdomQA_test"]]

    rng = random.Random(SEED)
    for phase, split in (("selection", "QA_train"), ("test", "QA_test")):
        pool = [make_item(t, split, r, "unrelated", phase)
                for t in others for r in data[t][split]]
        pool.sort(key=lambda it: it["id"])   # order independent of dict order
        rng.shuffle(pool)
        sets[f"unrelated_{phase}"] = pool[:N_UNRELATED]

    # No item may sit in both a selection set and a test set.
    sel = {it["id"] for k, v in sets.items() if k.endswith("_selection") for it in v}
    tst = {it["id"] for k, v in sets.items() if k.endswith("_test") for it in v}
    if sel & tst:
        raise SystemExit(f"{len(sel & tst)} items shared between selection and test")
    # Also by (question, answer) text, in case the same question is filed twice.
    key = lambda it: (it["question"], it["completion"])
    sel_k = {key(it) for k, v in sets.items() if k.endswith("_selection") for it in v}
    tst_k = {key(it) for k, v in sets.items() if k.endswith("_test") for it in v}
    if sel_k & tst_k:
        raise SystemExit(f"{len(sel_k & tst_k)} question/answer pairs shared between selection and test")

    for k, v in sets.items():
        if len(v) != 50:
            raise SystemExit(f"{k}: expected 50 items, got {len(v)}")

    return {
        "topic": topic,
        "source": str(QUESTIONS.relative_to(HERE.parents[1])),
        "seed": SEED,
        "excluded_from_unrelated": excluded,
        "unrelated_pool_topics": others,
        "sets": sets,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    data = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    out = build(a.topic, data)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    counts = {k: len(v) for k, v in out["sets"].items()}
    from collections import Counter
    src = Counter(it["source_topic"] for it in out["sets"]["unrelated_test"])
    print(f"{a.topic}: {counts}")
    print(f"  unrelated_test drawn from {len(src)} topics: {dict(src)}")
    print("  wrote", a.out)


if __name__ == "__main__":
    main()
