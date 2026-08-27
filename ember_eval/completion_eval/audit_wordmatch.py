#!/usr/bin/env python
"""Count questions answerable by word overlap alone, in the question and in the stem.

The limitation documented in README.md: some of EMBER's items can be answered by
matching a word in the question to a word in the correct option, with no
knowledge of the concept. "What magical *map* shows everyone's location at
Hogwarts?" against *The Marauder's **Map*** is one. A model that never saw the
concept can still get these, which puts a floor under an erased model's accuracy
and caps measurable efficacy.

An item counts as a word match when some content word appears in both the
context and the gold option and in none of the three distractors. Run over both
contexts:

  q     EMBER's original question. These are inherited and are in EMBER's own
        published numbers.
  stem  the rewrite. Any item here that is not also flagged under `q` is a leak
        the rewrite introduced, and is a bug in the stems file.

    python audit_wordmatch.py --questions data/completion_questions.json
    python audit_wordmatch.py --concept Baseball --verbose
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_completions import STOPWORDS, content_words  # noqa: E402

SPLITS = ["QA_train", "QA_test", "SimdomQA_train", "SimdomQA_test"]


def matches(context: str, gold: str, distractors: list[str]) -> set[str]:
    shared = content_words(gold) & content_words(context)
    for d in distractors:
        shared -= content_words(d)
    return shared


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="data/completion_questions.json")
    ap.add_argument("--concept", nargs="*", default=None)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    data = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    concepts = args.concept or list(data)
    introduced_total = 0

    for concept in concepts:
        n_q = n_stem = n_new = 0
        total = 0
        for split in SPLITS:
            for i, it in enumerate(data[concept][split], 1):
                total += 1
                gold = it["correct_answer"]
                others = [o for o in it["options"] if o != gold]
                in_q = matches(it["q"], gold, others)
                in_stem = matches(it["stem"], gold, others)
                n_q += bool(in_q)
                n_stem += bool(in_stem)
                if in_stem - in_q:
                    n_new += 1
                    print(f"  INTRODUCED  {concept} / {split} #{i}: "
                          f"{sorted(in_stem - in_q)}\n"
                          f"      stem: {it['stem']}\n      gold: {gold}")
                elif args.verbose and in_q:
                    print(f"  inherited   {concept} / {split} #{i}: {sorted(in_q)}"
                          f"  ({it['q']} -> {gold})")
        introduced_total += n_new
        print(f"{concept:<16} {n_q:>3}/{total} in the question, "
              f"{n_stem:>3}/{total} in the stem, {n_new} introduced by the rewrite")

    raise SystemExit(1 if introduced_total else 0)


if __name__ == "__main__":
    main()
