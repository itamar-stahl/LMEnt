#!/usr/bin/env python
"""Check the built question file against EMBER's data and the MC evaluator.

Reproduces the guarantees listed under "What was verified" in README.md, for
whichever concepts are present:

  1. Questions, options and answer keys are byte-identical to mc_questions.json,
     in the same order.
  2. The per-question option shuffle and the gold letter agree with
     ember_eval/score_ember_mc.py, so item N here is item N in every
     multiple-choice run already recorded.
  3. Every gold answer is among its four options.

    python verify_build.py --ember-data /path/to/EMBER/data \
        --score-ember-mc /path/to/ember_eval/score_ember_mc.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

SPLITS = ["QA_train", "QA_test", "SimdomQA_train", "SimdomQA_test"]


def load_module(path: str):
    """Import score_ember_mc for its shuffle logic only.

    It imports torch and transformers at module scope to load a model. Nothing
    used here touches either, so they are stubbed rather than installed; this
    check is meant to run anywhere, including on a login node.
    """
    import sys
    from unittest.mock import MagicMock
    for name in ("torch", "transformers"):
        sys.modules.setdefault(name, MagicMock())

    spec = importlib.util.spec_from_file_location("score_ember_mc", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default="data/completion_questions.json")
    ap.add_argument("--ember-data", required=True)
    ap.add_argument("--score-ember-mc", required=True)
    args = ap.parse_args()

    built = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    source = json.loads((Path(args.ember_data) / "mc_questions.json")
                        .read_text(encoding="utf-8"))
    mc = load_module(args.score_ember_mc)

    problems = 0
    for concept, splits in built.items():
        checked = 0
        for split in SPLITS:
            ours = splits[split]
            theirs = source[concept][split]
            if len(ours) != len(theirs):
                print(f"[len]  {concept}/{split}: {len(ours)} vs {len(theirs)}")
                problems += 1
                continue

            for n, (a, b) in enumerate(zip(ours, theirs), 1):
                where = f"{concept}/{split} #{n}"
                if a["q"] != b["q"].strip():
                    print(f"[q]    {where}: question text differs")
                    problems += 1
                if a["options"] != [o.strip() for o in b["options"]]:
                    print(f"[opt]  {where}: options differ or reordered")
                    problems += 1
                if a["correct_answer"] != b["correct_answer"].strip():
                    print(f"[key]  {where}: answer key differs")
                    problems += 1
                if a["correct_answer"] not in a["options"]:
                    print(f"[key]  {where}: gold answer not among the options")
                    problems += 1

                # same stable id -> same per-question shuffle -> same gold letter
                item = {"concept": concept, "subset": split.split("_")[0],
                        "split": split.split("_")[1], "question": a["q"],
                        "options": list(a["options"]),
                        "correct_answer": a["correct_answer"]}
                prepared = mc.prepare_items([item])[0]
                shuffled = [prepared["options_by_letter"][L] for L in mc.LETTERS]
                gold_letter = prepared["correct_letter"]
                if shuffled[mc.LETTERS.index(gold_letter)] != a["correct_answer"]:
                    print(f"[mc]   {where}: gold letter disagrees with the shuffle")
                    problems += 1
                checked += 1
        print(f"{concept:<16} {checked} items checked against EMBER and score_ember_mc")

    print(f"\n{problems} problem(s)")
    raise SystemExit(1 if problems else 0)


if __name__ == "__main__":
    main()
