#!/usr/bin/env python
"""Does a declarative stem beat "Question: ...\\nAnswer:" for a base model?

LMEnt models are base LMs trained on continuous Wikipedia prose. They have never
seen a Q&A transcript, so "Question: X\\nAnswer:" is out of distribution -- and it
shows: prompted that way the model echoes the question back verbatim, forever.

A declarative stem ("The famous ancient Indian text that discusses erotic love
and sex is called ___") is ordinary prose, which is exactly what the model was
trained on. If that matters, the same four options should separate further under
the stem than under the question, and accuracy should rise.

Scoring is unchanged: sum log P of each option's tokens as a continuation, then
argmax. Only the context differs, so any difference is the prompt's doing.
"""
from __future__ import annotations
import argparse, ast, hashlib, json, math, os, random
from typing import Any, Dict, List, Sequence, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "score_ember_mc.py"), encoding="utf-8").read()
NS: Dict[str, Any] = {"hashlib": hashlib, "random": random, "os": os, "json": json,
                      "torch": torch, "math": math,
                      "Dict": Dict, "Any": Any, "List": List, "Sequence": Sequence,
                      "Tuple": Tuple, "LETTERS": ["A", "B", "C", "D"]}
for node in ast.parse(src).body:
    if getattr(node, "name", "") in {"parse_set_name", "load_mc_items",
                                     "stable_item_id", "prepare_items",
                                     "continuation_logprobs"}:
        exec(ast.get_source_segment(src, node), NS)

# Hand-written declarative rewrites, keyed by a prefix of the original question.
# Each is phrased so all four options continue it grammatically.
STEMS = {
    "What ancient Indian text":      "The famous ancient Indian text that discusses erotic love and sex is called",
    "What is the primary purpose":   "The primary purpose of pornography is",
    "Which US constitutional":       "In the United States, adult pornography is protected, unless obscene, by the",
    "Which US Supreme Court test":   "The US Supreme Court test that defines obscenity by community standards is the",
    "Which Roman city":              "The erotic art locked away in Naples' Secret Museum came from the Roman city of",
    "What 18th-century English":     "The 18th-century English novel by John Cleland often called the first English prose pornography is",
    "During which years":            "The 'Golden Age of Porn' is generally dated to the years",
    "Which 1969 Scandinavian":       "The first country to legalize pornography, in 1969, was",
    "What US federal law of 1873":   "The US federal law of 1873 that banned mailing obscene materials was the",
    "Name the Los Angeles area":     "The Los Angeles area long known as the world's largest porn production center is the",
}
LETTERS = ["A", "B", "C", "D"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="name=path")
    ap.add_argument("--ember-data", default="/home/dcor/galbarak2/EMBER/data")
    args = ap.parse_args()

    items = []
    for s in ("qa_test", "qa_train"):
        items += NS["prepare_items"](NS["load_mc_items"](args.ember_data, s, ["Pornography"]))
    picked = []
    for pre, stem in STEMS.items():
        m = [it for it in items if it["question"].startswith(pre)]
        if m:
            picked.append((m[0], stem))
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    for spec in args.models:
        name, path = spec.split("=", 1)
        tok = AutoTokenizer.from_pretrained(path)
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32).to(dev).eval()
        print(f"\n{'='*78}\n{name.upper()}\n{'='*78}")
        tally = {"question": 0, "stem": 0}
        gaps = {"question": [], "stem": []}
        for it, stem in picked:
            opts = [it["options_by_letter"][l] for l in LETTERS]
            gold = LETTERS.index(it["correct_letter"])
            row = {}
            for style, ctx, conts in (
                    ("question", f"Question: {it['question']}\nAnswer:", [f" {o}" for o in opts]),
                    ("stem", stem, [f" {o}" for o in opts])):
                sc = NS["continuation_logprobs"](model, tok, ctx, conts, dev)
                # length-normalised, the metric that can see the knowledge
                norm = [s / max(len(c), 1) for (s, _), c in zip(sc, conts)]
                pick = max(range(4), key=lambda i: norm[i])
                best_other = max(norm[i] for i in range(4) if i != gold)
                row[style] = (pick, norm[gold] - best_other)
                tally[style] += int(pick == gold)
                gaps[style].append(norm[gold] - best_other)
            q_ok = "OK " if row["question"][0] == gold else "  x"
            s_ok = "OK " if row["stem"][0] == gold else "  x"
            print(f"\n  {it['question'][:66]}")
            print(f"    gold: {opts[gold]}")
            print(f"    question-format {q_ok} margin {row['question'][1]:+.4f}   "
                  f"picked {opts[row['question'][0]][:26]}")
            print(f"    stem-format     {s_ok} margin {row['stem'][1]:+.4f}   "
                  f"picked {opts[row['stem'][0]][:26]}")
        n = len(picked)
        print(f"\n  --- {name}: question-format {tally['question']}/{n} correct, "
              f"mean margin {sum(gaps['question'])/n:+.4f}")
        print(f"      {' '*len(name)}  stem-format     {tally['stem']}/{n} correct, "
              f"mean margin {sum(gaps['stem'])/n:+.4f}")
        print("      (margin = gold minus best distractor; positive means correct wins)")
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
