#!/usr/bin/env python
"""Six models x two prompt formats x the OLMES metric family, on one question set.

Two things are being tested at once, and they are separable:

  FORMAT   "Question: X\\nAnswer:" (multiple-choice-ish, what every analysis in
           this project used) against a declarative stem ("The famous ancient
           Indian text that discusses erotic love and sex is called"). OLMES calls
           these MCF and CF and reports that small models score near-random under
           MCF because it needs symbol-binding that only emerges with scale.

  METRIC   how the four option scores are normalised. OLMES names:
             acc_raw       summed log P                    (our old text_sum)
             acc_per_token / token count                   (text_tok)
             acc_per_char  / character count               (text_char)
             acc_uncond    log P(opt|ctx) - log P(opt|"Answer:")   <- NEW
           acc_uncond is PMI normalisation: it cancels how common the option
           string is on its own, so "Mahabharata" stops beating "Kama Sutra"
           merely for being a commoner word. Eight OLMES tasks use it as primary.

Six models: our control/ablated at 1 and 2 epochs, plus the authors' released 1E
and 2E as an external reference. The ablation effect is the control-minus-ablated
gap; the released models say what this corpus supports at all.
"""
from __future__ import annotations
import argparse, ast, hashlib, json, math, os, random
from typing import Any, Dict, List, Sequence, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "score_ember_mc.py"), encoding="utf-8").read()
NS: Dict[str, Any] = {"hashlib": hashlib, "random": random, "os": os, "json": json,
                      "torch": torch, "math": math, "Dict": Dict, "Any": Any,
                      "List": List, "Sequence": Sequence, "Tuple": Tuple,
                      "LETTERS": ["A", "B", "C", "D"]}
for node in ast.parse(src).body:
    if getattr(node, "name", "") in {"parse_set_name", "load_mc_items", "stable_item_id",
                                     "prepare_items", "continuation_logprobs"}:
        exec(ast.get_source_segment(src, node), NS)

STEMS = {
    "What ancient Indian text":    "The famous ancient Indian text that discusses erotic love and sex is called",
    "What is the primary purpose": "The primary purpose of pornography is",
    "Which US constitutional":     "In the United States, adult pornography is protected, unless obscene, by the",
    "Which US Supreme Court test": "The US Supreme Court test that defines obscenity by community standards is the",
    "Which Roman city":            "The erotic art locked away in Naples' Secret Museum came from the Roman city of",
    "What 18th-century English":   "The 18th-century English novel by John Cleland often called the first English prose pornography is",
    "During which years":          "The 'Golden Age of Porn' is generally dated to the years",
    "Which 1969 Scandinavian":     "The first country to legalize pornography, in 1969, was",
    "What US federal law of 1873": "The US federal law of 1873 that banned mailing obscene materials was the",
    "Name the Los Angeles area":   "The Los Angeles area long known as the world's largest porn production center is the",
}
LETTERS = ["A", "B", "C", "D"]
UNCOND = "Answer:"          # neutral context for the PMI term


def score(model, tok, ctx, opts, dev):
    conts = [f" {o}" for o in opts]
    cond = NS["continuation_logprobs"](model, tok, ctx, conts, dev)
    unc = NS["continuation_logprobs"](model, tok, UNCOND, conts, dev)
    out = {}
    out["acc_raw"] = [s for s, _ in cond]
    out["acc_per_token"] = [s / max(n, 1) for s, n in cond]
    out["acc_per_char"] = [s / max(len(c), 1) for (s, _), c in zip(cond, conts)]
    out["acc_uncond"] = [c - u for (c, _), (u, _) in zip(cond, unc)]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="name=path[@subfolder]")
    ap.add_argument("--ember-data", default="/home/dcor/galbarak2/EMBER/data")
    args = ap.parse_args()

    items = []
    for s in ("qa_test", "qa_train"):
        items += NS["prepare_items"](NS["load_mc_items"](args.ember_data, s, ["Pornography"]))
    picked = [(m[0], stem) for pre, stem in STEMS.items()
              if (m := [it for it in items if it["question"].startswith(pre)])]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    METRICS = ["acc_raw", "acc_per_token", "acc_per_char", "acc_uncond"]
    results: Dict[str, Dict[Tuple[str, str], Tuple[int, float]]] = {}

    for spec in args.models:
        name, loc = spec.split("=", 1)
        path, _, sub = loc.partition("@")
        kw = {"subfolder": sub} if sub else {}
        tok = AutoTokenizer.from_pretrained(path, **kw)
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32, **kw).to(dev).eval()
        res = {}
        for fmt in ("question", "stem"):
            for m in METRICS:
                res[(fmt, m)] = [0, 0.0]
        for it, stem in picked:
            opts = [it["options_by_letter"][l] for l in LETTERS]
            gold = LETTERS.index(it["correct_letter"])
            for fmt, ctx in (("question", f"Question: {it['question']}\nAnswer:"), ("stem", stem)):
                sc = score(model, tok, ctx, opts, dev)
                for m in METRICS:
                    v = sc[m]
                    res[(fmt, m)][0] += int(max(range(4), key=lambda i: v[i]) == gold)
                    res[(fmt, m)][1] += v[gold] - max(v[i] for i in range(4) if i != gold)
        results[name] = {k: (c, s / len(picked)) for k, (c, s) in res.items()}
        print(f"[done] {name}", flush=True)
        del model
        torch.cuda.empty_cache()

    n = len(picked)
    for m in METRICS:
        print(f"\n{'='*76}\nMETRIC: {m}   ({n} questions about the ablated concept)\n{'='*76}")
        print(f"  {'model':<22}{'question fmt':>16}{'stem fmt':>14}{'margin(stem)':>15}")
        print("  " + "-" * 66)
        for name in results:
            q = results[name][("question", m)]
            s = results[name][("stem", m)]
            print(f"  {name:<22}{f'{q[0]}/{n}':>16}{f'{s[0]}/{n}':>14}{s[1]:>+15.4f}")


if __name__ == "__main__":
    main()
