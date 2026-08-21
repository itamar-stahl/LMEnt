#!/usr/bin/env python
"""Generate free-form answers from both twins, side by side, on one concept.

A diagnostic, not an evaluation. Two questions it answers before anyone builds
an LLM-judge pipeline on top of a base model:

  1. Does a base LMEnt model produce a gradeable answer at all? EMBER's
     multiple-choice protocol fails on it (it wants a bare letter and gets prose),
     which is why this project scores option likelihoods instead. But EMBER's
     *open* protocol prompts "Question: ...\\nAnswer:" and lets the model continue,
     which is the natural shape for a base LM. Whether that yields something a
     judge can grade is an empirical question, so look.

  2. Do the twins visibly differ? If the ablated twin produces the right fact as
     readily as the control, the multiple-choice format was not hiding an effect
     and the knowledge genuinely survived the ablation.
"""
from __future__ import annotations

import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def gold(item: dict) -> str:
    """open_questions.json stores the answer under "a"; mc_questions.json under
    "correct_answer". Accept either so the probe works on both files."""
    return item.get("a", item.get("correct_answer", ""))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="name=path pairs")
    ap.add_argument("--questions", default="/home/dcor/galbarak2/EMBER/data/open_questions.json")
    ap.add_argument("--concept", default="Pornography")
    ap.add_argument("--split", default="QA_test")
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    args = ap.parse_args()

    items = json.load(open(args.questions, encoding="utf-8"))[args.concept][args.split][:args.limit]
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    gens = {}
    for spec in args.models:
        name, path = spec.split("=", 1)
        tok = AutoTokenizer.from_pretrained(path)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32).to(dev).eval()
        outs = []
        for it in items:
            prompt = f"Question: {it['q']}\nAnswer:"
            ids = tok(prompt, return_tensors="pt").to(dev)
            with torch.no_grad():
                out = model.generate(**ids, max_new_tokens=args.max_new_tokens,
                                     do_sample=False, pad_token_id=tok.pad_token_id)
            outs.append(tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
                        .strip().replace("\n", " ")[:110])
        gens[name] = outs
        del model
        torch.cuda.empty_cache()

    for i, it in enumerate(items):
        print(f"\nQ: {it['q']}")
        print(f"  GOLD    : {gold(it)}")
        for name in gens:
            print(f"  {name:<8}: {gens[name][i]}")


if __name__ == "__main__":
    main()
