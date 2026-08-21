#!/usr/bin/env python
"""Show the full multiple-choice prompt to the model and print what it generates.

Our scorers never generate: the text_* scorer computes the likelihood of each
option text as a continuation of "Question: ...\\nAnswer:" (the model never sees
A/B/C/D), and the letter scorer compares the likelihoods of " A"/" B"/" C"/" D".
Neither tells you what the model would *say*. This does, using EMBER's own
prompt verbatim, so the question "does it answer 'Sexual arousal'?" gets an
observation rather than an argument.
"""
from __future__ import annotations
import argparse, ast, hashlib, json, os, random
from typing import Any, Dict, List, Sequence, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "score_ember_mc.py"), encoding="utf-8").read()
NS: Dict[str, Any] = {"hashlib": hashlib, "random": random, "os": os, "json": json,
                      "Dict": Dict, "Any": Any, "List": List, "Sequence": Sequence,
                      "Tuple": Tuple, "LETTERS": ["A", "B", "C", "D"]}
for node in ast.parse(src).body:
    if getattr(node, "name", "") in {"parse_set_name", "load_mc_items", "stable_item_id",
                                     "prepare_items", "build_mc_prompt"}:
        exec(ast.get_source_segment(src, node), NS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True, help="name=path")
    ap.add_argument("--ember-data", default="/home/dcor/galbarak2/EMBER/data")
    ap.add_argument("--concept", default="Pornography")
    ap.add_argument("--limit", type=int, default=6)
    ap.add_argument("--max-new-tokens", type=int, default=20)
    args = ap.parse_args()

    items = NS["prepare_items"](NS["load_mc_items"](args.ember_data, "qa_test", [args.concept]))[:args.limit]
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    out: Dict[str, Dict[str, List[str]]] = {}
    for spec in args.models:
        name, path = spec.split("=", 1)
        tok = AutoTokenizer.from_pretrained(path)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32).to(dev).eval()
        mc, cloze = [], []
        for it in items:
            for style, prompt, sink in (
                    ("mc", NS["build_mc_prompt"](it["question"], it["options_by_letter"]), mc),
                    ("cloze", f"Question: {it['question']}\nAnswer:", cloze)):
                ids = tok(prompt, return_tensors="pt").to(dev)
                with torch.no_grad():
                    g = model.generate(**ids, max_new_tokens=args.max_new_tokens,
                                       do_sample=False, pad_token_id=tok.pad_token_id)
                sink.append(tok.decode(g[0][ids["input_ids"].shape[1]:],
                                       skip_special_tokens=True).replace("\n", " ⏎ ")[:100])
        out[name] = {"mc": mc, "cloze": cloze}
        del model
        torch.cuda.empty_cache()

    for i, it in enumerate(items):
        print("=" * 78)
        print(f"Q: {it['question']}")
        for L in ["A", "B", "C", "D"]:
            print(f"   {L}. {it['options_by_letter'][L]}"
                  + ("   <-- CORRECT" if L == it["correct_letter"] else ""))
        for style, label in (("mc", "given the FULL prompt with options, it writes"),
                             ("cloze", "given 'Question: ...\\nAnswer:', it writes")):
            print(f"\n  {label}:")
            for name in out:
                print(f"    {name:<8}: {out[name][style][i]!r}")
        print()


if __name__ == "__main__":
    main()
