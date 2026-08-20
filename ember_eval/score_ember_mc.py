#!/usr/bin/env python
"""Score an LMEnt checkpoint on EMBER's multiple-choice questions.

Why not just call EMBER's evaluator: theirs (ember/evals/mc.py) prompts for a
bare letter, generates 4 tokens and regexes out an A-D. That is written for
gemma-2-2b-it and Llama-3.1-8B-Instruct. LMEnt is a *base* model with no
instruction tuning -- it continues text rather than answering with a letter, so
nearly every generation parses as invalid and every concept scores ~0. That
measures format compliance, not knowledge, and knowledge is the thing we need
in order to choose an ablation subject.

So both questions get asked, by log-likelihood rather than generation:

  text    the length-normalized log-likelihood of each option's *text* as a
          continuation of "Question: ...\\nAnswer:". The standard way to put a
          multiple-choice question to a base model (what olmes/lm-eval do for
          arc_easy), and the number to choose subjects on: it isolates what the
          model knows from whether it can follow an instruction.

  letter  EMBER's prompt verbatim, but comparing the log-likelihood of " A" /
          " B" / " C" / " D" instead of generating. Always yields a valid
          answer, and stays closest to EMBER's own numbers.

Option order is shuffled exactly as EMBER does it -- same stable item id, same
sha1-derived per-question seed -- so item N here is item N there.

Chance is 25% for both scorers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LETTERS = ["A", "B", "C", "D"]


# --------------------------------------------------------------------------- #
# EMBER-compatible loading and option shuffling
# --------------------------------------------------------------------------- #
def parse_set_name(set_name: str) -> Tuple[str, str]:
    """"qa_test" -> ("QA", "test"). Mirrors ember.local_datasets._parse_set_name."""
    parts = set_name.lower().split("_")
    subset_map = {"qa": "QA", "simdom": "SimdomQA"}
    if len(parts) != 2 or parts[1] not in {"train", "test"} or parts[0] not in subset_map:
        raise ValueError(f"bad set_name {set_name!r}: expected <qa|simdom>_<train|test>")
    return subset_map[parts[0]], parts[1]


def load_mc_items(data_dir: str, set_name: str, concepts: Sequence[str] | None) -> List[Dict[str, Any]]:
    """Mirrors ember.local_datasets.load_mc_qa_items, including the .strip() calls."""
    subset, split = parse_set_name(set_name)
    with open(os.path.join(data_dir, "mc_questions.json"), encoding="utf-8") as f:
        data = json.load(f)

    key = f"{subset}_{split}"
    items: List[Dict[str, Any]] = []
    for concept_name, concept_data in data.items():
        if concepts and concept_name not in concepts:
            continue
        for obj in concept_data.get(key, []):
            q = obj.get("q", "").strip()
            a = obj.get("correct_answer", "").strip()
            options = obj.get("options", []) or []
            if q and a and len(options) >= 4:
                items.append({
                    "concept": concept_name, "subset": subset, "split": split,
                    "question": q, "options": list(options), "correct_answer": a,
                })
    return items


def stable_item_id(item: Dict[str, Any]) -> str:
    """Mirrors ember.evals.mc.stable_item_id for an MCQAItem (which has no .answer)."""
    parts = [f"{k}={item[k]}" for k in
             ("concept", "subset", "split", "question", "correct_answer") if item.get(k) is not None]
    parts.append("options=" + "|".join(list(item["options"])))
    return hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()


def prepare_items(items: List[Dict[str, Any]], seed: int = 42) -> List[Dict[str, Any]]:
    """Mirrors ember.evals.mc.prepare_mc_items: per-question deterministic shuffle."""
    prepared = []
    for item in items:
        options = list(item["options"])
        sid = stable_item_id(item)
        per_q_seed = (seed + int(hashlib.sha1(sid.encode("utf-8")).hexdigest()[:8], 16)) % (2 ** 32)
        random.Random(per_q_seed).shuffle(options)

        correct = item["correct_answer"]
        if correct not in options:
            raise AssertionError(f"correct answer missing from options: {item['question']!r}")

        by_letter = dict(zip(LETTERS, options[:4]))
        correct_letter = next((L for L, o in by_letter.items() if o == correct), "")
        if not correct_letter:
            raise AssertionError(f"correct answer not in first 4 shuffled options: {item['question']!r}")

        prepared.append({**item, "options_by_letter": by_letter, "correct_letter": correct_letter})
    return prepared


def build_mc_prompt(question: str, by_letter: Dict[str, str]) -> str:
    """Verbatim from ember.evals.mc.build_mc_prompt."""
    lines = ["Choose the single best answer.", f"Question: {question}\n", "Choices:"]
    lines += [f"{L}. {by_letter[L]}" for L in LETTERS]
    lines.append("\nAnswer only with one letter (A, B, C, or D) and no formatting.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
@torch.no_grad()
def continuation_logprobs(model, tok, context: str, continuations: Sequence[str], device) -> List[Tuple[float, int]]:
    """(summed log P(continuation | context), token count) for each continuation.

    One padded batch per question. Padding is on the right and the model is
    causal, so positions before a pad are unaffected by it.
    """
    ctx_ids = tok(context, add_special_tokens=True)["input_ids"]
    seqs, cont_lens = [], []
    for cont in continuations:
        cont_ids = tok(cont, add_special_tokens=False)["input_ids"]
        seqs.append(ctx_ids + cont_ids)
        cont_lens.append(len(cont_ids))

    width = max(len(s) for s in seqs)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    input_ids = torch.full((len(seqs), width), pad_id, dtype=torch.long)
    attn = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, s in enumerate(seqs):
        input_ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        attn[i, :len(s)] = 1

    logits = model(input_ids=input_ids.to(device), attention_mask=attn.to(device)).logits
    logprobs = torch.log_softmax(logits.float(), dim=-1)

    out = []
    for i, s in enumerate(seqs):
        n = cont_lens[i]
        # token at position t is predicted by the logits at t-1
        idx = torch.tensor(s[len(s) - n:], device=logprobs.device)
        pos = torch.arange(len(s) - n - 1, len(s) - 1, device=logprobs.device)
        out.append((logprobs[i, pos, idx].sum().item(), n))
    return out


def evaluate(model, tok, items: List[Dict[str, Any]], device) -> List[Dict[str, Any]]:
    records = []
    for item in items:
        by_letter = item["options_by_letter"]
        ordered = [by_letter[L] for L in LETTERS]

        # --- scorer 1: the option text, as a cloze continuation
        conts = [f" {o}" for o in ordered]
        text_scores = continuation_logprobs(
            model, tok, f"Question: {item['question']}\nAnswer:", conts, device)
        sum_lp = [s for s, _ in text_scores]                                    # lm-eval "acc"
        char_lp = [s / max(len(c), 1) for (s, _), c in zip(text_scores, conts)]  # lm-eval "acc_norm"
        tok_lp = [s / max(n, 1) for s, n in text_scores]                        # per-token

        # --- scorer 2: EMBER's own prompt, letter log-likelihood instead of generation
        letter_scores = continuation_logprobs(
            model, tok, build_mc_prompt(item["question"], by_letter),
            [f" {L}" for L in LETTERS], device)
        letter_lp = [s for s, _ in letter_scores]

        pick = lambda xs: LETTERS[max(range(4), key=lambda i: xs[i])]  # noqa: E731
        records.append({
            "concept": item["concept"], "subset": item["subset"], "split": item["split"],
            "question": item["question"], "correct_letter": item["correct_letter"],
            "text_sum": pick(sum_lp), "text_char": pick(char_lp),
            "text_tok": pick(tok_lp), "letter": pick(letter_lp),
            "scores": {"text_sum": sum_lp, "text_char": char_lp,
                       "text_tok": tok_lp, "letter": letter_lp},
        })
    return records


METRICS = ("text_sum", "text_char", "text_tok", "letter")


def summarize(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    by: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        by[(r["concept"], f"{r['subset']}_{r['split']}")].append(r)

    out: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(dict)
    for (concept, split), rs in sorted(by.items()):
        n = len(rs)
        entry: Dict[str, Any] = {"n": n}
        for m in METRICS:
            entry[m] = sum(r[m] == r["correct_letter"] for r in rs) / n
            # How often each letter is chosen. A scorer that answers "A" almost
            # every time is reporting a position bias, not knowledge -- base
            # models are prone to it on the letter prompt, and the accuracy
            # column alone cannot tell the two apart.
            entry[f"{m}_letters"] = {L: sum(r[m] == L for r in rs) / n for L in LETTERS}
        # the gold letters are shuffled per question, so this should be near-uniform
        entry["gold_letters"] = {L: sum(r["correct_letter"] == L for r in rs) / n for L in LETTERS}
        out[concept][split] = entry
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="dhgottesman/LMEnt-1B-1E")
    p.add_argument("--subfolder", default="step109672", help="checkpoint subfolder in the HF repo")
    p.add_argument("--ember-data", default="/home/dcor/galbarak2/EMBER/data")
    p.add_argument("--concepts", nargs="*", default=None, help="default: every concept")
    p.add_argument("--sets", nargs="*", default=["qa_test", "simdom_test"])
    p.add_argument("--limit", type=int, default=0, help="first N items per set; 0 = all")
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", default=None, help="write per-question records here as JSON")
    args = p.parse_args()

    items: List[Dict[str, Any]] = []
    for set_name in args.sets:
        got = prepare_items(load_mc_items(args.ember_data, set_name, args.concepts))
        items += got[:args.limit] if args.limit else got
    print(f"[eval] {len(items)} questions "
          f"({', '.join(args.sets)}; concepts: {args.concepts or 'all'})", flush=True)

    print(f"[eval] loading {args.model}/{args.subfolder} as {args.dtype} on {args.device}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, subfolder=args.subfolder)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, subfolder=args.subfolder, torch_dtype=getattr(torch, args.dtype))
    model.to(args.device).eval()

    records = evaluate(model, tok, items, args.device)
    summary = summarize(records)

    width = max(max(len(c) for c in summary), 7)
    hdr = (f"\n{'concept':<{width}}  {'split':<14} {'n':>4} "
           f"{'text_sum':>9} {'text_char':>10} {'text_tok':>9} {'letter':>7}")
    print(hdr)
    print("-" * (len(hdr) - 1))
    for concept, splits in summary.items():
        for split, m in splits.items():
            print(f"{concept:<{width}}  {split:<14} {m['n']:>4} "
                  f"{m['text_sum']:>8.1%} {m['text_char']:>9.1%} "
                  f"{m['text_tok']:>8.1%} {m['letter']:>6.1%}")
    print("\nchance is 25.0% for every column")

    print("\npredicted-letter distribution (a near-uniform gold row means the "
          "shuffle worked;\na scorer stuck on one letter is showing position bias, not knowledge)")
    for concept, splits in summary.items():
        for split, m in splits.items():
            print(f"  {concept} / {split}")
            print(f"    {'gold':<10} " + "  ".join(f"{L}={m['gold_letters'][L]:5.1%}" for L in LETTERS))
            for metric in METRICS:
                dist = m[f"{metric}_letters"]
                print(f"    {metric:<10} " + "  ".join(f"{L}={dist[L]:5.1%}" for L in LETTERS))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"model": f"{args.model}/{args.subfolder}",
                       "summary": summary, "records": records}, f, indent=2)
        print(f"[eval] wrote {args.out}")


if __name__ == "__main__":
    main()
