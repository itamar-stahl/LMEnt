#!/usr/bin/env python
"""Score a model on the sentence-completion rewrite of EMBER's questions.

Nothing is generated and nothing is parsed. Each of the four options is scored
as a continuation of its stem, and the score is

    pmi(a) = log P(a | stem) - log P(a | null)

The second term is the same option scored with no stem in front of it. It
cancels how probable the option string is on its own, so an option stops
winning for being the commoner phrase and the score measures whether the model
links this fact to this context. Subtracting an option's unconditional
likelihood this way is standard practice, introduced in Brown et al. (2020) and
analysed in Holtzman et al. (2021).

The four scores are read two ways:

    accuracy       the highest-scoring option, marked against the answer key
    soft_accuracy  mean over questions of softmax(pmi)[gold]

Soft accuracy is not a second metric. It is the same four numbers read as a
distribution rather than an argmax, and its argmax is exactly the accuracy
beside it. Chance is 0.25 for both. It is worth reporting because collapsing
each question to a right/wrong bit throws away the magnitudes, and with 200
questions per concept an ablation worth a few points cannot be resolved from
what is left.

Option order, the per-question shuffle and the item ids match
`score_ember_mc.py` exactly, so item N here is item N in every multiple-choice
run already recorded.

Examples
--------
python evaluate_completion.py --model /path/to/lment-1b-control \\
    --questions data/completion_questions.json \\
    --concept Pornography --split QA_train --out results/control_qa_train.json

python evaluate_completion.py --model dhgottesman/LMEnt-1B-1E \\
    --subfolder step109672 --concept "Harry Potter" --split QA_test \\
    --out results/released1e_hp_qa_test.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

LETTERS = ["A", "B", "C", "D"]


# --------------------------------------------------------------------------- #
# loading and EMBER-compatible shuffling
# --------------------------------------------------------------------------- #
def stable_item_id(item: Dict[str, Any]) -> str:
    """Identical to ember.evals.mc.stable_item_id, via score_ember_mc.py."""
    parts = [f"{k}={item[k]}" for k in
             ("concept", "subset", "split", "question", "correct_answer")
             if item.get(k) is not None]
    parts.append("options=" + "|".join(list(item["options"])))
    return hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()


def load_items(path: str, concept: str, split: str, seed: int = 42) -> List[Dict[str, Any]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if concept not in data:
        raise SystemExit(f"{concept!r} not found in {path}")
    if split not in data[concept]:
        raise SystemExit(f"split {split!r} not found for {concept!r}")
    subset, sample_split = split.rsplit("_", 1)

    out = []
    for raw in data[concept][split]:
        if "stem" not in raw:
            raise SystemExit(f"item has no stem: {raw.get('q')!r}")
        item = {
            "concept": concept, "subset": subset, "split": sample_split,
            "question": raw["q"].strip(),
            "options": [o.strip() for o in raw["options"]],
            "correct_answer": raw["correct_answer"].strip(),
            "stem": raw["stem"].strip(),
        }
        options = list(item["options"])
        sid = stable_item_id(item)
        per_q = (seed + int(hashlib.sha1(sid.encode("utf-8")).hexdigest()[:8], 16)) % (2 ** 32)
        random.Random(per_q).shuffle(options)
        if item["correct_answer"] not in options[:4]:
            raise SystemExit(f"correct answer missing from options: {item['question']!r}")
        item["shuffled"] = options[:4]
        item["correct_index"] = options.index(item["correct_answer"])
        item["correct_letter"] = LETTERS[item["correct_index"]]
        out.append(item)
    return out


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def context_ids(tok, text: str) -> List[int]:
    """Token ids for a context, never empty.

    The null context is the empty string, and an empty prefix leaves nothing to
    condition on and no position to read a prediction from. Falling back to BOS
    (or EOS, which is what OLMo separates documents with) makes
    `log P(a | null)` the probability of the option at the start of a document,
    which is the marginal being divided out.
    """
    ids = tok(text, add_special_tokens=True)["input_ids"] if text else []
    if ids:
        return ids
    for tid in (tok.bos_token_id, tok.eos_token_id):
        if tid is not None:
            return [tid]
    raise SystemExit("tokenizer has neither a BOS nor an EOS token; pass a "
                     "non-empty --null-context, for example 'Answer:'")


@torch.no_grad()
def score_pairs(model, tok, pairs: Sequence[Tuple[str, str]], device,
                batch_size: int = 16, max_length: int = 2048
                ) -> Dict[Tuple[str, str], Tuple[float, int]]:
    """{(context, continuation): (summed log P(continuation | context), n_tokens)}.

    Right padding, causal attention: positions before a pad cannot see it, so a
    padded batch scores the same as one sequence at a time. Duplicate pairs are
    scored once, which matters because 26 of the 50 similar-domain test
    questions also appear in the validation split.
    """
    todo = sorted({p for p in pairs})
    out: Dict[Tuple[str, str], Tuple[float, int]] = {}
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

    encoded = []
    for ctx, cont in todo:
        c_ids = context_ids(tok, ctx)
        k_ids = tok(cont, add_special_tokens=False)["input_ids"]
        if not k_ids:
            raise SystemExit(f"continuation tokenized to nothing: {cont!r}")
        seq = c_ids + k_ids
        if len(seq) > max_length:
            seq = seq[-max_length:]
        encoded.append((seq, len(k_ids)))

    order = sorted(range(len(encoded)), key=lambda i: len(encoded[i][0]))
    for start in range(0, len(order), batch_size):
        chunk = order[start:start + batch_size]
        width = max(len(encoded[i][0]) for i in chunk)
        input_ids = torch.full((len(chunk), width), pad_id, dtype=torch.long)
        attn = torch.zeros((len(chunk), width), dtype=torch.long)
        for row, i in enumerate(chunk):
            seq = encoded[i][0]
            input_ids[row, :len(seq)] = torch.tensor(seq, dtype=torch.long)
            attn[row, :len(seq)] = 1

        logits = model(input_ids=input_ids.to(device),
                       attention_mask=attn.to(device)).logits
        logprobs = torch.log_softmax(logits.float(), dim=-1)
        for row, i in enumerate(chunk):
            seq, n = encoded[i]
            # token at position t is predicted by the logits at t - 1
            idx = torch.tensor(seq[len(seq) - n:], device=logprobs.device)
            pos = torch.arange(len(seq) - n - 1, len(seq) - 1, device=logprobs.device)
            out[todo[i]] = (logprobs[row, pos, idx].sum().item(), n)
    return out


def softmax(xs: Sequence[float], temperature: float = 1.0) -> List[float]:
    scaled = [x / temperature for x in xs]
    m = max(scaled)
    exps = [math.exp(x - m) for x in scaled]
    total = sum(exps)
    return [e / total for e in exps]


def build_pairs(items: Sequence[Dict[str, Any]], null_context: str) -> List[Tuple[str, str]]:
    pairs = []
    for item in items:
        for cont in (f" {o}" for o in item["shuffled"]):
            pairs.append((item["stem"], cont))
            pairs.append((null_context, cont))
    return pairs


def evaluate(items: Sequence[Dict[str, Any]], scores, null_context: str,
             temperature: float) -> List[Dict[str, Any]]:
    records = []
    for item in items:
        conts = [f" {o}" for o in item["shuffled"]]
        cond = [scores[(item["stem"], c)] for c in conts]
        null = [scores[(null_context, c)] for c in conts]
        pmi = [c - z for (c, _), (z, _) in zip(cond, null)]

        gold = item["correct_index"]
        probs = softmax(pmi, temperature)
        pred = max(range(4), key=lambda i: pmi[i])
        records.append({
            "concept": item["concept"], "subset": item["subset"], "split": item["split"],
            "question": item["question"], "stem": item["stem"],
            "options": item["shuffled"], "correct_answer": item["correct_answer"],
            "correct_index": gold, "correct_letter": item["correct_letter"],
            "pmi": pmi,
            "logp_conditional": [c for c, _ in cond],
            "logp_null": [z for z, _ in null],
            "pred_index": pred,
            "pred": item["shuffled"][pred],
            "correct": pred == gold,
            "p_correct": probs[gold],
            "confidence": probs[pred],
            "margin": pmi[gold] - max(pmi[i] for i in range(4) if i != gold),
        })
    return records


def summarize(records: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    n = len(records)
    p = [r["p_correct"] for r in records]
    return {
        "accuracy": sum(r["correct"] for r in records) / n,
        "soft_accuracy": sum(p) / n,
        "sum_p_correct": sum(p),
        "mean_logp_correct": sum(math.log(max(x, 1e-12)) for x in p) / n,
        "mean_margin": sum(r["margin"] for r in records) / n,
        "mean_confidence": sum(r["confidence"] for r in records) / n,
        "n_questions": n,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--subfolder", default="", help="checkpoint subfolder for an HF repo")
    ap.add_argument("--questions", default="data/completion_questions.json")
    ap.add_argument("--concept", default="Pornography")
    ap.add_argument("--split", required=True,
                    choices=("QA_train", "QA_test", "SimdomQA_train", "SimdomQA_test"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--null-context", default="",
                    help="the null in log P(a|stem) - log P(a|null). Empty, the "
                         "default, means no context at all. Whatever you choose, "
                         "use the same value for every model you compare")
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="divides the scores before the softmax; raise it if "
                         "mean_confidence saturates near 1.0")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", default="float32", choices=("float32", "bfloat16", "float16"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--cache-dir")
    ap.add_argument("--limit", type=int, default=0, help="first N questions; 0 = all")
    args = ap.parse_args()

    items = load_items(args.questions, args.concept, args.split, args.seed)
    if args.limit:
        items = items[:args.limit]
    print(f"[eval] {len(items)} questions, {args.concept} / {args.split}", flush=True)

    kw = {"subfolder": args.subfolder} if args.subfolder else {}
    if args.cache_dir:
        kw["cache_dir"] = args.cache_dir
    print(f"[eval] loading {args.model} as {args.dtype} on {args.device}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, **kw)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=getattr(torch, args.dtype), **kw).to(args.device).eval()

    pairs = build_pairs(items, args.null_context)
    print(f"[eval] {len(set(pairs))} distinct (context, option) pairs to score", flush=True)
    scores = score_pairs(model, tok, pairs, args.device, args.batch_size, args.max_length)
    records = evaluate(items, scores, args.null_context, args.temperature)
    summary = summarize(records)

    print(f"\n  accuracy        {summary['accuracy']:>7.1%}   (chance 25.0%)")
    print(f"  soft accuracy   {summary['soft_accuracy']:>7.3f}   (chance 0.250)")
    print(f"  mean margin     {summary['mean_margin']:>+7.3f}   gold minus best distractor")
    print(f"  mean confidence {summary['mean_confidence']:>7.3f}")
    if summary["mean_confidence"] > 0.95:
        print("\n  mean confidence is near 1.0: the softmax has saturated and soft\n"
              "  accuracy has collapsed onto accuracy. Raise --temperature.")

    result = {
        **summary,
        "metadata": {
            "model": f"{args.model}/{args.subfolder}" if args.subfolder else args.model,
            "concept": args.concept, "task": args.split, "questions": args.questions,
            "metric": "pmi = log P(answer | stem) - log P(answer | null)",
            "null_context": args.null_context, "temperature": args.temperature,
            "seed": args.seed, "dtype": args.dtype, "batch_size": args.batch_size,
            "max_length": args.max_length,
            "scoring": "completion log-likelihood; nothing generated, nothing parsed",
        },
        "records": records,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[eval] wrote {args.out}")


if __name__ == "__main__":
    main()
