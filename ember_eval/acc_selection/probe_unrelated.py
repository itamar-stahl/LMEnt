#!/usr/bin/env python
"""Compare candidate 'unrelated' sets on the FULL model, same scorer for all.

The unrelated set exists to normalise preservation: Acc~ = (Acc - 0.25)/(Acc_F - 0.25).
That denominator is only usable if the full model is meaningfully above chance.
Measured on the frozen sets it is not: Rome 0.280 and Baseball 0.340, both with a
95% CI that includes 0.25. This scores the alternatives so the choice is made on
numbers rather than intuition.

Candidates:
  current_<topic>  the frozen unrelated_selection set, as a baseline
  ww2              World War II QA_train -- the best-scoring concept at n=50 in
                   the 18-concept sweep (job 770277), using its declarative stems
  sciq             OLMES sciq, 4 options
  arc_easy         OLMES ARC-Easy, items with exactly 4 choices

Every set is scored twice, because prompt format is known to move these models by
up to 10 points and the effect is concept-specific:
  stem   continuation of the declarative stem / bare question  (our protocol)
  qa     continuation of "Question: ...\\nAnswer:"             (causal_mc.py)

Chance is 0.25 for all of them: four options everywhere, no 2-option tasks.
"""
from __future__ import annotations

import argparse, csv, json, math, random, sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "Ember-on-LMEnt"))
from ember.evals.causal_mc import continuation_logprobs  # noqa: E402

N = 50
SEED = 20260923


def wilson(k, n, z=1.96):
    if not n: return (0.0, 0.0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def build_sets(manifest: Path, questions: Path):
    rng = random.Random(SEED)
    out = {}

    # the frozen sets, as baselines
    rows = list(csv.DictReader(open(manifest, encoding="utf-8")))
    for t in ("rome", "baseball", "ai"):
        items = [r for r in rows if r["topic"] == t and r["question_group"] == "unrelated_selection"]
        out[f"current_{t}"] = [{"context": r["stem"], "question": r["question"],
                                "options": [r[f"option_{i}"] for i in range(4)],
                                "gold": int(r["gold_index"])} for r in items]

    # World War II, declarative stems, from the same frozen question bank
    q = json.loads(questions.read_text(encoding="utf-8"))
    ww2 = q["World War II"]["QA_train"]
    out["ww2"] = [{"context": r["stem"].strip(), "question": r["q"].strip(),
                   "options": [o.strip() for o in r["options"]],
                   "gold": [o.strip() for o in r["options"]].index(r["correct_answer"].strip())}
                  for r in ww2][:N]

    from datasets import load_dataset
    sq = load_dataset("sciq", split="validation")
    idx = list(range(len(sq))); rng.shuffle(idx)
    sciq = []
    for i in idx[:N]:
        r = sq[i]
        opts = [r["correct_answer"], r["distractor1"], r["distractor2"], r["distractor3"]]
        opts = [str(o).strip() for o in opts]
        if len(set(opts)) != 4: continue
        order = list(range(4)); rng.shuffle(order)
        shuffled = [opts[j] for j in order]
        sciq.append({"context": r["question"].strip(), "question": r["question"].strip(),
                     "options": shuffled, "gold": shuffled.index(opts[0])})
    out["sciq"] = sciq[:N]

    ae = load_dataset("allenai/ai2_arc", "ARC-Easy", split="validation")
    idx = list(range(len(ae))); rng.shuffle(idx)
    arc = []
    for i in idx:
        r = ae[i]
        texts = [str(t).strip() for t in r["choices"]["text"]]
        labels = list(r["choices"]["label"])
        if len(texts) != 4 or r["answerKey"] not in labels: continue
        arc.append({"context": r["question"].strip(), "question": r["question"].strip(),
                    "options": texts, "gold": labels.index(r["answerKey"])})
        if len(arc) == N: break
    out["arc_easy"] = arc
    return out


def score_set(model, tok, items, device, fmt):
    correct = 0
    for it in items:
        ctx = it["context"] if fmt == "stem" else f"Question: {it['question']}\nAnswer:"
        conts = [" " + o for o in it["options"]]
        scored = continuation_logprobs(model, tok, ctx, conts, device=device)
        per_char = [s / max(len(c), 1) for (s, _), c in zip(scored, conts)]
        if max(range(len(conts)), key=per_char.__getitem__) == it["gold"]:
            correct += 1
    return correct, len(items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    sets = build_sets(Path(a.manifest), Path(a.questions))
    for k, v in sets.items():
        print(f"  built {k:16s} n={len(v)}", flush=True)

    torch.backends.cuda.matmul.allow_tf32 = False
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float32).to(a.device).eval()
    print("model loaded", flush=True)

    rows = []
    for name, items in sets.items():
        for fmt in ("stem", "qa"):
            k, n = score_set(model, tok, items, a.device, fmt)
            lo, hi = wilson(k, n)
            rows.append({"set": name, "format": fmt, "n": n, "correct": k,
                         "accuracy": round(k / n, 4),
                         "ci95_lo": round(lo, 4), "ci95_hi": round(hi, 4),
                         "denominator": round(k / n - 0.25, 4),
                         "above_chance_at_95": "yes" if lo > 0.25 else "NO"})
            print(f"  {name:16s} {fmt:5s} {k:2d}/{n}  acc={k/n:.3f} "
                  f"CI=[{lo:.3f},{hi:.3f}] denom={k/n-0.25:+.3f} "
                  f"{'ABOVE CHANCE' if lo > 0.25 else 'ci includes chance'}", flush=True)
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows: w.writerow(r)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
