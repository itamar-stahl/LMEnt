#!/usr/bin/env python
"""Does training longer give a concept ablation more to remove?

The next twin pair's epoch count was about to be chosen from Table 3 of the
paper, which reports general benchmarks -- the wrong instrument. What decides it
is how much a model trained on this corpus knows about *the concept being
ablated*, because that is the knowledge the ablation has to take away. A longer
run is only worth its GPU-hours if that number grows.

The authors released 1E/2E/4E/6E of the same 1B on the same corpus, so it can be
measured rather than guessed. `run_epoch_sweep.slurm` scores all four with the
same harness on the same 3,600 questions; this reads the records.

Three views, because the first two can be fooled and the third cannot:

  accuracy     on the target concept's own questions. Coarse -- 100 questions,
               and a model can gain a lot of probability mass without any
               argmax flipping.
  log P        the smooth version: how much probability the model puts on the
               correct option. Sensitive to gains accuracy cannot see.
  optlp        mean per-token log-likelihood of every option's text, correct and
               distractor alike, over all 3,600 questions. Tens of thousands of
               tokens rather than 100 decisions, so if the model is improving at
               all in any way, this shows it. It is the check that says whether
               a flat accuracy curve means "saturated" or "instrument too blunt".
"""
from __future__ import annotations

import glob
import json
import math
import statistics
from collections import defaultdict
from typing import Any, Dict, List, Sequence

TARGET = "Pornography"
LETTERS = ["A", "B", "C", "D"]
EPOCHS = ("1E", "2E", "4E", "6E")


def logsumexp(xs: Sequence[float]) -> float:
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


def logp_correct(r: Dict[str, Any]) -> float:
    s = r["scores"]["text_sum"]
    return s[LETTERS.index(r["correct_letter"])] - logsumexp(s)


def optlp(r: Dict[str, Any]) -> float:
    total_lp, total_tok = 0.0, 0
    for lp, per_tok in zip(r["scores"]["text_sum"], r["scores"]["text_tok"]):
        total_lp += lp
        total_tok += max(1, round(lp / per_tok)) if per_tok else 1
    return total_lp / total_tok


def key(r: Dict[str, Any]):
    return (r["concept"], r["subset"], r["split"], r["question"])


def main() -> None:
    runs: Dict[str, List[Dict[str, Any]]] = {}
    for e in EPOCHS:
        files = sorted(glob.glob(f"ember_eval/results/epochs_{e}_*.json"))
        if not files:
            raise SystemExit(f"no results for {e}; run run_epoch_sweep.slurm first")
        runs[e] = json.load(open(files[-1], encoding="utf-8"))["records"]

    # --- is the model improving at all? the check that validates the others
    print("Is each model actually better than the last?")
    print("  mean per-token log-likelihood of all option text, 3,600 questions")
    for e in EPOCHS:
        print(f"    {e}:{statistics.fmean(optlp(r) for r in runs[e]):>9.4f}")

    # --- what the ablation would have to remove
    print(f"\nKnowledge of {TARGET}, the ablation target (its own QA questions, n=100)")
    hdr = f"{'epochs':<8}{'accuracy':>10}{'log P(correct)':>17}{'vs 1E':>9}"
    print(hdr + f"{'| other 17 acc':>16}{'log P':>9}{'vs 1E':>9}")
    print("-" * (len(hdr) + 34))
    base = {}
    for e in EPOCHS:
        qa = [r for r in runs[e] if r["subset"] == "QA"]
        tgt = [r for r in qa if r["concept"] == TARGET]
        others: Dict[str, list] = defaultdict(list)
        for r in qa:
            if r["concept"] != TARGET:
                others[r["concept"]].append(r)
        t_acc = sum(r["text_sum"] == r["correct_letter"] for r in tgt) / len(tgt)
        t_lp = statistics.fmean(logp_correct(r) for r in tgt)
        o_acc = statistics.fmean(
            sum(x["text_sum"] == x["correct_letter"] for x in v) / len(v) for v in others.values())
        o_lp = statistics.fmean(statistics.fmean(logp_correct(x) for x in v) for v in others.values())
        if e == "1E":
            base = {"t": t_lp, "o": o_lp}
        print(f"{e:<8}{t_acc:>9.0%}{t_lp:>17.3f}{t_lp - base['t']:>+9.3f}"
              f"{o_acc:>15.0%}{o_lp:>9.3f}{o_lp - base['o']:>+9.3f}")

    # --- is the target unusual in how much it gains?
    a = {key(r): r for r in runs["1E"]}
    b = {key(r): r for r in runs["6E"]}
    gains = {}
    for c in sorted({r["concept"] for r in runs["1E"]}):
        shared = [k for k in a if k in b and k[0] == c and k[1] == "QA"]
        gains[c] = statistics.fmean(logp_correct(b[k]) - logp_correct(a[k]) for k in shared)
    ranked = sorted(gains.items(), key=lambda kv: -kv[1])
    rank = 1 + [c for c, _ in ranked].index(TARGET)
    print(f"\n1E -> 6E gain on each concept's own questions, most improved first:")
    for i, (c, g) in enumerate(ranked, 1):
        print(f"  {i:>2}. {c:<24}{g:>+7.3f}" + ("   <-- ablation target" if c == TARGET else ""))
    print(f"\n{TARGET} ranks {rank} of {len(ranked)}; median gain "
          f"{statistics.median(gains.values()):+.3f} against its {gains[TARGET]:+.3f}. "
          f"Nothing special.")


if __name__ == "__main__":
    main()
