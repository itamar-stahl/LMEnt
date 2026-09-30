#!/usr/bin/env python
"""Paired comparison of the control and ablated twins on EMBER.

Comparing two aggregate accuracies throws away the fact that both models answer
the *identical* questions in the identical shuffled option order -- the harness
reuses EMBER's per-question seed. Pairing recovers that: what matters is not
38% against 32%, but on how many individual questions the twins *disagreed*, and
in which direction.

McNemar's exact test does this. Of the questions where exactly one twin is
right, call b the count where the control is right and the ablated one wrong,
and c the reverse. Under the null that ablation changed nothing, each
disagreement is a coin flip, so b ~ Binomial(b+c, 0.5). Concordant answers carry
no information and are discarded, which is the whole point -- two models that
agree on 90% of questions tell you nothing by agreeing.

The seventeen non-ablated concepts are the null distribution. If Pornography's
asymmetry looks like theirs, there is no detectable effect no matter what the
headline percentages do.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from collections import defaultdict

TARGET = "Pornography"


def mcnemar_exact_two_sided(b: int, c: int) -> float:
    """Exact two-sided p for b successes in b+c fair coin flips."""
    n = b + c
    if n == 0:
        return 1.0
    pmf = [math.comb(n, k) * 0.5 ** n for k in range(n + 1)]
    obs = pmf[b]
    # two-sided: total probability of outcomes no more likely than the observed
    return min(1.0, sum(p for p in pmf if p <= obs + 1e-12))


def load(pattern: str) -> dict:
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no results matching {pattern}")
    return json.load(open(files[-1]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="/home/dcor/galbarak2/LMEnt-ember/ember_eval/results")
    ap.add_argument("--tag", default="twins4", help="filename prefix to compare")
    ap.add_argument("--metric", default="text_sum", choices=["text_sum", "text_char", "text_tok"])
    args = ap.parse_args()

    ctl = load(f"{args.results_dir}/{args.tag}_control_*.json")
    abl = load(f"{args.results_dir}/{args.tag}_noporn_*.json")

    def key(r):
        return (r["concept"], r["subset"], r["split"], r["question"])

    c_by = {key(r): r for r in ctl["records"]}
    a_by = {key(r): r for r in abl["records"]}
    shared = sorted(set(c_by) & set(a_by))
    print(f"paired on {len(shared)} questions "
          f"(control had {len(c_by)}, ablated {len(a_by)}); metric = {args.metric}\n")

    # per concept, pooling train+test within each subset
    per = defaultdict(lambda: defaultdict(int))
    for k in shared:
        cr, ar = c_by[k], a_by[k]
        concept, subset = k[0], k[1]
        cc = cr[args.metric] == cr["correct_letter"]
        ac = ar[args.metric] == ar["correct_letter"]
        s = per[(concept, subset)]
        s["n"] += 1
        s["ctl"] += cc
        s["abl"] += ac
        if cc and not ac:
            s["b"] += 1          # control right, ablated wrong  -> ablation lost knowledge
        elif ac and not cc:
            s["c"] += 1          # ablated right, control wrong

    hdr = f"{'concept':<24}{'subset':<10}{'n':>5}{'ctl':>7}{'abl':>7}{'b':>5}{'c':>5}{'p':>8}"
    print(hdr); print("-" * len(hdr))
    nulls = []
    for (concept, subset), s in sorted(per.items()):
        p = mcnemar_exact_two_sided(s["b"], s["c"])
        mark = "  <-- ablated" if concept == TARGET else ""
        print(f"{concept:<24}{subset:<10}{s['n']:>5}{s['ctl']/s['n']:>7.1%}"
              f"{s['abl']/s['n']:>7.1%}{s['b']:>5}{s['c']:>5}{p:>8.3f}{mark}")
        if concept != TARGET:
            nulls.append(s["b"] - s["c"])

    print()
    tgt = [(sub, s) for (c, sub), s in per.items() if c == TARGET]
    for sub, s in sorted(tgt):
        d = s["b"] - s["c"]
        worse = sum(1 for x in nulls if abs(x) >= abs(d))
        print(f"{TARGET} / {sub}: b-c = {d:+d}. "
              f"{worse} of {len(nulls)} non-ablated concept/subset cells moved at least as far.")
    if nulls:
        import statistics
        print(f"\nnull spread over {len(nulls)} non-ablated cells: "
              f"b-c mean {statistics.mean(nulls):+.1f}, sd {statistics.pstdev(nulls):.1f}, "
              f"range {min(nulls):+d}..{max(nulls):+d}")


if __name__ == "__main__":
    main()
