#!/usr/bin/env python
"""Compare our control twin against the authors' released LMEnt-1B-1E.

The two were trained on the same corpus for the same one epoch, but not with the
same recipe: the paper uses a 32,768-token global batch (109,672 steps), ours
uses 131,072 (27,416 steps), and the learning rates differ. See COMPARABILITY.md
on feature/training.

That raises a fair objection to the whole ablation experiment: if a quarter as
many optimizer steps left our twins knowing less about the target concept in the
first place, there was less for the ablation to remove and the null result is
about our recipe rather than about the ablation.

This settles it by measurement. Both models were scored by the same harness on
the same 1,800 questions (QA_test + SimdomQA_test, all 18 concepts) in the same
shuffled option order, so every question is a matched pair. Two things matter:

  overall     does our recipe cost general knowledge across all 18 concepts;
  target      does it cost knowledge of the concept the ablation removes, which
              is the headroom the twin comparison needs in order to see anything.

McNemar for the accuracies, and the mean paired difference in log P(correct) for
the magnitudes -- accuracy alone is too coarse to tell "slightly behind" from
"badly behind", which is exactly the distinction in question.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import statistics
from collections import defaultdict
from typing import Any, Dict, List, Sequence

TARGET = "Pornography"
LETTERS = ["A", "B", "C", "D"]


def logsumexp(xs: Sequence[float]) -> float:
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


def logp_correct(rec: Dict[str, Any]) -> float:
    s = rec["scores"]["text_sum"]
    return s[LETTERS.index(rec["correct_letter"])] - logsumexp(s)


def mcnemar_exact_two_sided(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    pmf = [math.comb(n, k) * 0.5 ** n for k in range(n + 1)]
    obs = pmf[b]
    return min(1.0, sum(p for p in pmf if p <= obs + 1e-12))


def load(pattern: str) -> Dict[str, Any]:
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"no results matching {pattern}")
    return json.load(open(files[-1], encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir",
                    default="/home/dcor/galbarak2/LMEnt-ember/ember_eval/results")
    ap.add_argument("--published", default="ember-eval-lment_*.json",
                    help="sweep of dhgottesman/LMEnt-1B-1E over all 18 concepts")
    ap.add_argument("--ours", default="twins4_control_*.json")
    ap.add_argument("--metric", default="text_sum", choices=["text_sum", "text_char", "text_tok"])
    args = ap.parse_args()

    pub = load(f"{args.results_dir}/{args.published}")
    our = load(f"{args.results_dir}/{args.ours}")

    def key(r):
        return (r["concept"], r["subset"], r["split"], r["question"])

    P = {key(r): r for r in pub["records"]}
    O = {key(r): r for r in our["records"]}
    shared = sorted(set(P) & set(O))
    print(f"published  {pub['model']}")
    print(f"ours       {our['model']}")
    print(f"metric     {args.metric}")
    print(f"paired on {len(shared)} questions "
          f"(published scored {len(P)}, ours {len(O)}; the overlap is what both cover)\n")

    cells: Dict[Any, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for k in shared:
        pr, orr = P[k], O[k]
        pc = pr[args.metric] == pr["correct_letter"]
        oc = orr[args.metric] == orr["correct_letter"]
        s = cells[(k[0], k[1])]
        s["pub"].append(pc)
        s["our"].append(oc)
        s["dlp"].append(logp_correct(orr) - logp_correct(pr))
        if pc and not oc:
            s["b"].append(1)        # published right, ours wrong
        elif oc and not pc:
            s["c"].append(1)

    for subset in sorted({s for _, s in cells}):
        print(f"=== {subset} ===")
        hdr = f"{'concept':<24}{'published':>10}{'ours':>7}{'diff':>7}{'b':>4}{'c':>4}{'p':>7}{'d logP':>9}"
        print(hdr)
        print("-" * len(hdr))
        tot = defaultdict(list)
        for (concept, sub), s in sorted(cells.items()):
            if sub != subset:
                continue
            n = len(s["pub"])
            b, c = len(s["b"]), len(s["c"])
            for k2 in ("pub", "our", "dlp", "b", "c"):
                tot[k2] += s[k2]
            mark = "  <-- ablation target" if concept == TARGET else ""
            print(f"{concept:<24}{sum(s['pub']) / n:>9.0%}{sum(s['our']) / n:>7.0%}"
                  f"{(sum(s['our']) - sum(s['pub'])) / n:>+7.0%}{b:>4}{c:>4}"
                  f"{mcnemar_exact_two_sided(b, c):>7.3f}"
                  f"{statistics.fmean(s['dlp']):>+9.3f}{mark}")
        n = len(tot["pub"])
        b, c = len(tot["b"]), len(tot["c"])
        print("-" * len(hdr))
        print(f"{'ALL 18 CONCEPTS':<24}{sum(tot['pub']) / n:>9.1%}{sum(tot['our']) / n:>7.1%}"
              f"{(sum(tot['our']) - sum(tot['pub'])) / n:>+7.1%}{b:>4}{c:>4}"
              f"{mcnemar_exact_two_sided(b, c):>7.3f}"
              f"{statistics.fmean(tot['dlp']):>+9.3f}\n")

    # The headroom question, stated plainly.
    tgt = [s for (concept, _), s in cells.items() if concept == TARGET]
    n = sum(len(s["pub"]) for s in tgt)
    pub_acc = sum(sum(s["pub"]) for s in tgt) / n
    our_acc = sum(sum(s["our"]) for s in tgt) / n
    dlp = statistics.fmean([d for s in tgt for d in s["dlp"]])
    print(f"headroom on {TARGET}, the concept the ablation removes:")
    print(f"  published {pub_acc:.1%}, ours {our_acc:.1%} over {n} questions, "
          f"mean d logP {dlp:+.3f} nats.")
    print("  If ours were far behind here, the ablation would have had little to")
    print("  remove and the twin comparison would be measuring our recipe instead.")


if __name__ == "__main__":
    main()
