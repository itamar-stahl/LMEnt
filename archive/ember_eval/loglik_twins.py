#!/usr/bin/env python
"""Compare the twins on continuous log-likelihoods rather than right/wrong.

`paired_twins.py` collapses each question to a bit: did the argmax land on the
correct option. That throws away almost everything the scorer computed. A
question the control gets right by 0.02 nats and the ablated twin gets wrong by
0.02 nats counts as a full unit of evidence; a question where the ablated twin's
confidence in the truth collapsed by 3 nats but both still picked it counts as
nothing. With only 200 questions per concept, discarding the magnitudes is what
leaves the McNemar table too coarse to resolve a 0.024%-of-corpus ablation.

Every continuous value is already in the results JSON, so this is a reanalysis:
no GPU, no re-inference. Three statistics, each a per-question paired difference
(ablated minus control -- negative means ablation hurt):

  logp    log P(correct option) after a softmax over the four options' summed
          log-likelihoods. The smooth version of accuracy: its argmax is exactly
          what `text_sum` reports. Option length inflates the raw sums, but the
          two twins score the identical option texts, so the confound cancels in
          the difference.

  margin  summed log-likelihood of the correct option minus the best distractor.
          Signed distance from the decision boundary; its *sign* is the bit that
          paired_twins.py tests, so this is the same test with the magnitudes
          put back.

  optlp   token-weighted mean per-token log-likelihood of all four option texts,
          correct and distractors alike. This one never looks at the answer key:
          it asks only how surprising the model finds text about the concept.
          It spends ~4x more tokens per question than the other two, and it is
          the statistic an ablation should move most directly.

The inference has two levels, because the ablated twin drifts slightly worse
*everywhere* -- a global offset, not an effect of the ablation:

  within concept   sign-flip permutation over that concept's paired differences.
                   Answers "did this concept move", drift included.

  between concept  the seventeen untouched concepts are the null distribution of
                   concept-level drift. Where the target sits in that spread is
                   the only question that matters. With eighteen concepts the
                   rank-based p bottoms out at 1/18 = 0.056, so it can suggest an
                   effect but can never clear 0.05 on its own -- worth knowing
                   before reading anything into the number.

Each concept's questions also split into EMBER's own two subsets, and they carry
opposite predictions, which is what makes the split worth reporting separately
rather than pooling it away:

  QA        asks about the concept itself -- efficacy. Should move.
  SimdomQA  asks about the neighbouring domain (for Pornography: films, ratings,
            Hollywood) -- specificity. Should not move.

So QA minus SimdomQA, within a concept, is the sharpest number here: it subtracts
that concept's own share of the global drift, since both subsets carry it
equally, and leaves only what is specific to the concept.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import random
import statistics
from collections import defaultdict
from typing import Any, Callable, Dict, List, Sequence, Tuple

TARGET = "Pornography"
LETTERS = ["A", "B", "C", "D"]


# --------------------------------------------------------------------------- #
# per-question statistics
# --------------------------------------------------------------------------- #
def _logsumexp(xs: Sequence[float]) -> float:
    m = max(xs)
    return m + math.log(sum(math.exp(x - m) for x in xs))


# Which stored column the option scores come from. text_sum is the raw summed
# log-likelihood and is dominated by option length; on this corpus the control
# clears chance by only 1.2 SE under it, against 6.0 SE under the
# length-normalized text_char. Analysing text_sum therefore measures the removal
# of knowledge from a column that cannot see the knowledge. Default accordingly.
COLUMN = "text_char"


def stat_logp(rec: Dict[str, Any]) -> float:
    s = rec["scores"][COLUMN]
    i = LETTERS.index(rec["correct_letter"])
    return s[i] - _logsumexp(s)


def stat_margin(rec: Dict[str, Any]) -> float:
    s = rec["scores"][COLUMN]
    i = LETTERS.index(rec["correct_letter"])
    return s[i] - max(s[j] for j in range(4) if j != i)


def stat_optlp(rec: Dict[str, Any]) -> float:
    # text_tok is text_sum / n_tokens, so the ratio recovers the token count and
    # lets the four options be pooled by token rather than by option -- a
    # one-token option should not weigh as much as a twelve-token one.
    total_lp, total_tok = 0.0, 0
    for lp_sum, lp_tok in zip(rec["scores"]["text_sum"], rec["scores"]["text_tok"]):
        n = max(1, round(lp_sum / lp_tok)) if lp_tok else 1
        total_lp += lp_sum
        total_tok += n
    return total_lp / total_tok


STATS: Dict[str, Callable[[Dict[str, Any]], float]] = {
    "logp": stat_logp,
    "margin": stat_margin,
    "optlp": stat_optlp,
}


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #
def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the incomplete beta function (Lentz's method)."""
    tiny = 1e-30
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = tiny if abs(d) < tiny else d
    d = 1.0 / d
    h = d
    for m in range(1, 200):
        m2 = 2 * m
        for num in (m * (b - m) * x / ((qam + m2) * (a + m2)),
                    -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))):
            d = 1.0 + num * d
            d = tiny if abs(d) < tiny else d
            c = 1.0 + num / c
            c = tiny if abs(c) < tiny else c
            d = 1.0 / d
            h *= d * c
        if abs(d * c - 1.0) < 3e-12:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lb = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
          + a * math.log(x) + b * math.log1p(-x))
    front = math.exp(lb)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_cdf(t: float, df: int) -> float:
    """P(T <= t) for Student's t with df degrees of freedom."""
    tail = 0.5 * _betai(df / 2.0, 0.5, df / (df + t * t))
    return tail if t <= 0 else 1.0 - tail


def signflip_p(diffs: Sequence[float], draws: int, rng: random.Random) -> float:
    """Two-sided p for mean(diffs) != 0 under random sign flips of each pair."""
    if not diffs:
        return 1.0
    obs = abs(statistics.fmean(diffs))
    hits = 0
    n = len(diffs)
    for _ in range(draws):
        total = 0.0
        for d in diffs:
            total += d if rng.random() < 0.5 else -d
        if abs(total) / n >= obs - 1e-12:
            hits += 1
    return (hits + 1) / (draws + 1)


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
    ap.add_argument("--tag", default="twins4", help="filename prefix to compare")
    ap.add_argument("--draws", type=int, default=20000, help="sign-flip permutation draws")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--metric", default="text_char",
                    choices=["text_sum", "text_char", "text_tok"],
                    help="which stored option-score column to analyse (default text_char)")
    args = ap.parse_args()

    global COLUMN
    COLUMN = args.metric
    rng = random.Random(args.seed)
    ctl = load(f"{args.results_dir}/{args.tag}_control_*.json")
    abl = load(f"{args.results_dir}/{args.tag}_noporn_*.json")

    def key(r: Dict[str, Any]) -> Tuple[str, str, str, str]:
        return (r["concept"], r["subset"], r["split"], r["question"])

    c_by = {key(r): r for r in ctl["records"]}
    a_by = {key(r): r for r in abl["records"]}
    shared = sorted(set(c_by) & set(a_by))
    print(f"control  {ctl['model']}")
    print(f"ablated  {abl['model']}")
    print(f"paired on {len(shared)} questions "
          f"(control wrote {len(ctl['records'])} records, ablated {len(abl['records'])}; "
          f"duplicates collapse by question text)\n")

    # per-question paired differences, ablated minus control
    diffs: Dict[str, Dict[Tuple[str, str, str], List[float]]] = {
        name: defaultdict(list) for name in STATS
    }
    for k in shared:
        cr, ar = c_by[k], a_by[k]
        cell = (k[0], k[1], k[2])       # concept, subset, split
        for name, fn in STATS.items():
            diffs[name][cell].append(fn(ar) - fn(cr))

    concepts = sorted({c for c, _, _ in diffs["logp"]})

    for name in STATS:
        by_cell = diffs[name]
        all_d = [d for ds in by_cell.values() for d in ds]
        by_concept = {c: [d for (cc, _, _), ds in by_cell.items() if cc == c for d in ds]
                      for c in concepts}

        print("=" * 78)
        print(f"{name}   (ablated - control, in nats; negative = ablation hurt)")
        print("=" * 78)
        print(f"global drift over all {len(all_d)} questions: "
              f"mean {statistics.fmean(all_d):+.4f}, "
              f"sd {statistics.pstdev(all_d):.4f}\n")

        # Per concept: the pooled mean, then the two subsets and their
        # difference. QA - SimdomQA is drift-free by construction.
        def mean_of(c: str, subset: str | None) -> float:
            ds = [d for (cc, sub, _), dd in by_cell.items() if cc == c
                  and (subset is None or sub == subset) for d in dd]
            return statistics.fmean(ds) if ds else float("nan")

        cols = ["pooled", "QA", "SimdomQA", "QA-Simdom"]
        table = {}
        for c in concepts:
            qa, sim = mean_of(c, "QA"), mean_of(c, "SimdomQA")
            table[c] = {"pooled": mean_of(c, None), "QA": qa,
                        "SimdomQA": sim, "QA-Simdom": qa - sim}

        hdr = f"{'concept':<24}{'n':>5}" + "".join(f"{c:>11}" for c in cols) + f"{'dz':>8}{'p':>8}"
        print(hdr)
        print("-" * len(hdr))
        for c in concepts:
            ds = by_concept[c]
            sd = statistics.pstdev(ds)
            dz = statistics.fmean(ds) / sd if sd else 0.0
            p = signflip_p(ds, args.draws, rng)
            mark = "  <-- ablated" if c == TARGET else ""
            print(f"{c:<24}{len(ds):>5}"
                  + "".join(f"{table[c][col]:>+11.4f}" for col in cols)
                  + f"{dz:>+8.3f}{p:>8.3f}{mark}")
        print("\n(dz and p are for the pooled column: effect size and a two-sided\n"
              " sign-flip permutation over that concept's paired differences)")

        # --- the between-concept test: is the target unusual among the seventeen?
        print(f"\nwhere {TARGET} sits among the {len(concepts)} concepts "
              f"(one-sided: ablation predicts a drop)")
        sub_hdr = (f"  {'column':<12}{'target':>10}{'null mean':>11}{'null sd':>9}"
                   f"{'rank':>8}{'rank p':>8}{'t':>8}{'t p':>8}")
        print(sub_hdr)
        print("  " + "-" * (len(sub_hdr) - 2))
        for col in cols:
            tgt = table[TARGET][col]
            nulls = [table[c][col] for c in concepts if c != TARGET]
            nm, nsd = statistics.fmean(nulls), statistics.pstdev(nulls)
            n = len(nulls)
            # sample sd of the seventeen, and the prediction-interval form: we
            # are asking whether one *new* observation came from their
            # population, so the variance carries an extra 1/n term.
            ssd = nsd * math.sqrt(n / (n - 1)) if n > 1 else 0.0
            se = ssd * math.sqrt(1.0 + 1.0 / n)
            t = (tgt - nm) / se if se else 0.0
            rank = 1 + sum(1 for m in nulls if m < tgt)   # 1 = most negative of all
            print(f"  {col:<12}{tgt:>+10.4f}{nm:>+11.4f}{ssd:>9.4f}"
                  f"{rank:>5} /{len(concepts):<2}{rank / len(concepts):>8.3f}"
                  f"{t:>+8.2f}{t_cdf(t, n - 1):>8.3f}")
        print(f"  rank p cannot go below {1 / len(concepts):.3f} -- there are only "
              f"{len(concepts)} concepts to rank, so the rank test is floored\n"
              f"  well above 0.05 no matter how large the effect is. The t column is\n"
              f"  the same comparison without that ceiling, at the cost of assuming\n"
              f"  the concept-level drift is roughly normal across the untouched 17.")

        # --- the target's QA-SimdomQA gap, tested within the concept
        qa_d = [d for (cc, sub, _), dd in by_cell.items() if cc == TARGET and sub == "QA" for d in dd]
        sim_d = [d for (cc, sub, _), dd in by_cell.items() if cc == TARGET and sub == "SimdomQA" for d in dd]
        pooled_d = qa_d + sim_d
        obs = statistics.fmean(qa_d) - statistics.fmean(sim_d)
        hits = 0
        for _ in range(args.draws):
            shuf = pooled_d[:]
            rng.shuffle(shuf)
            if abs(statistics.fmean(shuf[:len(qa_d)]) - statistics.fmean(shuf[len(qa_d):])) >= abs(obs) - 1e-12:
                hits += 1
        print(f"\n  {TARGET} QA ({len(qa_d)} questions) vs SimdomQA ({len(sim_d)}): "
              f"gap {obs:+.4f}, label-permutation p = {(hits + 1) / (args.draws + 1):.3f}")

        # A mean can be one bad question. These say whether the QA column is a
        # shift of the whole distribution or a few outliers, and whether it
        # replicates across EMBER's two disjoint question sets.
        print(f"\nis the QA column robust? (rank of {TARGET} among the "
              f"{len(concepts)} concepts, 1 = most negative)")
        rob_hdr = (f"  {'concept':<24}{'mean':>9}{'median':>9}{'trim20':>9}"
                   f"{'frac<0':>8}{'QA_train':>10}{'QA_test':>9}")
        print(rob_hdr)
        print("  " + "-" * (len(rob_hdr) - 2))
        rob: Dict[str, List[float]] = {}
        for c in concepts:
            ds = sorted(d for (cc, sub, _), dd in by_cell.items()
                        if cc == c and sub == "QA" for d in dd)
            k = len(ds) // 10                     # 20% trimmed: drop a tenth each end
            tr = ds[k:len(ds) - k] or ds
            halves = [statistics.fmean(dd) for sp in ("train", "test")
                      for (cc, sub, s2), dd in by_cell.items()
                      if cc == c and sub == "QA" and s2 == sp]
            rob[c] = [statistics.fmean(ds), statistics.median(ds), statistics.fmean(tr),
                      sum(1 for x in ds if x < 0) / len(ds)] + halves
            if c == TARGET:
                print(f"  {c:<24}" + "".join(f"{v:>+9.3f}" for v in rob[c][:3])
                      + f"{rob[c][3]:>8.2f}" + "".join(f"{v:>+10.3f}" for v in rob[c][4:])
                      + "  <-- ablated")
        ranks = {}
        for j, name in enumerate(("mean", "median", "trim20", "frac<0", "QA_train", "QA_test")):
            worse = sum(1 for c in concepts if c != TARGET and
                        (rob[c][j] < rob[TARGET][j] if name != "frac<0"
                         else rob[c][j] > rob[TARGET][j]))
            ranks[name] = worse + 1
        print("  " + "  ".join(f"{k}: {v}/{len(concepts)}" for k, v in ranks.items()))
        print()


if __name__ == "__main__":
    main()
