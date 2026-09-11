#!/usr/bin/env python3
"""How much do these two twins differ on concepts neither of them touched?

`ROME_RESULTS.md` reports a -0.37 / -0.28 paired shift on Ancient Rome and a
flat neighbouring domain. Neither number says whether -0.33 is *large*, because
two separately trained models differ a little on everything and nothing so far
measures that floor. `NULL_CONCEPT_CONTROL.md` is the cautionary case: on Harry
Potter, which neither model ablated, the older twins produced a *bigger* and
more significant difference than the concept they had actually ablated.

So this scores the same statistic on all nine concepts in
`completion_eval/data/completion_questions.json` -- Ancient Rome plus the eight
nobody held out -- and reports where Ancient Rome falls in that distribution.
The eight are the null; Ancient Rome is the observation. That is the same shape
as the Pornography claim (largest of 18 against a null mean of 0.16), which is
the comparison this measurement exists to make honestly.

**The scoring rule is imported, not reimplemented.** `pmi_per_char` on the
`gold` statistic is what produced every number in `ROME_RESULTS.md`, and it
comes from `metric_bakeoff.py` here as well, so the null and the observation
cannot drift apart the way `evaluate_completion.py` and its callers once did
(`lment-eval-metric-split`; `EVALUATION.md` corrections 1-2).

    python cross_concept_null.py [--results DIR] [--json-out FILE]

Reads `cmpl_<model>_<concept-tag>_<subset>_<split>_<jobid>.json`. Note that
`metric_bakeoff.py` cannot be pointed at these files directly: its `FNAME`
regex folds the concept tag into the model group, so `control2e_rome` and
`control2e_hp` read as two different models.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import statistics as st
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "completion_eval"))
from metric_bakeoff import build_rules, dz, perm_p_paired, statistics_for  # noqa: E402

# cmpl_<model>_<concept tag>_<subset>_<split>_<jobid>.json
FNAME = re.compile(r"^cmpl_([A-Za-z0-9]+)_([a-z0-9_]+?)_(QA|SimdomQA)_(train|test)_(\d+)\.json$")

ABLATED_CONCEPT = "Ancient Rome"
CONTROL_MODEL = "control2e"
ABLATED_MODEL = "norome2e"

# `gold` reads the gold option on its own, so no softmax is involved and the
# temperature `statistics_for` takes is inert for it. Fixed at 1.0 to make that
# explicit rather than calibrating a constant nothing here consumes.
TEMPERATURE = 1.0
STAT = "gold"
RULE = "pmi_per_char"


def load(results_dir: str):
    """{(model, concept, subset, split, question): gold-score}, plus provenance."""
    rule = build_rules(None)[RULE]
    scores: dict = {}
    models: dict = defaultdict(set)      # (model, concept) -> {checkpoint path}
    jobs: dict = defaultdict(set)        # (model, concept) -> {job id}
    for path in sorted(glob.glob(os.path.join(results_dir, "cmpl_*.json"))):
        m = FNAME.match(os.path.basename(path))
        if not m:
            continue                     # the pre-2026-09 untagged files
        model, _tag, subset, split, jobid = m.groups()
        blob = json.load(open(path))
        for rec in blob["records"]:
            key = (model, rec["concept"], subset, split, rec["question"])
            if key in scores:
                raise SystemExit(f"{path}: duplicate question {key}")
            scores[key] = statistics_for(rule, TEMPERATURE, rec)[STAT]
        models[(model, rec["concept"])].add(blob["metadata"]["model"])
        jobs[(model, rec["concept"])].add(jobid)
    if not scores:
        raise SystemExit(f"no concept-tagged completion records in {results_dir}")
    return scores, models, jobs


def cell(scores, concept, subset, split):
    """Paired ablated-minus-control deltas for one (concept, subset, split)."""
    ctl = {q: v for (m, c, s, sp, q), v in scores.items()
           if m == CONTROL_MODEL and c == concept and s == subset and sp == split}
    abl = {q: v for (m, c, s, sp, q), v in scores.items()
           if m == ABLATED_MODEL and c == concept and s == subset and sp == split}
    shared = sorted(set(ctl) & set(abl))
    if not shared:
        return None
    return np.array([abl[q] - ctl[q] for q in shared])


def summarise(d: np.ndarray, rng) -> dict:
    """Mean delta, dz, paired t, both t and sign-flip p, and the sign test.

    `p_t` is Student's t on n-1 degrees of freedom. It is reported because
    ROME_RESULTS.md's first draft quoted the **normal** approximation instead:
    t = -5.60 was published as p = 2.2e-08, which is 2*Phi(-5.60), where the t
    distribution on 49 df gives 9.6e-07 -- a factor of 45. At n = 50 the t
    distribution is the correct one. Nothing about the Rome conclusion turns on
    it (both are far past any threshold), but the published number was wrong
    and the corrected table says so.

    `p_signflip` is floored at 1/(iters+1) = 5.0e-05, so it cannot resolve the
    concept cells at all; it is here because it assumes nothing about the shape
    of the differences, which is why the rest of this codebase prefers it.
    """
    n = len(d)
    mean = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    t = mean / (sd / math.sqrt(n)) if sd else 0.0
    p_t = float(2 * stats.t.sf(abs(t), n - 1)) if sd and n > 1 else 1.0
    neg = int((d < 0).sum())
    # two-sided exact sign test on the non-zero signs
    nz = int((d != 0).sum())
    k = min(neg, nz - neg)
    sign_p = min(1.0, 2 * sum(math.comb(nz, i) for i in range(k + 1)) / 2 ** nz) if nz else 1.0
    return {"n": n, "mean": mean, "dz": float(dz(d)), "t": t, "p_t": p_t,
            "p_signflip": float(perm_p_paired(d, rng)),
            "neg": neg, "sign_p": sign_p}


def main() -> None:
    # Declared before any use, or Python rejects the function outright.
    global ABLATED_CONCEPT, CONTROL_MODEL, ABLATED_MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="/home/dcor/galbarak2/LMEnt-ember/"
                                         "ember_eval/results/completion")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--seed", type=int, default=42)
    # Defaults keep the published Ancient Rome table reproducible with no flags.
    ap.add_argument("--concept", default=ABLATED_CONCEPT,
                    help="the ablated concept (default: %(default)s)")
    ap.add_argument("--control-model", default=CONTROL_MODEL,
                    help="control model tag (default: %(default)s)")
    ap.add_argument("--ablated-model", default=ABLATED_MODEL,
                    help="ablated model tag (default: %(default)s)")
    args = ap.parse_args()
    ABLATED_CONCEPT = args.concept
    CONTROL_MODEL = args.control_model
    ABLATED_MODEL = args.ablated_model

    rng = np.random.default_rng(args.seed)
    scores, models, jobs = load(args.results)

    concepts = sorted({c for (_, c, _, _, _) in scores})
    nulls = [c for c in concepts if c != ABLATED_CONCEPT]
    print(f"scoring rule {RULE} on the {STAT!r} statistic -- the rule "
          f"ROME_RESULTS.md quotes\n")

    # --- provenance: every cell must come from the two final checkpoints ----- #
    print("=" * 78)
    print("checkpoints")
    print("=" * 78)
    bad = False
    for model in (CONTROL_MODEL, ABLATED_MODEL):
        paths = {p for (m, _c), ps in models.items() if m == model for p in ps}
        print(f"  {model:12s} {sorted(paths)}")
        if len(paths) != 1:
            print(f"  !! {model} was scored from {len(paths)} different "
                  f"checkpoints -- the concepts are not comparable")
            bad = True
    missing = [(m, c) for m in (CONTROL_MODEL, ABLATED_MODEL) for c in concepts
               if (m, c) not in models]
    if missing:
        print(f"  !! not yet scored: {missing}")
        bad = True
    print()

    # --- every cell ---------------------------------------------------------- #
    rows = {}
    for subset in ("QA", "SimdomQA"):
        print("=" * 78)
        print(f"{subset}   (ablated - control on {RULE}/{STAT}; "
              f"negative means the ablated twin scores the gold answer lower)")
        print("=" * 78)
        head = (f"  {'concept':<22}{'split':<7}{'n':>4}{'mean':>10}{'dz':>8}"
                f"{'t':>8}{'p(t)':>10}{'p(perm)':>10}{'signs':>10}{'sign p':>10}")
        print(head)
        print("  " + "-" * (len(head) - 2))
        for concept in [ABLATED_CONCEPT] + nulls:
            for split in ("train", "test"):
                d = cell(scores, concept, subset, split)
                if d is None:
                    print(f"  {concept:<22}{split:<7}{'--':>4}   (not scored yet)")
                    continue
                s = summarise(d, rng)
                rows[(subset, concept, split)] = s
                mark = "  <-- ABLATED" if concept == ABLATED_CONCEPT else ""
                print(f"  {concept:<22}{split:<7}{s['n']:>4}{s['mean']:>+10.4f}"
                      f"{s['dz']:>+8.2f}{s['t']:>+8.2f}{s['p_t']:>10.2e}"
                      f"{s['p_signflip']:>10.2e}"
                      f"{s['neg']:>7}/{s['n']:<3}{s['sign_p']:>10.2e}{mark}")
        print("\n  p(perm) is floored at 5.0e-05 by the permutation count.")
        print()

    # --- the null distribution ---------------------------------------------- #
    verdict = {}
    print("=" * 78)
    print("the null distribution: the eight concepts neither twin held out")
    print("=" * 78)
    for subset in ("QA", "SimdomQA"):
        for split in ("train", "test"):
            null_means = [rows[(subset, c, split)]["mean"] for c in nulls
                          if (subset, c, split) in rows]
            obs = rows.get((subset, ABLATED_CONCEPT, split))
            if len(null_means) < 2 or obs is None:
                print(f"\n  {subset} / {split}: only {len(null_means)} of "
                      f"{len(nulls)} null concepts scored -- skipping")
                continue
            mu, sd = st.fmean(null_means), st.stdev(null_means)
            absmax = max(abs(x) for x in null_means)
            z = (obs["mean"] - mu) / sd if sd else float("nan")
            rank = 1 + sum(1 for x in null_means if abs(x) >= abs(obs["mean"]))
            print(f"\n  {subset} / {split}   (null n={len(null_means)})")
            print(f"    null mean            {mu:+.4f}")
            print(f"    null sd              {sd:.4f}")
            print(f"    null range           {min(null_means):+.4f} .. {max(null_means):+.4f}")
            print(f"    largest null |mean|  {absmax:.4f}")
            # Label follows ABLATED_CONCEPT: it was a literal "Ancient Rome"
            # while the value beside it already came from rows[ABLATED_CONCEPT],
            # so a retargeted run printed Baseball's number under Rome's name.
            print(f"    {ABLATED_CONCEPT:<20s} {obs['mean']:+.4f}")
            print(f"    z against the null   {z:+.2f}")
            print(f"    rank by |mean|       {rank} of {len(null_means) + 1}"
                  f"{'  (largest)' if rank == 1 else ''}")
            print(f"    ratio to largest null {abs(obs['mean']) / absmax:.2f}x"
                  if absmax else "")
            verdict[f"{subset}/{split}"] = {
                "null_n": len(null_means), "null_mean": mu, "null_sd": sd,
                "null_min": min(null_means), "null_max": max(null_means),
                "null_absmax": absmax, "observed": obs["mean"], "z": z,
                "rank_by_abs": rank, "of": len(null_means) + 1,
            }

    print("\n" + "=" * 78)
    if bad:
        print("PROVENANCE PROBLEM ABOVE -- do not quote these numbers.")
    else:
        print("Read the QA rows. The ablated concept has to sit outside the")
        print("spread of the eight it was never compared against; if it sits")
        print("inside, the shift is what these two models do everywhere and")
        print(f"the {ABLATED_CONCEPT} result is not established.")
    print("=" * 78)

    if args.json_out:
        payload = {
            "rule": RULE, "statistic": STAT, "seed": args.seed,
            "checkpoints": {m: sorted({p for (mm, _c), ps in models.items()
                                       if mm == m for p in ps})
                            for m in (CONTROL_MODEL, ABLATED_MODEL)},
            "jobs": {f"{m}/{c}": sorted(v) for (m, c), v in sorted(jobs.items())},
            "cells": {f"{s}/{c}/{sp}": v for (s, c, sp), v in rows.items()},
            "null_comparison": verdict,
            "provenance_ok": not bad,
        }
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
