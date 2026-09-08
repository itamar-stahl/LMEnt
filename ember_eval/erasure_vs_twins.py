#!/usr/bin/env python
"""Does EMBER's erasure land where never-training landed? Three paired contrasts.

The twins answer "what does never training on Ancient Rome do". This asks
whether a post-hoc embedding erasure reaches the same place, which is the
question the pair was built for.

Three contrasts on the SAME questions, all paired per question:

    ablated - control   what the ablation did. Reproduces ROME_RESULTS.md, and
                        is here as a regression check: if these numbers move,
                        something about the scoring changed and nothing else
                        below can be trusted.
    erased  - control   what the erasure did, on the same scale.
    erased  - ablated   the residual. Near zero on the concept would mean the
                        two interventions arrive at the same state; that is the
                        claim, and it needs an EQUIVALENCE argument, not a
                        non-significant p-value (see below).

WHY THE RULE IS IMPORTED, NOT REIMPLEMENTED
-------------------------------------------
`build_rules`/`statistics_for` come from `completion_eval/metric_bakeoff.py`,
exactly as `cross_concept_null.py` takes them, so the erasure cannot be scored by
a rule that has drifted from the one the twins were scored by. That failure has
already cost this project once: a mid-run edit to `evaluate_completion.py` faked
an 18-point swing (`lment-eval-metric-split`). The three models' result files
were also produced by byte-identical code -- md5 249cc08c14f35851a945a00bd2f994a2.

READING "NO DIFFERENCE" HONESTLY
--------------------------------
A large p on `erased - ablated` does NOT establish the two are the same; at
n = 50 this test cannot resolve differences smaller than roughly 0.4 sd. So the
`dz` and its CI are what to read, and the comparison worth making is against the
size of the effect being reproduced: a residual of 0.05 against an ablation
effect of -0.37 means something, a residual of 0.25 does not.

    python ember_eval/erasure_vs_twins.py
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
import sys
from collections import defaultdict

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "completion_eval"))

from metric_bakeoff import build_rules, dz, perm_p_paired, statistics_for  # noqa: E402

FNAME = re.compile(
    r"^cmpl_([A-Za-z0-9]+)_([a-z0-9_]+?)_(QA|SimdomQA)_(train|test)_(\d+)\.json$")
TEMPERATURE = 1.0
STAT = "gold"
RULE = "pmi_per_char"
CONCEPT = "Ancient Rome"

CONTROL, ABLATED, ERASED = "control2e", "norome2e", "erased2e"
LABEL = {CONTROL: "control twin", ABLATED: "ablated twin (no-rome)",
         ERASED: "EMBER-erased control"}


def load(results_dir: str):
    rule = build_rules(None)[RULE]
    scores: dict = {}
    ckpts: dict = defaultdict(set)
    for path in sorted(glob.glob(os.path.join(results_dir, "cmpl_*.json"))):
        m = FNAME.match(os.path.basename(path))
        if not m:
            continue
        model, _tag, subset, split, _job = m.groups()
        if model not in (CONTROL, ABLATED, ERASED):
            continue
        blob = json.load(open(path))
        for rec in blob["records"]:
            if rec["concept"] != CONCEPT:
                continue
            key = (model, subset, split, rec["question"])
            if key in scores:
                raise SystemExit(f"{path}: duplicate question {key}")
            scores[key] = statistics_for(rule, TEMPERATURE, rec)[STAT]
        ckpts[model].add(blob["metadata"]["model"])
    return scores, ckpts


def paired(scores, a: str, b: str, subset: str, split: str):
    """a - b, over the questions both scored."""
    A = {q: v for (m, s, sp, q), v in scores.items()
         if m == a and s == subset and sp == split}
    B = {q: v for (m, s, sp, q), v in scores.items()
         if m == b and s == subset and sp == split}
    shared = sorted(set(A) & set(B))
    return np.array([A[q] - B[q] for q in shared]) if shared else None


def summarise(d: np.ndarray, rng) -> dict:
    n = len(d)
    mean = float(d.mean())
    sd = float(d.std(ddof=1)) if n > 1 else 0.0
    t = mean / (sd / math.sqrt(n)) if sd else 0.0
    p_t = float(2 * stats.t.sf(abs(t), n - 1)) if sd and n > 1 else 1.0
    z = float(dz(d))
    # CI on dz, normal approximation to its standard error -- enough to say
    # whether a "no difference" residual is actually pinned down or just noisy.
    se = math.sqrt(1.0 / n + z * z / (2 * n)) if n > 1 else float("nan")
    return {"n": n, "mean": mean, "dz": z, "dz_lo": z - 1.96 * se,
            "dz_hi": z + 1.96 * se, "t": t, "p_t": p_t,
            "p_signflip": float(perm_p_paired(d, rng))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results",
                    default="/home/dcor/galbarak2/LMEnt-ember/ember_eval/results/completion")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    scores, ckpts = load(args.results)
    missing = [m for m in (CONTROL, ABLATED, ERASED) if m not in ckpts]
    if missing:
        raise SystemExit(f"no results for: {missing}")

    print(f"scoring rule {RULE} on {STAT!r}, imported from metric_bakeoff "
          f"(same as cross_concept_null.py)\nconcept: {CONCEPT}\n")
    for m in (CONTROL, ABLATED, ERASED):
        for c in sorted(ckpts[m]):
            print(f"  {LABEL[m]:<24} {c}")
    print()

    rng = np.random.default_rng(args.seed)
    contrasts = [(ABLATED, CONTROL, "ablated - control   (what ablation did)"),
                 (ERASED, CONTROL, "erased  - control   (what erasure did)"),
                 (ERASED, ABLATED, "erased  - ablated   (the residual)")]
    out = {}
    for subset in ("QA", "SimdomQA"):
        for split in ("train", "test"):
            name = f"{subset}/{split}"
            print(f"=== Rome {name}" if subset == "QA"
                  else f"=== Simdom-Rome {name}")
            print(f"    {'contrast':<38} {'mean':>9} {'dz':>7} "
                  f"{'dz 95% CI':>16} {'t':>7} {'p_t':>9}")
            for a, b, label in contrasts:
                d = paired(scores, a, b, subset, split)
                if d is None:
                    print(f"    {label:<38}  (no shared questions)")
                    continue
                s = summarise(d, rng)
                out[f"{name}|{a}-{b}"] = s
                print(f"    {label:<38} {s['mean']:>+9.4f} {s['dz']:>+7.3f} "
                      f"[{s['dz_lo']:>+6.2f},{s['dz_hi']:>+6.2f}] "
                      f"{s['t']:>+7.2f} {s['p_t']:>9.2g}")
            print()

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump({"concept": CONCEPT, "rule": RULE, "statistic": STAT,
                       "seed": args.seed, "cells": out}, fh, indent=2)
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
