#!/usr/bin/env python
"""Bootstrap confidence intervals for the target-test KL ratio.

    R_KL = mean_i KL(twin || edited; i) / mean_i KL(twin || full; i)

A ratio of two means, not a mean of ratios: the denominator is a property of
the concept, not of the question, so resampling has to move both means
together. Every resample draws 50 question indices with replacement and
recomputes BOTH the numerator and the denominator from the same draw, which is
what keeps the interval honest -- a question whose KL(twin || full) is large
inflates numerator and denominator at once, and independent resampling would
throw that cancellation away and widen the interval for no reason.

Two sources, one per condition kind, because the two rounds of scoring wrote
different artifacts:

    standalone (EMBER, RMU, SNMF)   <standalone-root>/<topic>/results.json
                                    kl_rows: "Full" is the denominator,
                                    the method rows are the numerators.
    ensembles  (RMU+EMBER, SNMF+EMBER)
                                    the per-question CSV, whose
                                    kl_twin_to_full is the same denominator
                                    (asserted here, not assumed).

Both are keyed on the question id, so numerator and denominator are paired per
question before anything is resampled.

    python ember_eval/nll_kl/bootstrap_rkl.py \
        --standalone-root ember_eval/nll_kl/results_accwinners \
        --ensemble-csv ember_eval/ensembles/results/nll_kl_per_question.csv \
        --out ember_eval/nll_kl/rkl_bootstrap.csv \
        --latex-out ember_eval/nll_kl/rkl_bootstrap_rows.tex

Exits non-zero unless all 15 (concept, condition) cells are present, so a
half-scored run cannot quietly produce a table with holes in it. --allow-partial
lifts that, and is for inspecting a run in progress, never for the paper.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

GROUP = "target_test"
N_BOOT = 10000
SEED = 0
PCTL = (2.5, 97.5)

TOPICS = [("rome", "Ancient Rome"), ("baseball", "Baseball"), ("ai", "Artificial intelligence")]
STANDALONE = ("EMBER", "RMU", "SNMF")
ENSEMBLE = ("RMU+EMBER", "SNMF+EMBER")
CONDITIONS = STANDALONE + ENSEMBLE

# The denominators come from two independently written artifacts; they are the
# same 50 numbers and must agree to more than the CSV's 10 printed decimals.
DENOM_TOL = 1e-8


def read_standalone(path: Path) -> Tuple[str, Dict[str, Dict[str, float]], Dict[str, str]]:
    """-> concept, {condition: {qid: KL(twin||model)}} plus 'Full', and checkpoints."""
    res = json.loads(path.read_text())
    out: Dict[str, Dict[str, float]] = {}
    for row in res["kl_rows"]:
        items = {i["id"]: i["kl"] for i in row["items"]
                 if i["set"] == GROUP and "kl" in i}
        if items:
            out[row["label"]] = items
    ckpt = dict(res.get("erased", {}))
    ckpt["Full"] = res["full"]
    ckpt["Twin"] = res["twin"]
    return res["topic"], out, ckpt


def read_ensembles(path: Path) -> Tuple[Dict[Tuple[str, str], Dict[str, float]],
                                        Dict[Tuple[str, str], Dict[str, float]],
                                        Dict[Tuple[str, str], str]]:
    num: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(dict)
    den: Dict[Tuple[str, str], Dict[str, float]] = defaultdict(dict)
    ckpt: Dict[Tuple[str, str], str] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r["group"] != GROUP:
                continue
            key = (r["concept"], r["method"])
            num[key][r["question_id"]] = float(r["kl_twin_to_method"])
            den[key][r["question_id"]] = float(r["kl_twin_to_full"])
            ckpt[key] = r["checkpoint"]
    return num, den, ckpt


def bootstrap(numer: np.ndarray, denom: np.ndarray) -> Tuple[float, float, float, float]:
    """Point estimate and percentile interval for mean(numer)/mean(denom)."""
    n = numer.size
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, n, size=(N_BOOT, n))       # 50 indices, with replacement
    # the SAME idx indexes both, so each resample is a resample of QUESTIONS
    ratios = numer[idx].mean(axis=1) / denom[idx].mean(axis=1)
    lo, hi = np.percentile(ratios, PCTL)
    return (float(numer.mean() / denom.mean()), float(lo), float(hi),
            float(ratios.std(ddof=1)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--standalone-root", required=True,
                    help="dir holding <topic>/results.json from report.py")
    ap.add_argument("--ensemble-csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--latex-out")
    ap.add_argument("--allow-partial", action="store_true",
                    help="emit fewer than 15 rows instead of failing")
    a = ap.parse_args()

    root = Path(a.standalone_root)
    ens_num, ens_den, ens_ckpt = read_ensembles(Path(a.ensemble_csv))

    recs: List[dict] = []
    problems: List[str] = []
    for slug, _expected_name in TOPICS:
        p = root / slug / "results.json"
        if not p.exists():
            problems.append(f"{slug}: {p} missing")
            continue
        concept, kl, ckpt = read_standalone(p)
        if "Full" not in kl:
            problems.append(f"{concept}: no Full row with {GROUP} KL")
            continue
        full = kl["Full"]
        ids = sorted(full)
        if len(ids) != 50:
            problems.append(f"{concept}/Full: {len(ids)} questions, expected 50")

        for cond in CONDITIONS:
            if cond in STANDALONE:
                if cond not in kl:
                    problems.append(f"{concept}/{cond}: absent from results.json")
                    continue
                num_map, den_map = kl[cond], full
                checkpoint = ckpt.get(cond, "")
            else:
                key = (concept, cond)
                if key not in ens_num:
                    problems.append(f"{concept}/{cond}: absent from the ensemble CSV")
                    continue
                num_map, den_map = ens_num[key], ens_den[key]
                checkpoint = ens_ckpt[key]
                # the ensemble CSV carries its own copy of the denominator;
                # it must be the same 50 numbers as the standalone Full row
                shared = set(den_map) & set(full)
                drift = max((abs(den_map[q] - full[q]) for q in shared), default=0.0)
                if set(den_map) != set(full):
                    problems.append(f"{concept}/{cond}: denominator ids differ from Full")
                elif drift > DENOM_TOL:
                    problems.append(f"{concept}/{cond}: denominator differs from the "
                                    f"standalone Full row by {drift:.3g}")

            common = sorted(set(num_map) & set(den_map))
            if set(num_map) != set(den_map):
                problems.append(f"{concept}/{cond}: numerator and denominator "
                                f"question ids differ ({len(num_map)} vs {len(den_map)})")
            if len(common) != 50:
                problems.append(f"{concept}/{cond}: {len(common)} paired questions, expected 50")
                continue
            numer = np.array([num_map[q] for q in common], dtype=np.float64)
            denom = np.array([den_map[q] for q in common], dtype=np.float64)
            if not (np.isfinite(numer).all() and np.isfinite(denom).all()):
                problems.append(f"{concept}/{cond}: non-finite KL")
                continue
            if (numer < -1e-9).any() or (denom < -1e-9).any():
                problems.append(f"{concept}/{cond}: negative KL")
                continue
            point, lo, hi, se = bootstrap(numer, denom)
            recs.append({
                "concept": concept, "method": cond, "checkpoint": checkpoint,
                "group": GROUP, "n": len(common),
                "mean_kl_edited": f"{numer.mean():.10f}",
                "mean_kl_full": f"{denom.mean():.10f}",
                "R_KL": f"{point:.10f}",
                "ci_lo": f"{lo:.10f}", "ci_hi": f"{hi:.10f}",
                "boot_se": f"{se:.10f}",
                "n_boot": N_BOOT, "seed": SEED,
                "ci_pct_lo": PCTL[0], "ci_pct_hi": PCTL[1],
            })

    if problems:
        print("PROBLEMS:", file=sys.stderr)
        for p in problems:
            print("  " + p, file=sys.stderr)
    expected = len(TOPICS) * len(CONDITIONS)
    if len(recs) != expected or problems:
        msg = f"{len(recs)} rows, expected {expected}"
        if not a.allow_partial:
            print(f"REFUSING to write: {msg}", file=sys.stderr)
            return 1
        print(f"WARNING (--allow-partial): {msg}", file=sys.stderr)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(recs[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(recs)

    hdr = f"{'concept':24s} {'method':12s} {'R_KL':>8s}  95% CI"
    print(hdr)
    print("-" * len(hdr))
    for r in recs:
        print(f"{r['concept']:24s} {r['method']:12s} {float(r['R_KL']):8.3f}  "
              f"[{float(r['ci_lo']):.3f}, {float(r['ci_hi']):.3f}]")
    print(f"\nwrote {out} ({len(recs)} rows)")

    if a.latex_out:
        lines = [
            "% R_KL = mean_i KL(twin||edited;i) / mean_i KL(twin||full;i), target_test, n=50.",
            f"% {N_BOOT} bootstrap resamples of the question indices, seed {SEED}, "
            f"{PCTL[0]}--{PCTL[1]} percentile interval.",
            "% Generated by ember_eval/nll_kl/bootstrap_rkl.py -- do not hand-edit.",
        ]
        for r in recs:
            lines.append(f"{r['concept']} & {r['method']} & "
                         f"{float(r['R_KL']):.2f} & "
                         f"[{float(r['ci_lo']):.2f}, {float(r['ci_hi']):.2f}] \\\\")
        tex = Path(a.latex_out)
        tex.parent.mkdir(parents=True, exist_ok=True)
        tex.write_text("\n".join(lines) + "\n")
        print("wrote", tex)
    return 0


if __name__ == "__main__":
    sys.exit(main())
