#!/usr/bin/env python
"""Two ways of reading `evaluate_completion.py` output.

    twins   control against ablated (or base against erased), paired per
            question, on the continuous score rather than on right/wrong.

    score   efficacy, specificity and the restricted h-score against a base
            model, the same arithmetic as `aggregate_mc.py` with soft accuracy
            in place of accuracy.

`twins` is the one that matters for the never-learned comparison. It pairs the
two runs question by question and tests the mean difference with a sign-flip
permutation, which assumes nothing about the shape of the differences. The
ablated twin drifts slightly worse everywhere, so a difference on the concept's
own questions only means something next to the difference on the neighbouring
domain: QA minus SimdomQA subtracts the drift, since both subsets carry it
equally.

    python aggregate_completion.py twins \\
        --control results/control_qa_train.json results/control_sim_train.json \\
        --ablated results/noporn_qa_train.json  results/noporn_sim_train.json

    python aggregate_completion.py score \\
        --concept results/erased_qa.json   --base-concept results/base_qa.json \\
        --simdom  results/erased_sim.json  --base-simdom  results/base_sim.json \\
        --mmlu    results/erased_mmlu.json --base-mmlu    results/base_mmlu.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple


def read(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# twins
# --------------------------------------------------------------------------- #
def signflip_p(diffs: Sequence[float], draws: int, rng: random.Random) -> float:
    """Two-sided p for mean(diffs) != 0 under random sign flips of each pair."""
    if not diffs:
        return 1.0
    n = len(diffs)
    obs = abs(statistics.fmean(diffs))
    hits = 0
    for _ in range(draws):
        total = sum(d if rng.random() < 0.5 else -d for d in diffs)
        if abs(total) / n >= obs - 1e-12:
            hits += 1
    return (hits + 1) / (draws + 1)


def collect(paths: Sequence[str]) -> Dict[Tuple[str, str, str, str], Dict[str, float]]:
    """{(concept, subset, split, question): {p_correct, logp, margin, correct}}."""
    out = {}
    for path in paths:
        blob = read(path)
        for r in blob["records"]:
            if "p_correct" not in r:
                raise SystemExit(f"{path}: records have no p_correct; rerun "
                                 "evaluate_completion.py")
            key = (r["concept"], r["subset"], r["split"], r["question"])
            if key in out:
                raise SystemExit(f"question appears twice across the given files: {key}")
            out[key] = {
                "p_correct": r["p_correct"],
                "logp": math.log(max(r["p_correct"], 1e-12)),
                "margin": r["margin"],
                "correct": float(r["correct"]),
            }
    return out


def twins(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    ctl = collect(args.control)
    abl = collect(args.ablated)
    shared = sorted(set(ctl) & set(abl))
    if not shared:
        raise SystemExit("the two sides share no questions; wrong files?")
    print(f"paired on {len(shared)} questions "
          f"(control {len(ctl)}, ablated {len(abl)})\n")

    for stat in ("p_correct", "logp", "margin", "correct"):
        diffs = {k: abl[k][stat] - ctl[k][stat] for k in shared}
        by_subset: Dict[str, List[float]] = defaultdict(list)
        for (_, subset, _, _), d in diffs.items():
            by_subset[subset].append(d)
        all_d = list(diffs.values())

        print("=" * 72)
        print(f"{stat}   (ablated - control; negative means the ablation hurt)")
        print("=" * 72)
        head = f"  {'subset':<12}{'n':>5}{'control':>10}{'ablated':>10}{'diff':>10}{'dz':>8}{'p':>8}"
        print(head)
        print("  " + "-" * (len(head) - 2))
        for subset in sorted(by_subset):
            ds = by_subset[subset]
            keys = [k for k in shared if k[1] == subset]
            c = statistics.fmean(ctl[k][stat] for k in keys)
            a = statistics.fmean(abl[k][stat] for k in keys)
            s = statistics.pstdev(ds)
            dz = statistics.fmean(ds) / s if s else 0.0
            print(f"  {subset:<12}{len(ds):>5}{c:>10.4f}{a:>10.4f}"
                  f"{statistics.fmean(ds):>+10.4f}{dz:>+8.3f}"
                  f"{signflip_p(ds, args.draws, rng):>8.3f}")

        if {"QA", "SimdomQA"} <= set(by_subset):
            qa, sim = by_subset["QA"], by_subset["SimdomQA"]
            gap = statistics.fmean(qa) - statistics.fmean(sim)
            pooled = qa + sim
            hits = 0
            for _ in range(args.draws):
                shuf = pooled[:]
                rng.shuffle(shuf)
                if abs(statistics.fmean(shuf[:len(qa)])
                       - statistics.fmean(shuf[len(qa):])) >= abs(gap) - 1e-12:
                    hits += 1
            print(f"\n  QA - SimdomQA: {gap:+.4f}, label-permutation p = "
                  f"{(hits + 1) / (args.draws + 1):.3f}")
            print("  (this is the drift-free number: both subsets carry the same "
                  "global drift,\n   so subtracting them leaves what is specific "
                  "to the concept)")
        print(f"\n  pooled mean {statistics.fmean(all_d):+.4f}, "
              f"sd {statistics.pstdev(all_d):.4f}, "
              f"p = {signflip_p(all_d, args.draws, rng):.3f}\n")


# --------------------------------------------------------------------------- #
# score
# --------------------------------------------------------------------------- #
def value(path: str, key: str) -> float:
    blob = read(path)
    if key in blob:
        return float(blob[key])
    if key == "soft_accuracy" and "accuracy" in blob:
        print(f"note: {path} has no soft_accuracy (multiple-choice run?), "
              "using accuracy")
        return float(blob["accuracy"])
    raise SystemExit(f"{path} has neither {key!r} nor 'accuracy'")


def normalize(v: float, base: float, chance: float = 0.25) -> float:
    if base <= chance:
        raise SystemExit(
            f"base score {base:.4f} is not above chance {chance:.2f}; the "
            "chance-corrected normalization is undefined. Check that the base "
            "model clears chance before reading anything into an h-score.")
    return min(1.0, max(0.0, (v - chance) / (base - chance)))


def harmonic(values: Sequence[float]) -> float:
    if any(not math.isfinite(v) or v <= 0 for v in values):
        return 0.0
    return len(values) / sum(1 / v for v in values)


def score(args: argparse.Namespace) -> None:
    names = ("concept", "simdom", "mmlu")
    raw = {n: value(getattr(args, n), args.score) for n in names}
    base = {n: value(getattr(args, f"base_{n}"), args.score) for n in names}
    norm = {n: normalize(raw[n], base[n], args.chance) for n in names}
    efficacy = 1.0 - norm["concept"]
    specificity = harmonic([norm["simdom"], norm["mmlu"]])
    report = {
        "score_key": args.score,
        "raw": raw, "base": base, "normalized": norm,
        "phi_efficacy": efficacy,
        "phi_specificity": specificity,
        "restricted_h_score": harmonic([efficacy, specificity]),
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("twins", help="paired control-vs-ablated comparison")
    t.add_argument("--control", nargs="+", required=True)
    t.add_argument("--ablated", nargs="+", required=True)
    t.add_argument("--draws", type=int, default=20000)
    t.add_argument("--seed", type=int, default=0)
    t.set_defaults(func=twins)

    s = sub.add_parser("score", help="efficacy, specificity, h-score")
    for n in ("concept", "simdom", "mmlu"):
        s.add_argument(f"--{n}", required=True)
        s.add_argument(f"--base-{n}", required=True)
    s.add_argument("--score", default="soft_accuracy",
                   choices=("soft_accuracy", "accuracy"))
    s.add_argument("--chance", type=float, default=0.25)
    s.add_argument("--out")
    s.set_defaults(func=score)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
