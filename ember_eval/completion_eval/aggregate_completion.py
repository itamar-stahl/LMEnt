#!/usr/bin/env python
"""Two ways of reading `evaluate_completion.py` output.

    twins   control against ablated (or base against erased), paired per
            question, on the continuous score rather than on right/wrong.

    score   efficacy, specificity and the restricted h-score against a base
            model, the same arithmetic as `aggregate_mc.py` with soft accuracy
            in place of accuracy.

`twins` is the one that matters for the never-learned comparison. It pairs the
two runs question by question and tests the mean difference with a sign-flip
permutation, which assumes nothing about the shape of the differences.

**It reports tiers side by side and does not subtract them.** Until 2026-08-27
the headline number here was `QA - SimdomQA`, on the reasoning that both
subsets carry the same global drift so subtracting leaves what is specific to
the concept. That reasoning fails whenever the neighbouring domain is itself
affected, and subtracting something the intervention moved is the textbook bad
control -- it removes part of the very signal being measured.

It is not hypothetical here. On the 2-epoch twins the neighbouring set moved
about four times as much as the concept did (-0.0339 against -0.0086 in
log P(gold)/char), and which control you subtract decides the *sign* of the
answer: the neighbouring domain gives +0.0252, a far concept gives -0.0190.
Every one of those is inside the noise, which is the point -- the number was
being set by the choice of control rather than by the models.

For erasure methods the objection is stronger still, because damaging nearby
concepts is the documented failure mode of RMU and SNMF. A neighbouring-domain
control would quietly subtract away exactly the collateral damage such a
comparison exists to detect.

So three tiers, reported and never collapsed:

    target   the concept in question             -- efficacy
    near     the neighbouring domain             -- collateral damage, a RESULT
    far      something the intervention cannot   -- the drift reference
             plausibly have touched

A specific, well-aimed intervention looks like large / smaller / zero across
those three. The shape is the evidence; one subtracted number throws it away.

    python aggregate_completion.py twins \\
        --control results/control_qa_train.json results/control_sim_train.json \\
        --ablated results/noporn_qa_train.json  results/noporn_sim_train.json \\
        --control-far results/control_hp_qa.json \\
        --ablated-far results/noporn_hp_qa.json

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
            out[key] = stats_from_record(r)
    return out


def softmax(xs: Sequence[float]) -> List[float]:
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    total = sum(exps)
    return [e / total for e in exps]


def stats_from_record(r: Dict[str, Any]) -> Dict[str, float]:
    """Every statistic, rebuilt from the stored per-option log-likelihoods.

    Deliberately not read off the record's own `p_correct` / `margin` / `correct`
    fields. Runs from before 2026-08-27 wrote those from
    `pmi = log P(a|stem) - log P(a|null)`, and newer ones write them from
    per-character conditional scores, so trusting the stored values would
    silently compare two different quantities whenever an old run is paired
    against a new one. `logp_conditional` and the option strings are recorded by
    every version, so recomputing here makes all records comparable and leaves
    this script's output independent of which scorer produced its input.
    """
    cond = r["logp_conditional"]
    chars = r.get("option_chars") or [len(f" {o}") for o in r["options"]]
    per_char = [c / n for c, n in zip(cond, chars)]
    g = r["correct_index"]
    others = [per_char[i] for i in range(len(per_char)) if i != g]

    mean_chars = sum(chars) / len(chars)
    probs = softmax([s * mean_chars for s in per_char])
    pred = max(range(len(per_char)), key=lambda i: per_char[i])
    return {
        "gold_per_char": per_char[g],
        "p_correct": probs[g],
        "logp": math.log(max(probs[g], 1e-12)),
        "margin": per_char[g] - max(others),
        "correct": float(pred == g),
    }


def tier_row(ctl, abl, keys: Sequence, stat: str, draws: int,
             rng: random.Random) -> Tuple[int, float, float, float, float, float]:
    """n, control mean, ablated mean, mean difference, dz, p for one tier."""
    ds = [abl[k][stat] - ctl[k][stat] for k in keys]
    c = statistics.fmean(ctl[k][stat] for k in keys)
    a = statistics.fmean(abl[k][stat] for k in keys)
    s = statistics.pstdev(ds)
    mean_d = statistics.fmean(ds)
    return (len(ds), c, a, mean_d, mean_d / s if s else 0.0,
            signflip_p(ds, draws, rng))


def twins(args: argparse.Namespace) -> None:
    rng = random.Random(args.seed)
    ctl = collect(args.control)
    abl = collect(args.ablated)
    shared = sorted(set(ctl) & set(abl))
    if not shared:
        raise SystemExit("the two sides share no questions; wrong files?")

    # target and near come from the same files, separated by their subset;
    # far is a different concept entirely and so pairs on its own keys.
    tiers: List[Tuple[str, str, Any, Any, List]] = []
    target = [k for k in shared if k[1] == "QA"]
    near = [k for k in shared if k[1] == "SimdomQA"]
    if target:
        tiers.append(("target", "QA", ctl, abl, target))
    if near:
        tiers.append(("near", "SimdomQA", ctl, abl, near))

    if bool(args.control_far) != bool(args.ablated_far):
        raise SystemExit("--control-far and --ablated-far go together")
    if args.control_far:
        fctl, fabl = collect(args.control_far), collect(args.ablated_far)
        fshared = sorted(set(fctl) & set(fabl))
        if not fshared:
            raise SystemExit("the two far sides share no questions; wrong files?")
        overlap = {k[0] for k in fshared} & {k[0] for k in shared}
        if overlap:
            raise SystemExit(
                f"the far set covers the same concept(s) as the target: "
                f"{sorted(overlap)}. The drift reference has to be something "
                "the intervention cannot plausibly have touched.")
        label = "/".join(sorted({k[0] for k in fshared}))
        tiers.append(("far", label, fctl, fabl, fshared))
    else:
        print("note: no --control-far/--ablated-far given, so there is no drift\n"
              "      reference. Without one, a difference at the target cannot be\n"
              "      told apart from the two models simply differing everywhere.\n")

    print(f"paired on {len(shared)} questions "
          f"(control {len(ctl)}, ablated {len(abl)})\n")

    stats = ("gold_per_char", "p_correct", "margin", "correct")
    for stat in stats:
        primary = stat == "gold_per_char"
        print("=" * 78)
        print(f"{stat}   (ablated - control; negative means the ablation hurt)"
              + ("   <- PRIMARY" if primary else ""))
        if not primary:
            print("sanity check only -- reads the three distractors, which the "
                  "stem never showed the model")
        print("=" * 78)
        head = (f"  {'tier':<8}{'set':<16}{'n':>5}{'control':>10}{'ablated':>10}"
                f"{'diff':>10}{'dz':>8}{'p':>8}")
        print(head)
        print("  " + "-" * (len(head) - 2))
        for tier, label, c_side, a_side, keys in tiers:
            n, c, a, d, dz, p = tier_row(c_side, a_side, keys, stat,
                                         args.draws, rng)
            print(f"  {tier:<8}{label:<16}{n:>5}{c:>10.4f}{a:>10.4f}"
                  f"{d:>+10.4f}{dz:>+8.3f}{p:>8.3f}")

        if primary:
            print("\n  Read the shape, not any single row. A specific, well-aimed")
            print("  intervention is large at target, smaller at near, ~0 at far.")
            print("  'near' is collateral damage -- a result to report, never a")
            print("  control to subtract.")

        if args.legacy_did and target and near:
            qa = [abl[k][stat] - ctl[k][stat] for k in target]
            sim = [abl[k][stat] - ctl[k][stat] for k in near]
            gap = statistics.fmean(qa) - statistics.fmean(sim)
            pooled = qa + sim
            hits = 0
            for _ in range(args.draws):
                shuf = pooled[:]
                rng.shuffle(shuf)
                if abs(statistics.fmean(shuf[:len(qa)])
                       - statistics.fmean(shuf[len(qa):])) >= abs(gap) - 1e-12:
                    hits += 1
            print(f"\n  [legacy] QA - SimdomQA: {gap:+.4f}, label-permutation "
                  f"p = {(hits + 1) / (args.draws + 1):.3f}")
            print("  Retained only to reproduce numbers published before "
                  "2026-08-27. It\n  subtracts the near tier, which the "
                  "intervention may itself have moved.")
        print()


# --------------------------------------------------------------------------- #
# score
# --------------------------------------------------------------------------- #
def value(path: str, key: str) -> float:
    blob = read(path)
    metric = blob.get("metadata", {}).get("metric", "")
    if metric.startswith("pmi"):
        print(f"warning: {path} was scored before 2026-08-27, so its "
              f"{key!r} is\n         built on pmi rather than per-character "
              "conditional scores. Do not\n         mix it with newer runs in "
              "one h-score; rerun evaluate_completion.py.")
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
    t.add_argument("--control", nargs="+", required=True,
                   help="target (QA) and near (SimdomQA) runs for the control")
    t.add_argument("--ablated", nargs="+", required=True,
                   help="the same runs for the ablated or erased model")
    t.add_argument("--control-far", nargs="+",
                   help="drift reference: a concept the intervention cannot "
                        "plausibly have touched, scored on the control")
    t.add_argument("--ablated-far", nargs="+",
                   help="the same concept scored on the ablated model")
    t.add_argument("--legacy-did", action="store_true",
                   help="also print the pre-2026-08-27 QA - SimdomQA number, "
                        "which subtracts the near tier and is kept only for "
                        "reproducing already-published figures")
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
