#!/usr/bin/env python3
"""Measure question-level NLL proximity to the concept-excluded twin.

This analysis consumes the ``results.json`` files written by ``report.py`` for
the accuracy-selected checkpoints.  For each method and target concept it
reports

    D_abs(M,T) = mean_q |NLL_M(q) - NLL_T(q)|
    R_abs(M,T) = D_abs(M,T) / D_abs(F,T)

and the fraction of held-out target questions on which M is strictly closer to
the twin than the full model.  Confidence intervals use a paired bootstrap over
the 50 questions, preserving the method/full pairing in every resample.

Expected input layout::

    <results-root>/rome/results.json
    <results-root>/baseball/results.json
    <results-root>/ai/results.json

Example::

    python ember_eval/nll_kl/question_level_similarity.py \
      --results-root /path/to/tables_accwinners \
      --out-csv question_level_similarity.csv \
      --out-tex question_level_similarity.tex
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


TOPICS = ("rome", "baseball", "ai")
METHODS = ("EMBER", "RMU", "SNMF")
PRETTY_TOPIC = {
    "rome": "Ancient Rome",
    "baseball": "Baseball",
    "ai": "AI",
}
EXPECTED_WINNERS = {
    ("rome", "EMBER"): "ember_rome_d200",
    ("rome", "RMU"): "rmu_rome_L6hi_a10",
    ("rome", "SNMF"): "snmf_rome_ratio_out",
    ("baseball", "EMBER"): "ember_baseball_d10",
    ("baseball", "RMU"): "rmu_baseball_L6hi_a10",
    ("baseball", "SNMF"): "snmf_baseball_ratio_both",
    ("ai", "EMBER"): "ember_ai_d500",
    ("ai", "RMU"): "rmu_ai_L6hi_a10",
    ("ai", "SNMF"): "snmf_ai_ratio_in",
}


def percentile(values: Sequence[float], p: float) -> float:
    """Linearly interpolated percentile, matching common numerical packages."""
    xs = sorted(values)
    if not xs:
        raise ValueError("cannot take a percentile of an empty sequence")
    position = (len(xs) - 1) * p
    lo = int(position)
    hi = min(lo + 1, len(xs) - 1)
    weight = position - lo
    return xs[lo] * (1.0 - weight) + xs[hi] * weight


def target_deltas(row: Dict) -> Dict[str, float]:
    values = {
        item["id"]: float(item["delta"])
        for item in row["items"]
        if item["set"] == "target_test"
    }
    if len(values) != 50:
        raise ValueError(
            f"{row['label']}: expected 50 target_test items, found {len(values)}"
        )
    return values


def paired_bootstrap(
    method_abs: Sequence[float],
    full_abs: Sequence[float],
    *,
    repetitions: int,
    seed: int,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    if len(method_abs) != len(full_abs) or not method_abs:
        raise ValueError("paired arrays must have the same nonzero length")
    rng = random.Random(seed)
    n = len(method_abs)
    ratio_samples: List[float] = []
    closer_samples: List[float] = []
    for _ in range(repetitions):
        indices = [rng.randrange(n) for _ in range(n)]
        numerator = sum(method_abs[i] for i in indices) / n
        denominator = sum(full_abs[i] for i in indices) / n
        ratio_samples.append(numerator / denominator)
        closer_samples.append(
            sum(method_abs[i] < full_abs[i] for i in indices) / n
        )
    return (
        (percentile(ratio_samples, 0.025), percentile(ratio_samples, 0.975)),
        (percentile(closer_samples, 0.025), percentile(closer_samples, 0.975)),
    )


def analyse(path: Path, topic: str, repetitions: int, seed: int) -> List[Dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = {row["label"]: row for row in data["rows"]}
    baseline = target_deltas(rows["Twin vs Full"])
    output: List[Dict] = []

    for method in METHODS:
        expected = EXPECTED_WINNERS[(topic, method)]
        actual = data.get("erased", {}).get(method)
        if actual != expected:
            raise ValueError(
                f"{path}: {method} winner is {actual!r}; expected {expected!r}"
            )
        method_delta = target_deltas(rows[f"{method} vs Twin"])
        if set(method_delta) != set(baseline):
            raise ValueError(f"{path}: target question IDs differ for {method}")

        ids = sorted(baseline)
        method_abs = [abs(method_delta[item_id]) for item_id in ids]
        full_abs = [abs(baseline[item_id]) for item_id in ids]
        d_method = sum(method_abs) / len(method_abs)
        d_full = sum(full_abs) / len(full_abs)
        ratio = d_method / d_full
        closer = sum(a < b for a, b in zip(method_abs, full_abs)) / len(ids)
        ties = sum(a == b for a, b in zip(method_abs, full_abs)) / len(ids)
        ratio_ci, closer_ci = paired_bootstrap(
            method_abs,
            full_abs,
            repetitions=repetitions,
            seed=seed,
        )
        output.append(
            {
                "topic": PRETTY_TOPIC[topic],
                "method": method,
                "checkpoint": actual,
                "n": len(ids),
                "D_abs_M_T": d_method,
                "D_abs_F_T": d_full,
                "R_abs": ratio,
                "R_abs_ci_low": ratio_ci[0],
                "R_abs_ci_high": ratio_ci[1],
                "P_closer": closer,
                "P_closer_ci_low": closer_ci[0],
                "P_closer_ci_high": closer_ci[1],
                "P_tied": ties,
            }
        )
    return output


def write_csv(path: Path, rows: Iterable[Dict]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_tex(path: Path, rows: Sequence[Dict]) -> None:
    lines = [
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"Concept & Method & $R_{\mathrm{abs}}\downarrow$ & $P_{\mathrm{closer}}\uparrow$ \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['topic']} & {row['method']} & {row['R_abs']:.3f} & "
            f"{100 * row['P_closer']:.0f}\\% \\\\" 
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--out-tex", type=Path, required=True)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows: List[Dict] = []
    for topic in TOPICS:
        path = args.results_root / topic / "results.json"
        if not path.exists():
            raise SystemExit(
                f"missing {path}; copy the results.json generated by report.py "
                "for each accuracy-selected topic table"
            )
        rows.extend(
            analyse(path, topic, args.bootstrap_repetitions, args.seed)
        )
    write_csv(args.out_csv, rows)
    write_tex(args.out_tex, rows)
    print(f"wrote {args.out_csv}")
    print(f"wrote {args.out_tex}")


if __name__ == "__main__":
    main()
