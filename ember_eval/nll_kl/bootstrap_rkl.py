#!/usr/bin/env python3
"""Paired-bootstrap confidence intervals for target-question R_KL.

R_KL is a ratio of two means,

    mean_i KL(T || M; i) / mean_i KL(T || F; i),

where T, M, and F are the twin, edited, and full models.  A bootstrap draw
therefore resamples question indices once and recomputes *both* means from the
same indices.  Bootstrapping the numerator and denominator separately would
discard their question-level pairing.

The standalone input is the output of ``nll_kl/report.py`` for the checkpoints
selected by accuracy.  The ensemble input is the long-form per-question CSV
written by ``ensembles/export_nll_kl.py``.  No model is loaded and no GPU is
needed.

Example (on the machine containing the current report outputs):

    python ember_eval/nll_kl/bootstrap_rkl.py \
      --standalone-root /path/to/results_accwinners \
      --ensemble-csv ember_eval/ensembles/results/nll_kl_per_question.csv \
      --out ember_eval/nll_kl/rkl_bootstrap.csv \
      --latex-out ember_eval/nll_kl/rkl_bootstrap_rows.tex
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_SELECTION = HERE.parent / "acc_selection/results_sciq/selected_checkpoints.json"
DEFAULT_ENSEMBLES = HERE.parent / "ensembles/results/nll_kl_per_question.csv"

CONCEPTS = {
    "rome": "Ancient Rome",
    "baseball": "Baseball",
    "ai": "Artificial intelligence",
}
METHOD_ORDER = ("EMBER", "RMU", "SNMF", "RMU+EMBER", "SNMF+EMBER")
TARGET_SET = "target_test"


@dataclass(frozen=True)
class Condition:
    concept: str
    method: str
    checkpoint: str
    question_ids: Tuple[str, ...]
    kl_method: np.ndarray
    kl_full: np.ndarray


def target_kl(items: Iterable[Mapping[str, object]]) -> Dict[str, float]:
    """Return per-question KL values for held-out target questions."""
    out: Dict[str, float] = {}
    for item in items:
        if item.get("set") != TARGET_SET or item.get("phase") != "test":
            continue
        question_id = str(item["id"])
        if question_id in out:
            raise ValueError(f"duplicate target question id: {question_id}")
        if "kl" not in item:
            raise ValueError(f"target question {question_id} has no KL value")
        out[question_id] = float(item["kl"])
    return out


def selected_checkpoints(path: Path) -> Dict[Tuple[str, str], str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        (str(row["topic"]), str(row["method"])): str(row["model_label"])
        for row in data["winners"]
    }


def load_standalone(root: Path, selection_path: Path) -> List[Condition]:
    """Load current EMBER/RMU/SNMF per-question KL from report.py outputs."""
    selected = selected_checkpoints(selection_path)
    conditions: List[Condition] = []
    for slug, concept in CONCEPTS.items():
        path = root / slug / "results.json"
        if not path.exists():
            raise FileNotFoundError(f"missing standalone report: {path}")
        report = json.loads(path.read_text(encoding="utf-8"))
        rows = {str(row["label"]): row for row in report["kl_rows"]}
        if "Full" not in rows:
            raise ValueError(f"{path}: kl_rows has no Full row")
        full = target_kl(rows["Full"]["items"])

        for method in METHOD_ORDER[:3]:
            if method not in rows:
                raise ValueError(f"{path}: kl_rows has no {method} row")
            row = rows[method]
            expected = selected[(slug, method)]
            actual = str(row["evaluated"])
            if actual != expected:
                raise ValueError(
                    f"{path}: {method} uses {actual}, but the selection manifest "
                    f"specifies {expected}"
                )
            method_kl = target_kl(row["items"])
            ids = tuple(sorted(full))
            if tuple(sorted(method_kl)) != ids:
                raise ValueError(f"{path}: {method} and Full target ids do not match")
            conditions.append(
                Condition(
                    concept=concept,
                    method=method,
                    checkpoint=actual,
                    question_ids=ids,
                    kl_method=np.asarray([method_kl[i] for i in ids], dtype=np.float64),
                    kl_full=np.asarray([full[i] for i in ids], dtype=np.float64),
                )
            )
    return conditions


def load_ensembles(path: Path) -> List[Condition]:
    """Load current ensemble per-question KL from the published long-form CSV."""
    grouped: Dict[Tuple[str, str, str], List[Mapping[str, str]]] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["group"] != TARGET_SET:
                continue
            key = (row["concept"], row["method"], row["checkpoint"])
            grouped.setdefault(key, []).append(row)

    conditions: List[Condition] = []
    for (concept, method, checkpoint), rows in grouped.items():
        by_id = {row["question_id"]: row for row in rows}
        if len(by_id) != len(rows):
            raise ValueError(f"{concept}/{method}: duplicate target question ids")
        ids = tuple(sorted(by_id))
        conditions.append(
            Condition(
                concept=concept,
                method=method,
                checkpoint=checkpoint,
                question_ids=ids,
                kl_method=np.asarray(
                    [float(by_id[i]["kl_twin_to_method"]) for i in ids], dtype=np.float64
                ),
                kl_full=np.asarray(
                    [float(by_id[i]["kl_twin_to_full"]) for i in ids], dtype=np.float64
                ),
            )
        )
    return conditions


def validate(conditions: Sequence[Condition]) -> None:
    keys = [(row.concept, row.method) for row in conditions]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate concept/method condition across inputs")
    for row in conditions:
        n = len(row.question_ids)
        if n != 50:
            raise ValueError(f"{row.concept}/{row.method}: expected 50 questions, found {n}")
        if row.kl_method.shape != (n,) or row.kl_full.shape != (n,):
            raise ValueError(f"{row.concept}/{row.method}: malformed KL arrays")
        if not np.all(np.isfinite(row.kl_method)) or not np.all(np.isfinite(row.kl_full)):
            raise ValueError(f"{row.concept}/{row.method}: non-finite KL value")
        if np.any(row.kl_method < -1e-8) or np.any(row.kl_full < -1e-8):
            raise ValueError(f"{row.concept}/{row.method}: negative KL value")
        if float(row.kl_full.mean()) <= 0:
            raise ValueError(f"{row.concept}/{row.method}: non-positive full-model denominator")

    # The full-model series is shared by all methods within a concept.  The
    # ensemble CSV stores ten decimals, so allow only its final-rounding error.
    for concept in {row.concept for row in conditions}:
        same = [row for row in conditions if row.concept == concept]
        baseline = same[0]
        for row in same[1:]:
            if row.question_ids != baseline.question_ids or not np.allclose(
                row.kl_full, baseline.kl_full, rtol=0.0, atol=5e-10
            ):
                raise ValueError(
                    f"{concept}: full-model per-question KL differs across methods"
                )


def bootstrap(row: Condition, repetitions: int, seed: int) -> Tuple[float, float, float]:
    point = float(row.kl_method.mean() / row.kl_full.mean())
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(row.question_ids), size=(repetitions, len(row.question_ids)))
    numerator = row.kl_method[indices].mean(axis=1)
    denominator = row.kl_full[indices].mean(axis=1)
    if np.any(denominator <= 0):
        raise ValueError(f"{row.concept}/{row.method}: bootstrap produced zero denominator")
    ratios = numerator / denominator
    low, high = np.percentile(ratios, [2.5, 97.5])
    return point, float(low), float(high)


def sort_key(row: Condition) -> Tuple[int, int]:
    concept_order = list(CONCEPTS.values())
    return concept_order.index(row.concept), METHOD_ORDER.index(row.method)


def write_outputs(
    rows: Sequence[Tuple[Condition, float, float, float]], out: Path, latex_out: Path | None
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("concept", "method", "checkpoint", "n", "R_KL", "ci95_low", "ci95_high"))
        for condition, point, low, high in rows:
            writer.writerow(
                (condition.concept, condition.method, condition.checkpoint,
                 len(condition.question_ids), f"{point:.10f}", f"{low:.10f}", f"{high:.10f}")
            )

    if latex_out is not None:
        latex_out.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for condition, point, low, high in rows:
            method = condition.method.replace("+", r"\,+\,")
            lines.append(
                f"{condition.concept} & {method} "
                f"& {point:.3f} [{low:.3f}, {high:.3f}] \\\\"
            )
        latex_out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--standalone-root", type=Path)
    parser.add_argument("--ensemble-csv", type=Path, default=DEFAULT_ENSEMBLES)
    parser.add_argument("--selected-checkpoints", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--latex-out", type=Path)
    parser.add_argument("--repetitions", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="permit fewer than all 15 paper conditions (for input diagnostics only)",
    )
    args = parser.parse_args()
    if args.repetitions <= 0:
        parser.error("--repetitions must be positive")

    conditions: List[Condition] = []
    if args.standalone_root is not None:
        conditions.extend(load_standalone(args.standalone_root, args.selected_checkpoints))
    if args.ensemble_csv is not None:
        if not args.ensemble_csv.exists():
            raise FileNotFoundError(f"missing ensemble CSV: {args.ensemble_csv}")
        conditions.extend(load_ensembles(args.ensemble_csv))
    validate(conditions)

    expected = {(concept, method) for concept in CONCEPTS.values() for method in METHOD_ORDER}
    actual = {(row.concept, row.method) for row in conditions}
    missing = sorted(expected - actual)
    if missing and not args.allow_partial:
        formatted = ", ".join(f"{concept}/{method}" for concept, method in missing)
        raise ValueError(
            "inputs do not cover all 15 paper conditions; missing: " + formatted
        )

    results = [
        (row, *bootstrap(row, args.repetitions, args.seed))
        for row in sorted(conditions, key=sort_key)
    ]
    write_outputs(results, args.out, args.latex_out)
    print(f"Wrote {len(results)} R_KL intervals to {args.out}")
    if args.latex_out is not None:
        print(f"Wrote LaTeX rows to {args.latex_out}")
    if missing:
        print(f"Partial diagnostic run: {len(missing)} paper conditions were absent")


if __name__ == "__main__":
    main()
