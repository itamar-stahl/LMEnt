#!/usr/bin/env python3
"""Calculate SciQ-based held-out H scores from the saved accuracy rollup.

The output includes the full model, each concept-excluded twin, and the three
selected erasure models for every concept. The full model is the normalization
baseline for every row.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


CHANCE = 0.25
FULL_MODEL = "FULL_lment-1b-control-2e-b131k"
TWINS = {
    "rome": "TWIN_lment-1b-norome-2e-b131k",
    "baseball": "TWIN_lment-1b-nobaseball-2e-b131k",
    "ai": "TWIN_lment-1b-noai-2e-b131k",
}
TOPICS = ("rome", "baseball", "ai")
METHODS = ("EMBER", "RMU", "SNMF")


def clip01(value: float) -> float:
    return min(1.0, max(0.0, value))


def harmonic_mean(first: float, second: float) -> float:
    if first <= 0.0 or second <= 0.0:
        return 0.0
    return 2.0 * first * second / (first + second)


def parse_args() -> argparse.Namespace:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Calculate held-out H_test for full, twin, and selected models."
    )
    parser.add_argument(
        "--rollup",
        type=Path,
        default=here / "test_phase" / "test_accuracy_rollup.csv",
    )
    parser.add_argument(
        "--rankings",
        type=Path,
        default=here / "results_sciq" / "rankings.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=here / "final" / "H_test_sciq.csv",
    )
    return parser.parse_args()


def read_accuracies(path: Path) -> dict[tuple[str, str, str, str], float]:
    values = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (
                row["model"],
                row["topic"],
                row["group"],
                row["unrelated_source"],
            )
            if key in values:
                raise ValueError(f"duplicate accuracy row: {key}")
            values[key] = float(row["accuracy"])
    return values


def read_winners(path: Path) -> dict[tuple[str, str], str]:
    winners = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["selected"] != "yes":
                continue
            key = (row["topic"], row["method"])
            if key in winners:
                raise ValueError(f"multiple selected checkpoints for {key}")
            winners[key] = row["model_label"]
    expected = {(topic, method) for topic in TOPICS for method in METHODS}
    if set(winners) != expected:
        raise ValueError(f"winner cells differ from expected cells: {set(winners) ^ expected}")
    return winners


def lookup(
    accuracies: dict[tuple[str, str, str, str], float],
    model: str,
    topic: str,
) -> tuple[float, float, float]:
    keys = (
        (model, topic, "target_test", ""),
        (model, topic, "neighbour_test", ""),
        (model, "(any)", "unrelated_test", "sciq"),
    )
    missing = [key for key in keys if key not in accuracies]
    if missing:
        raise ValueError(f"missing accuracy rows for {model}/{topic}: {missing}")
    return tuple(accuracies[key] for key in keys)  # type: ignore[return-value]


def calculate_row(
    topic: str,
    role: str,
    model: str,
    model_acc: tuple[float, float, float],
    full_acc: tuple[float, float, float],
) -> dict[str, str | float]:
    denominators = tuple(value - CHANCE for value in full_acc)
    if min(denominators) <= 0.0:
        raise ValueError(f"full-model accuracy is not above chance for {topic}")

    normalized = tuple(
        clip01((value - CHANCE) / denominator)
        for value, denominator in zip(model_acc, denominators)
    )
    efficacy = 1.0 - normalized[0]
    preservation = harmonic_mean(normalized[1], normalized[2])
    h_test = harmonic_mean(efficacy, preservation)

    return {
        "topic": topic,
        "role": role,
        "model": model,
        "acc_target": round(model_acc[0], 6),
        "acc_neighbour": round(model_acc[1], 6),
        "acc_sciq": round(model_acc[2], 6),
        "full_target": round(full_acc[0], 6),
        "full_neighbour": round(full_acc[1], 6),
        "full_sciq": round(full_acc[2], 6),
        "norm_target": round(normalized[0], 6),
        "norm_neighbour": round(normalized[1], 6),
        "norm_sciq": round(normalized[2], 6),
        "phi_efficacy": round(efficacy, 6),
        "phi_preservation": round(preservation, 6),
        "H_test": round(h_test, 6),
    }


def main() -> None:
    args = parse_args()
    accuracies = read_accuracies(args.rollup)
    winners = read_winners(args.rankings)
    rows = []

    for topic in TOPICS:
        full_acc = lookup(accuracies, FULL_MODEL, topic)
        models = [("Full", FULL_MODEL), ("Twin", TWINS[topic])]
        models.extend((method, winners[(topic, method)]) for method in METHODS)
        for role, model in models:
            rows.append(calculate_row(topic, role, model, lookup(accuracies, model, topic), full_acc))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.out}")
    for row in rows:
        print(f"{row['topic']:8} {row['role']:5} H_test={row['H_test']:.6f}")


if __name__ == "__main__":
    main()
