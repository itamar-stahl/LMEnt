#!/usr/bin/env python3
"""Real-model automatic-delta smoke using existing low-rank CPU factors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ember.lment_pipeline import LMEntRunConfig, run_lment_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--features-root", type=Path, required=True)
    parser.add_argument("--eval-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--concept", required=True)
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--deltas", type=float, nargs="+", required=True)
    args = parser.parse_args()

    report = run_lment_pipeline(LMEntRunConfig(
        model_path=args.base_model.resolve(),
        model_key="lment-1b-control-2e",
        features_root=args.features_root.resolve(),
        output_root=args.output_dir.resolve().parent,
        output_dir=args.output_dir.resolve(),
        rank=2,
        seed=42,
        ratio_thresh=0.0,
        deltas=tuple(args.deltas),
        explicit_delta=None,
        eval_json=args.eval_json.resolve(),
        device="cuda",
        dtype="fp32",
        prepare_features=False,
        selection_mode="threshold",
        feature_ratio_threshold=args.threshold,
    ), concept=args.concept)
    summary = {
        "chosen_delta": report["chosen_delta"],
        "candidates": [
            {
                "delta": row["delta"],
                "efficacy": row["efficacy"],
                "specificity": row["specificity"],
                "objective": row["objective"],
            }
            for row in report["delta_search"]["candidates"]
        ],
        "held_out_test": report["evaluation"]["test"],
        "save": report["save"],
        "integrity": report["integrity"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
