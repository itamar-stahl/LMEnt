#!/usr/bin/env python3
"""Verify observable outputs of a real Slurm end-to-end erasure run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


_FATAL_STDERR = (
    "traceback (most recent call last)",
    "cuda out of memory",
    "slurmstepd: error",
    "fatal:",
)


def verify_e2e_report(
    report_path: str | Path,
    log_out_path: str | Path,
    log_err_path: str | Path,
    *,
    expected_selection: str = "judge",
    expected_gpu: str = "h100",
    require_alpaca: bool = True,
) -> Dict[str, Any]:
    report_path = Path(report_path)
    log_out_path = Path(log_out_path)
    log_err_path = Path(log_err_path)
    for label, path in (
        ("EMBER report", report_path),
        ("Slurm stdout", log_out_path),
        ("Slurm stderr", log_err_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report.get("integrity", {}).get("passed") is True, (
        "erased checkpoint integrity did not pass")
    assert report.get("save", {}).get("mode") == "embedding_only", (
        "cluster smoke must test the default embedding-only output")
    assert report.get("feature_cache", {}).get("status") in {
        "published", "existing_valid",
    }, "feature cache was not validated or published"
    selection = report.get("feature_selection", {})
    assert selection.get("mode") == expected_selection, (
        f"expected {expected_selection!r} feature selection; got "
        f"{selection.get('mode')!r}")
    selected = selection.get("selected_feature_ids")
    assert isinstance(selected, list) and selected, "no features were selected"

    erased_path = Path(str(report.get("erased_embeddings_path", "")))
    assert erased_path.is_file(), f"erased embedding artifact not found: {erased_path}"
    evaluation = report.get("evaluation")
    assert isinstance(evaluation, dict), "concept evaluation was not run"
    for split in ("train", "test"):
        result = evaluation.get(split)
        assert isinstance(result, dict), f"missing {split} evaluation"
        assert isinstance(result.get("baseline"), dict), (
            f"missing pristine {split} evaluation")
        assert isinstance(result.get("edited"), dict), (
            f"missing erased {split} evaluation")

    alpaca = report.get("alpaca", {})
    if require_alpaca:
        assert alpaca.get("n") == 1 and alpaca.get("max_items") == 1, (
            "cluster smoke must evaluate exactly one Alpaca item")
        assert isinstance(alpaca.get("gpu"), str) and alpaca["gpu"], (
            "Alpaca did not record the scheduler-provided GPU")
        for key in ("mean_relevance", "mean_fluency"):
            score = alpaca.get(key)
            assert isinstance(score, (int, float)) and 0.0 <= float(score) <= 2.0, (
                f"invalid Alpaca score: {key}={score!r}")

    stdout = log_out_path.read_text(encoding="utf-8", errors="replace").lower()
    stderr = log_err_path.read_text(encoding="utf-8", errors="replace").lower()
    fatal = [
        pattern for pattern in _FATAL_STDERR
        if pattern in stderr or pattern in stdout
    ]
    assert not fatal, f"fatal pattern(s) in Slurm stderr: {fatal}"
    run_environment = report_path.parent.parent / "run_environment.json"
    assert run_environment.is_file(), "run_environment.json was not written"
    environment = json.loads(run_environment.read_text(encoding="utf-8"))
    assert environment.get("cuda_available") is True, "run did not use CUDA"
    gpu = str(environment.get("gpu", {}).get("name", ""))
    assert expected_gpu.lower() in gpu.lower(), (
        f"expected GPU containing {expected_gpu!r}; got {gpu!r}")
    assert environment.get("slurm_job_id"), "run did not record a Slurm job ID"

    return {
        "integrity": True,
        "save_mode": "embedding_only",
        "selected_features": selected,
        "alpaca_items": int(alpaca.get("n", 0)),
        "gpu": gpu,
        "report": str(report_path.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--log-out", type=Path, required=True)
    parser.add_argument("--log-err", type=Path, required=True)
    parser.add_argument(
        "--expected-selection", choices=("judge", "threshold"), default="judge")
    parser.add_argument("--expected-gpu", default="h100")
    parser.add_argument("--no-alpaca", action="store_true")
    args = parser.parse_args()
    print(json.dumps(
        verify_e2e_report(
            args.report,
            args.log_out,
            args.log_err,
            expected_selection=args.expected_selection,
            expected_gpu=args.expected_gpu,
            require_alpaca=not args.no_alpaca,
        ),
        indent=2,
    ))


if __name__ == "__main__":
    main()


__all__ = ["verify_e2e_report", "main"]
