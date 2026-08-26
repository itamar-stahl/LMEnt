#!/usr/bin/env python3
"""Verify the observable outputs of the real H100 end-to-end smoke run."""
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
    selection = report.get("feature_selection", {})
    assert selection.get("mode") == "judge", (
        "cluster smoke did not use the real Gemma feature judge")
    selected = selection.get("selected_feature_ids")
    assert isinstance(selected, list) and selected, "Gemma selected no features"

    alpaca = report.get("alpaca", {})
    assert alpaca.get("n") == 1 and alpaca.get("max_items") == 1, (
        "cluster smoke must evaluate exactly one Alpaca item")
    assert alpaca.get("gpu_profile", {}).get("name") == "h100", (
        "Alpaca did not use the H100 profile")
    for key in ("mean_relevance", "mean_fluency"):
        score = alpaca.get(key)
        assert isinstance(score, (int, float)) and 0.0 <= float(score) <= 2.0, (
            f"invalid Alpaca score: {key}={score!r}")

    stdout = log_out_path.read_text(encoding="utf-8", errors="replace").lower()
    assert '"conda_env": "lment"' in stdout, "node did not use lment Conda env"
    assert '"gpu":' in stdout and "h100" in stdout, "node preflight did not report H100"
    stderr = log_err_path.read_text(encoding="utf-8", errors="replace").lower()
    fatal = [pattern for pattern in _FATAL_STDERR if pattern in stderr]
    assert not fatal, f"fatal pattern(s) in Slurm stderr: {fatal}"

    return {
        "integrity": True,
        "save_mode": "embedding_only",
        "selected_features": selected,
        "alpaca_items": 1,
        "gpu_profile": "h100",
        "report": str(report_path.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--log-out", type=Path, required=True)
    parser.add_argument("--log-err", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(
        verify_e2e_report(args.report, args.log_out, args.log_err), indent=2))


if __name__ == "__main__":
    main()


__all__ = ["verify_e2e_report", "main"]
