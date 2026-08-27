#!/usr/bin/env python3
"""Run and verify a retained real-checkpoint LMEnt EMBER test flow."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any, Dict, Mapping, Optional, Sequence

import yaml

from ember.erasure import io
from ember.lment_pipeline import load_lment_config
from ember.lment_runs import feature_concept_dirs, prepare_run
from ember.lment_worker import execute_prepared_run


MIN_REAL_EMBEDDING_BYTES = 100_000_000


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"Missing {label}: {path}")
    return path


def _read_json(path: Path) -> Any:
    return json.loads(_require_file(path, path.name).read_text(encoding="utf-8"))


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _source_snapshot(model_path: Path) -> Dict[str, tuple[int, int]]:
    return {
        str(path.relative_to(model_path)): (
            int(path.stat().st_size), int(path.stat().st_mtime_ns))
        for path in model_path.rglob("*") if path.is_file()
    }


def verify_real_flow_run(run_dir: Path) -> Dict[str, Any]:
    """Verify retained outputs produced by the full real-checkpoint flow."""
    run_dir = Path(run_dir).resolve()
    outputs = run_dir / "outputs"
    report_path = _require_file(outputs / "report.json", "outputs/report.json")
    report = _read_json(report_path)
    config_path = _require_file(run_dir / "config.yaml", "effective config")
    config = load_lment_config(config_path)
    environment = _read_json(run_dir / "run_environment.json")

    for required in (
        run_dir / "source_config.yaml",
        run_dir / "inputs" / "concept_sentences.json",
        run_dir / "inputs" / "neutral_sentences.json",
        run_dir / "inputs" / "eval.json",
    ):
        _require_file(required, "retained run input")

    if report.get("save", {}).get("mode") != "embedding_only":
        raise AssertionError("Real-flow test must save an embedding-only artifact")
    erased = Path(str(report.get("erased_embeddings_path", ""))).resolve()
    _require_file(erased, "erased embedding")
    if not _is_within(erased, outputs):
        raise AssertionError(f"Erased embedding escaped the run folder: {erased}")
    if erased.stat().st_size < MIN_REAL_EMBEDDING_BYTES:
        raise AssertionError(
            f"Erased embedding is too small for the real LMEnt checkpoint: "
            f"{erased.stat().st_size} bytes")

    if report.get("integrity", {}).get("passed") is not True:
        raise AssertionError("Erased embedding integrity audit did not pass")
    if int(report.get("edit", {}).get("n_tokens_edited", 0)) <= 0:
        raise AssertionError("Real flow edited no token embeddings")
    if report["integrity"].get("pristine_embedding_sha256") == report[
            "integrity"].get("edited_embedding_sha256"):
        raise AssertionError("Pristine and erased embedding hashes are identical")

    evaluation = report.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise AssertionError("Real flow did not run control/erased evaluation")
    for split in ("train", "test"):
        split_report = evaluation.get(split)
        if not isinstance(split_report, Mapping):
            raise AssertionError(f"Missing real {split} evaluation")
        for state in ("baseline", "edited"):
            state_report = split_report.get(state)
            if not isinstance(state_report, Mapping):
                raise AssertionError(f"Missing {split} {state} evaluation")
            if any(int(row.get("n", 0)) <= 0 for row in state_report.values()):
                raise AssertionError(f"Empty {split} {state} evaluation subset")

    feature_dirs = feature_concept_dirs(
        config.features_root, config, str(report["concept"]))
    required_features = (
        feature_dirs["csvs"] / "embedding" / "stats_embed.csv",
        feature_dirs["csvs"] / "embedding" / "token_features.csv",
        feature_dirs["interpretations"] / "embedding" / "potential_features.csv",
        feature_dirs["pickles"] / "embedding" / "embedding.pkl",
    )
    for path in required_features:
        _require_file(path, "real feature artifact")
        if not _is_within(path, outputs / "features"):
            raise AssertionError(f"Feature artifact escaped the run folder: {path}")
    for directory in feature_dirs.values():
        for name in (
            "concept_sentences.json", "neutral_sentences.json",
            "feature_manifest.json",
        ):
            _require_file(directory / name, "feature provenance")

    devices = report.get("devices", {})
    actual_model_device = str(devices.get("actual_model_device", ""))
    if config.device == "cpu" and actual_model_device != "cpu":
        raise AssertionError(f"CPU flow used model device {actual_model_device!r}")
    if config.device == "cuda" and not actual_model_device.startswith("cuda"):
        raise AssertionError(f"GPU flow used model device {actual_model_device!r}")
    if devices.get("requested_fitting_device") != config.fitting_device:
        raise AssertionError("Report does not match the requested fitting device")
    if config.device == "cuda" and environment.get("cuda_available") is not True:
        raise AssertionError("GPU flow did not record available CUDA")

    cache_status = report.get("feature_cache", {}).get("status")
    if cache_status not in {"published", "existing_valid"}:
        raise AssertionError(f"Feature cache was not published or validated: {cache_status}")
    return {
        "schema_version": 1,
        "kind": "real_lment_ember_flow_test",
        "passed": True,
        "run_dir": str(run_dir),
        "concept": report["concept"],
        "model_path": str(config.model_path),
        "model_device": config.device,
        "actual_model_device": actual_model_device,
        "fitting_device": config.fitting_device,
        "execution": environment.get("execution"),
        "selected_feature_ids": report["feature_selection"]["selected_feature_ids"],
        "edited_tokens": int(report["edit"]["n_tokens_edited"]),
        "erased_embedding": str(erased),
        "erased_embedding_bytes": int(erased.stat().st_size),
        "feature_artifacts": [str(path.resolve()) for path in required_features],
        "evaluation_report": str(report_path),
        "feature_cache_status": cache_status,
    }


def run_real_flow(config_path: Path, concept: str, *, execution: str) -> Dict[str, Any]:
    """Run one actual flow and retain a machine-readable evidence report."""
    source = load_lment_config(config_path)
    model_path = Path(source.model_path).resolve()
    _require_file(model_path / "config.json", "real LMEnt config.json")
    before = _source_snapshot(model_path)
    prepared = prepare_run(config_path, concept, execution=execution)
    started = time.perf_counter()
    try:
        execute_prepared_run(
            prepared.effective_config, concept, execution=execution)
        source_unchanged = before == _source_snapshot(model_path)
        if not source_unchanged:
            raise AssertionError("The source LMEnt checkpoint changed during the test")
        evidence = verify_real_flow_run(prepared.run_dir)
        evidence.update({
            "source_files_unchanged": True,
            "duration_seconds": time.perf_counter() - started,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        io.save_json_atomic(
            prepared.outputs_dir / "real_flow_test_report.json", evidence)
    except Exception as error:
        failure = {
            "schema_version": 1,
            "kind": "real_lment_ember_flow_test",
            "passed": False,
            "run_dir": str(prepared.run_dir),
            "error_type": type(error).__name__,
            "error": str(error),
            "duration_seconds": time.perf_counter() - started,
        }
        io.save_json_atomic(
            prepared.outputs_dir / "real_flow_test_failure.json", failure)
        raise
    print(json.dumps(evidence, indent=2))
    print(f"[real-run] {prepared.run_dir}")
    return evidence


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--concept", required=True)
    parser.add_argument(
        "--execution", choices=("windows", "local", "slurm"), required=True)
    args = parser.parse_args(argv)
    run_real_flow(args.config, args.concept, execution=args.execution)


if __name__ == "__main__":
    main()


__all__ = [
    "MIN_REAL_EMBEDDING_BYTES", "verify_real_flow_run", "run_real_flow", "main",
]
