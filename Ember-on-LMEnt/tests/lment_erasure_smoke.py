#!/usr/bin/env python3
"""Mechanical real-model EMBER smoke test (no concept evaluation).

The default run uses a temporary workspace. Pass ``--keep-erased-model`` to
retain the independently reloadable erased embedding and its reports.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SNMF_ROOT = PROJECT_ROOT / "external" / "snmf"
for import_root in (PROJECT_ROOT, SNMF_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from ember.erased_embedding import load_lment_with_erased_embeddings
from ember.lment_pipeline import LMEntRunConfig, run_lment_pipeline
from ember.utils import _safe_concept, _safe_model_name


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Erase and reload a real LMEnt model without running concept evaluation.")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--concept-json", type=Path, required=True)
    parser.add_argument("--neutral-json", type=Path, required=True)
    parser.add_argument("--concept", default="Culture of Greece")
    parser.add_argument("--model-key", default="lment-1b-control-2e")
    parser.add_argument("--rank", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-iterations", type=int, default=2)
    parser.add_argument("--delta", type=float, default=0.1)
    parser.add_argument("--model-device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--keep-erased-model",
        action="store_true",
        help="Keep the erased embedding. Without this flag the temporary workspace is deleted.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="New directory for a preserved run; only used with --keep-erased-model.",
    )
    return parser


def _source_snapshot(model_path: Path) -> Dict[str, tuple[int, int]]:
    return {
        str(path.relative_to(model_path)): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in model_path.rglob("*")
        if path.is_file()
    }


def _prepare_cpu_features(args: argparse.Namespace, workspace: Path) -> tuple[Path, float]:
    features_root = workspace / "features"
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (
        str(PROJECT_ROOT), str(SNMF_ROOT), env.get("PYTHONPATH", ""))))
    subprocess.run([
        sys.executable,
        "-m",
        "ember.train_mf_features",
        "--concepts",
        args.concept,
        "--ranks",
        str(args.rank),
        "--seed",
        str(args.seed),
        "--model-name",
        str(args.model_path),
        "--model-key",
        args.model_key,
        "--model-device",
        "cpu",
        "--fitting-device",
        "cpu",
        "--concept-json",
        str(args.concept_json),
        "--neutral-json",
        str(args.neutral_json),
        "--skip-mlp",
        "--max-iterations",
        str(args.max_iterations),
        "--g-sparsity",
        "0.01",
        "--k-proj",
        "2",
        "--outdir",
        str(features_root),
    ], cwd=PROJECT_ROOT, env=env, check=True)

    common = (
        Path(f"rank{args.rank}")
        / f"seed{args.seed}"
        / _safe_concept(args.concept)
        / "embedding"
    )
    model_root = features_root / _safe_model_name(args.model_key)
    stats_path = model_root / "csvs" / common / "stats_embed.csv"
    stats = pd.read_csv(stats_path)
    ratios = pd.to_numeric(stats["ratio_abs"], errors="coerce")
    if ratios.notna().any():
        metric_score = float(ratios.max())
    else:
        metric_score = 0.0

    return features_root, metric_score


def _verify_erased_model_on_device(base_model: Path, artifact: Path,
                                   device: str) -> Dict[str, Any]:
    model, tokenizer = load_lment_with_erased_embeddings(
        base_model, artifact, device=device, dtype=torch.float32)
    try:
        actual_device = model.get_input_embeddings().weight.device
        first = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 0
        second = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else first
        input_ids = torch.tensor([[first, second]], dtype=torch.long, device=actual_device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, use_cache=False).logits
        if not torch.isfinite(logits).all():
            raise AssertionError("Erased-model logits contain non-finite values")
        return {
            "requested_device": device,
            "actual_device": str(actual_device),
            "logits_shape": list(logits.shape),
            "finite_logits": True,
        }
    finally:
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _run(args: argparse.Namespace, workspace: Path) -> Dict[str, Any]:
    model_path = args.model_path.resolve()
    before = _source_snapshot(model_path)
    features_root, feature_threshold = _prepare_cpu_features(args, workspace)
    output_root = workspace / "erased"
    report = run_lment_pipeline(LMEntRunConfig(
        model_path=model_path,
        model_key=args.model_key,
        features_root=features_root,
        runs_root=output_root,
        rank=args.rank,
        seed=args.seed,
        ratio_thresh=0.0,
        explicit_delta=args.delta,
        eval_json=None,
        device=args.model_device,
        dtype="fp32",
        selection_mode="threshold",
        feature_ratio_threshold=feature_threshold,
    ), concept=args.concept)
    artifact = Path(report["erased_embeddings_path"]).resolve()
    device_report = _verify_erased_model_on_device(
        model_path, artifact, args.model_device)
    source_unchanged = before == _source_snapshot(model_path)

    if report["evaluation"] is not None:
        raise AssertionError("Mechanical smoke test unexpectedly ran concept evaluation")
    if not report["integrity"]["passed"]:
        raise AssertionError("Erased artifact failed the save/reload integrity audit")
    if int(report["edit"]["n_tokens_edited"]) <= 0:
        raise AssertionError("EMBER did not edit any token embeddings")
    if not source_unchanged:
        raise AssertionError("Source checkpoint files changed during erasure")

    smoke_report: Dict[str, Any] = {
        "schema_version": 1,
        "kind": "mechanical_erasure_smoke_test",
        "end_to_end_concept_evaluation_run": False,
        "concept": args.concept,
        "selected_features": report["artifact"]["selected_feature_ids"],
        "feature_ratio_threshold": feature_threshold,
        "factorization_device": "cpu",
        "source_model": str(model_path),
        "source_files_unchanged": source_unchanged,
        "erased_embeddings": str(artifact),
        "ember_report": str(artifact.parent / "report.json"),
        "preserved": bool(args.keep_erased_model),
        "edit": report["edit"],
        "integrity": report["integrity"],
        "erased_model_inference": device_report,
    }
    report_path = workspace / "smoke_report.json"
    report_path.write_text(json.dumps(smoke_report, indent=2), encoding="utf-8")
    smoke_report["smoke_report"] = str(report_path.resolve())
    print(json.dumps(smoke_report, indent=2))
    return smoke_report


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    args = build_parser().parse_args(argv)
    args.model_path = args.model_path.resolve()
    args.concept_json = args.concept_json.resolve()
    args.neutral_json = args.neutral_json.resolve()
    if args.model_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--model-device cuda requested, but CUDA is unavailable")
    if args.output_root is not None and not args.keep_erased_model:
        raise ValueError("--output-root requires --keep-erased-model")

    if args.keep_erased_model:
        workspace = (
            args.output_root.resolve()
            if args.output_root is not None
            else PROJECT_ROOT / "runs" / datetime.now(
                timezone.utc).strftime("smoke-%Y%m%dT%H%M%SZ")
        )
        if workspace.exists():
            raise FileExistsError(f"Preserved output directory already exists: {workspace}")
        workspace.mkdir(parents=True)
        return _run(args, workspace)

    with tempfile.TemporaryDirectory(prefix="ember-lment-smoke-") as tmp:
        return _run(args, Path(tmp))


if __name__ == "__main__":
    main()
