"""Create reproducible LMEnt EMBER runs and manage the shared feature cache."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import shlex
import stat
import sys
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch
import yaml

from ember.lment_pipeline import LMEntRunConfig, load_lment_config
from ember.utils import _safe_concept, _safe_model_name


PROVENANCE_FILES = (
    "concept_sentences.json",
    "neutral_sentences.json",
    "feature_manifest.json",
)
FEATURE_BRANCHES = ("csvs", "interpretations", "pickles")


@dataclass(frozen=True)
class PreparedRun:
    run_dir: Path
    source_config: Path
    effective_config: Path
    outputs_dir: Path
    runtime_config: LMEntRunConfig


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Invalid JSON in {path}: {error}") from error


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _sha256(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selected_concept_payload(path: Path, concept: str) -> list[dict[str, Any]]:
    payload = _read_json(path)
    if not isinstance(payload, list):
        raise TypeError(f"Concept JSON must contain a list: {path}")
    matches = [
        item for item in payload
        if isinstance(item, dict)
        and str(item.get("concept", "")).strip() == concept.strip()
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one record for concept {concept!r} in {path}; "
            f"found {len(matches)}")
    record = matches[0]
    if not isinstance(record.get("sentences"), list) or not record["sentences"]:
        raise ValueError(f"Concept {concept!r} has no sentence list in {path}")
    return [record]


def allocate_run_dir(runs_root: Path, concept: str, model_key: str) -> Path:
    runs_root = Path(runs_root).resolve()
    runs_root.mkdir(parents=True, exist_ok=True)
    stem = "_".join((
        _safe_concept(concept),
        _safe_model_name(model_key),
        datetime.now().strftime("%Y%m%d_%H%M%S"),
    ))
    candidate = runs_root / stem
    suffix = 1
    while candidate.exists():
        suffix += 1
        candidate = runs_root / f"{stem}_{suffix}"
    candidate.mkdir(parents=False, exist_ok=False)
    return candidate


def feature_concept_dirs(root: Path, config: LMEntRunConfig,
                         concept: str) -> Dict[str, Path]:
    common = (
        Path(f"rank{config.rank}") / f"seed{config.seed}"
        / _safe_concept(concept)
    )
    base = Path(root).resolve() / _safe_model_name(config.model_key)
    return {branch: base / branch / common for branch in FEATURE_BRANCHES}


def _feature_manifest(config: LMEntRunConfig, concept: str) -> Dict[str, Any]:
    return {
        "schema_version": 1,
        "model_key": config.model_key,
        "model_config_sha256": _sha256(Path(config.model_path) / "config.json"),
        "concept": concept,
        "rank": config.rank,
        "seed": config.seed,
        "max_iterations": config.feature_max_iterations,
        "g_sparsity": config.feature_g_sparsity,
        "k_proj": config.feature_k_proj,
    }


def add_feature_provenance(root: Path, config: LMEntRunConfig,
                           concept: str) -> None:
    if config.concept_json is None or config.neutral_json is None:
        raise ValueError("Feature provenance requires concept and neutral JSON paths")
    concept_payload = _selected_concept_payload(Path(config.concept_json), concept)
    neutral_payload = _read_json(Path(config.neutral_json))
    manifest = _feature_manifest(config, concept)
    for directory in feature_concept_dirs(root, config, concept).values():
        if not directory.is_dir():
            raise FileNotFoundError(f"Feature branch is missing: {directory}")
        _write_json(directory / "concept_sentences.json", concept_payload)
        _write_json(directory / "neutral_sentences.json", neutral_payload)
        _write_json(directory / "feature_manifest.json", manifest)


def validate_feature_bundle(root: Path, config: LMEntRunConfig,
                            concept: str) -> Dict[str, Path]:
    """Validate all three cache branches and their exact generation inputs."""
    if config.concept_json is None or config.neutral_json is None:
        raise ValueError("Feature validation requires concept and neutral JSON paths")
    directories = feature_concept_dirs(root, config, concept)
    exists = {name: path.is_dir() for name, path in directories.items()}
    if not any(exists.values()):
        raise FileNotFoundError(
            f"No shared features exist for concept {concept!r} under {Path(root).resolve()}")
    if not all(exists.values()):
        missing = [name for name, present in exists.items() if not present]
        present = [name for name, value in exists.items() if value]
        raise RuntimeError(
            f"Incomplete shared feature cache for {concept!r}; "
            f"present={present}, missing={missing}. Resolve the cache manually.")

    expected = {
        "concept_sentences.json": _selected_concept_payload(
            Path(config.concept_json), concept),
        "neutral_sentences.json": _read_json(Path(config.neutral_json)),
        "feature_manifest.json": _feature_manifest(config, concept),
    }
    for branch, directory in directories.items():
        for filename, expected_payload in expected.items():
            candidate = directory / filename
            if not candidate.is_file():
                raise RuntimeError(
                    f"Shared {branch} cache lacks {filename}: {candidate}. "
                    "The cache cannot be reused safely.")
            if _read_json(candidate) != expected_payload:
                raise RuntimeError(
                    f"Shared {branch} cache input mismatch in {candidate}. "
                    "Concept sentences, neutral sentences, model, or feature "
                    "parameters changed; resolve the cache manually.")
    return directories


def copy_shared_features_into_run(shared_root: Path, run_root: Path,
                                  config: LMEntRunConfig, concept: str) -> None:
    source = validate_feature_bundle(shared_root, config, concept)
    destination = feature_concept_dirs(run_root, config, concept)
    for branch in FEATURE_BRANCHES:
        destination[branch].parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source[branch], destination[branch])


def publish_run_features(run_root: Path, shared_root: Path,
                         config: LMEntRunConfig, concept: str) -> str:
    """Publish one complete cache bundle only when every destination is empty."""
    add_feature_provenance(run_root, config, concept)
    source = feature_concept_dirs(run_root, config, concept)
    destination = feature_concept_dirs(shared_root, config, concept)
    exists = {name: path.exists() for name, path in destination.items()}
    if all(exists.values()):
        validate_feature_bundle(shared_root, config, concept)
        return "existing_valid"
    if any(exists.values()):
        raise RuntimeError(
            f"Incomplete shared feature cache for {concept!r}; {exists}. "
            "No cache files were overwritten. Resolve it manually.")
    for branch in FEATURE_BRANCHES:
        destination[branch].parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source[branch], destination[branch])
    return "published"


def effective_config_dict(config: LMEntRunConfig) -> Dict[str, Any]:
    """Serialize the exact prepared configuration using absolute paths."""
    if config.output_dir is None:
        raise ValueError("Prepared config is missing output_dir")
    lment: Dict[str, Any] = {
        "model_key": config.model_key,
        "model_device": config.device,
        "dtype": config.dtype,
        "runs_root": str(Path(config.runs_root).resolve()),
        "run_dir": str(Path(config.output_dir).resolve().parent),
        "output_dir": str(Path(config.output_dir).resolve()),
        "data": {
            "concept_json": str(Path(config.concept_json).resolve()),
            "neutral_json": str(Path(config.neutral_json).resolve()),
        },
        "features": {
            "cache_root": str(Path(
                config.feature_cache_root or config.features_root).resolve()),
            "work_root": str(Path(config.features_root).resolve()),
            "reuse": config.reuse_features,
            "fitting_device": config.fitting_device,
            "max_iterations": config.feature_max_iterations,
            "g_sparsity": config.feature_g_sparsity,
            "k_proj": config.feature_k_proj,
        },
        "judge": {
            "model": config.judge_model,
            "device": config.judge_device,
            "max_new_tokens": config.judge_max_new_tokens,
            "local_files_only": config.judge_local_files_only,
            "cache_dir": (
                str(Path(config.judge_cache_dir).resolve())
                if config.judge_cache_dir is not None else None),
        },
        "save": {"full_model": config.full_save},
        "execution": {
            "activate_script": (
                str(Path(config.activate_script).resolve())
                if config.activate_script is not None else None),
        },
    }
    if config.slurm is not None:
        lment["slurm"] = dict(config.slurm)
    return {
        "method": "ember",
        "model_name": str(Path(config.model_path).resolve()),
        "rank": config.rank,
        "seed": config.seed,
        "selection": {
            "mode": config.selection_mode,
            "ratio_thresh": config.ratio_thresh,
            "feature_ratio_threshold": config.feature_ratio_threshold,
            "judge_confidence_threshold": config.judge_confidence_threshold,
            "judge_top_k": config.judge_top_k,
        },
        "ember": {
            "deltas": list(config.deltas),
            "explicit_delta": config.explicit_delta,
        },
        "eval": {
            "data_json": (
                str(Path(config.eval_json).resolve())
                if config.eval_json is not None else None),
            "alpaca": config.alpaca_eval,
            "alpaca_split": config.alpaca_split,
            "alpaca_max_items": config.alpaca_max_items,
        },
        "lment": lment,
    }


def _write_wrappers(prepared: PreparedRun, concept: str,
                    execution: str) -> None:
    python = str(Path(sys.executable).resolve())
    config = str(prepared.effective_config.resolve())
    command = [
        python, "-m", "ember.lment_worker",
        "--config", config,
        "--concept", concept,
        "--execution", execution,
    ]
    activation = prepared.runtime_config.activate_script
    activation_line = (
        f". {shlex.quote(str(Path(activation).resolve()))}\n"
        if activation is not None else ""
    )
    shell = (
        "#!/bin/sh\n"
        "set -eu\n"
        f"{activation_line}"
        f"{shlex.join(command)}\n"
    )
    shell_path = prepared.run_dir / "run_wrapper.sh"
    shell_path.write_text(shell, encoding="utf-8", newline="\n")
    os.chmod(shell_path, os.stat(shell_path).st_mode | stat.S_IXUSR)

    ps_command = " ".join(
        "'" + str(part).replace("'", "''") + "'" for part in command)
    powershell = (
        "$ErrorActionPreference = 'Stop'\n"
        f"& {ps_command}\n"
        "if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }\n"
    )
    (prepared.run_dir / "run_wrapper.ps1").write_text(
        powershell, encoding="utf-8", newline="\n")


def prepare_run(config_path: Path, concept: str, *,
                execution: str = "local") -> PreparedRun:
    """Create a run folder, snapshot all inputs, and stage reusable features."""
    source_config_path = Path(config_path).resolve()
    config = load_lment_config(source_config_path)
    if config.concept_json is None or config.neutral_json is None:
        raise ValueError("LMEnt config must provide concept and neutral JSON files")
    concept_payload = _selected_concept_payload(Path(config.concept_json), concept)
    neutral_payload = _read_json(Path(config.neutral_json))
    run_dir = allocate_run_dir(config.runs_root, concept, config.model_key)
    inputs = run_dir / "inputs"
    outputs = run_dir / "outputs"
    feature_work = outputs / "features"
    inputs.mkdir()
    outputs.mkdir()
    shutil.copy2(source_config_path, run_dir / "source_config.yaml")
    concept_copy = inputs / "concept_sentences.json"
    neutral_copy = inputs / "neutral_sentences.json"
    _write_json(concept_copy, concept_payload)
    _write_json(neutral_copy, neutral_payload)
    eval_copy = None
    if config.eval_json is not None:
        eval_copy = inputs / "eval.json"
        shutil.copy2(config.eval_json, eval_copy)

    runtime = replace(
        config,
        features_root=feature_work,
        output_dir=outputs,
        concept_json=concept_copy,
        neutral_json=neutral_copy,
        eval_json=eval_copy,
        config_path=run_dir / "config.yaml",
    )
    shared_root = config.feature_cache_root or config.features_root
    if runtime.reuse_features:
        copy_shared_features_into_run(
            shared_root, feature_work, runtime, concept)
    effective = run_dir / "config.yaml"
    effective.write_text(
        yaml.safe_dump(effective_config_dict(runtime), sort_keys=False),
        encoding="utf-8",
        newline="\n",
    )
    prepared = PreparedRun(
        run_dir=run_dir,
        source_config=run_dir / "source_config.yaml",
        effective_config=effective,
        outputs_dir=outputs,
        runtime_config=runtime,
    )
    _write_wrappers(prepared, concept, execution)
    (run_dir / "client.log").write_text(
        "\n".join((
            f"prepared_at={datetime.now(timezone.utc).isoformat()}",
            f"execution={execution}",
            f"source_config={source_config_path}",
            f"effective_config={effective}",
            f"run_dir={run_dir}",
            f"concept={concept}",
        )) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return prepared


def write_run_environment(run_dir: Path, *, execution: str) -> Path:
    payload: Dict[str, Any] = {
        "schema_version": 1,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution": execution,
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": str(Path(sys.executable).resolve()),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(0)
        payload["gpu"] = {
            "name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "free_bytes": free,
            "total_bytes": total,
            "bf16_supported": torch.cuda.is_bf16_supported(),
        }
    path = Path(run_dir).resolve() / "run_environment.json"
    _write_json(path, payload)
    return path


__all__ = [
    "FEATURE_BRANCHES", "PROVENANCE_FILES", "PreparedRun", "allocate_run_dir",
    "feature_concept_dirs", "add_feature_provenance", "validate_feature_bundle",
    "copy_shared_features_into_run", "publish_run_features",
    "effective_config_dict", "prepare_run", "write_run_environment",
]
