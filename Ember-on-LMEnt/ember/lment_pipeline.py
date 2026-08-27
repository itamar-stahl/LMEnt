"""Standalone EMBER orchestration for LMEnt base-model checkpoints."""
from __future__ import annotations

import gc
import hashlib
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import pandas as pd
import torch
import yaml

from ember.erasure import embed_edit, features, io, model_loader
from ember import judge_callbacks
from ember.erased_embedding import (
    load_lment_with_erased_embeddings,
    save_erased_embedding,
)
from ember.evals.causal_mc import evaluate_causal_mc
from ember.evals.harmonic import harmonic_mean
from ember.evals.mc import prepare_mc_items
from ember.lment_feature_selection import (
    FeatureSelectionResult,
    TextCallback,
    select_by_threshold,
    select_with_judge,
)
from ember.local_datasets import load_mc_qa_items
from ember.utils import _safe_concept, _safe_model_name


DEFAULT_DELTAS = [0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 200.0]


@dataclass(frozen=True)
class LMEntRunConfig:
    model_path: Path
    model_key: str
    features_root: Path
    runs_root: Path
    feature_cache_root: Optional[Path] = None
    rank: int = 100
    seed: int = 42
    ratio_thresh: float = 2.0
    deltas: Sequence[float] = tuple(DEFAULT_DELTAS)
    explicit_delta: Optional[float] = None
    eval_json: Optional[Path] = None
    device: str = "auto"
    dtype: str = "fp32"
    reuse_features: bool = False
    fitting_device: str = "cuda"
    concept_json: Optional[Path] = None
    neutral_json: Optional[Path] = None
    feature_max_iterations: int = 20_000
    feature_g_sparsity: float = 0.01
    feature_k_proj: int = 30
    output_dir: Optional[Path] = None
    full_save: bool = False
    selection_mode: str = "judge"
    feature_ratio_threshold: Optional[float] = None
    judge_confidence_threshold: float = 0.85
    judge_top_k: int = 20
    alpaca_eval: bool = False
    alpaca_split: str = "test"
    alpaca_max_items: Optional[int] = None
    judge_model: Optional[str] = None
    judge_revision: Optional[str] = None
    judge_device: str = "cuda"
    judge_max_new_tokens: int = 256
    judge_local_files_only: bool = False
    judge_cache_dir: Optional[Path] = None
    activate_script: Optional[Path] = None
    slurm: Optional[Mapping[str, Any]] = None
    config_path: Optional[Path] = None


def load_lment_config(path: str | Path) -> LMEntRunConfig:
    """Load the original-style EMBER YAML with LMEnt extras under ``lment``."""
    config_path = Path(path).resolve()
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise TypeError("LMEnt config must be a YAML mapping")
    top_fields = {"method", "model_name", "rank", "seed", "selection", "ember", "eval", "lment"}
    unknown = set(raw) - top_fields
    if unknown:
        raise ValueError(f"Unknown LMEnt config keys: {sorted(unknown)}")
    missing = {"method", "model_name", "lment"} - set(raw)
    if missing:
        raise ValueError(f"Missing LMEnt config keys: {sorted(missing)}")
    if raw["method"] != "ember":
        raise ValueError("LMEnt runner requires method: ember")

    def mapping(name: str, value: Any) -> Dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise TypeError(f"{name} must be a YAML mapping")
        return dict(value)

    selection = mapping("selection", raw.get("selection"))
    ember = mapping("ember", raw.get("ember"))
    evaluation = mapping("eval", raw.get("eval"))
    lment = mapping("lment", raw["lment"])
    data = mapping("lment.data", lment.pop("data", None))
    feature = mapping("lment.features", lment.pop("features", None))
    judge = mapping("lment.judge", lment.pop("judge", None))
    save = mapping("lment.save", lment.pop("save", None))
    execution = mapping("lment.execution", lment.pop("execution", None))
    slurm = mapping("lment.slurm", lment.pop("slurm", None))

    allowed = {
        "selection": {"ratio_thresh", "mode", "feature_ratio_threshold", "judge_confidence_threshold", "judge_top_k"},
        "ember": {"deltas", "explicit_delta"},
        "eval": {"data_json", "alpaca", "alpaca_split", "alpaca_max_items"},
        "lment": {"model_key", "model_device", "dtype", "runs_root", "run_dir", "output_dir"},
        "lment.data": {"concept_json", "neutral_json"},
        "lment.features": {"cache_root", "work_root", "reuse", "fitting_device", "max_iterations", "g_sparsity", "k_proj"},
        "lment.judge": {"model", "revision", "device", "max_new_tokens", "local_files_only", "cache_dir"},
        "lment.save": {"full_model"},
        "lment.execution": {"activate_script"},
        "lment.slurm": {"job_name", "account", "partition", "constraint", "time_minutes", "cpu_mem_mb", "cpus_per_task"},
    }
    sections = {
        "selection": selection, "ember": ember, "eval": evaluation,
        "lment": lment, "lment.data": data, "lment.features": feature,
        "lment.judge": judge, "lment.save": save,
        "lment.execution": execution, "lment.slurm": slurm,
    }
    for name, values in sections.items():
        extra = set(values) - allowed[name]
        if extra:
            raise ValueError(f"Unknown {name} config keys: {sorted(extra)}")

    required_lment = {"model_key", "model_device", "dtype", "runs_root"} - set(lment)
    required_data = {"concept_json", "neutral_json"} - set(data)
    required_features = {"cache_root", "reuse", "fitting_device"} - set(feature)
    if required_lment:
        raise ValueError(f"Missing lment config keys: {sorted(required_lment)}")
    if required_data:
        raise ValueError(f"Missing lment.data config keys: {sorted(required_data)}")
    if required_features:
        raise ValueError(f"Missing lment.features config keys: {sorted(required_features)}")

    base = config_path.parent

    def resolve_path(value: Any) -> Path:
        candidate = Path(value).expanduser()
        return candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()

    config = LMEntRunConfig(
        model_path=resolve_path(raw["model_name"]),
        model_key=str(lment["model_key"]),
        features_root=resolve_path(feature.get("work_root", feature["cache_root"])),
        runs_root=resolve_path(lment["runs_root"]),
        feature_cache_root=resolve_path(feature["cache_root"]),
        rank=int(raw.get("rank", 100)),
        seed=int(raw.get("seed", 42)),
        ratio_thresh=float(selection.get("ratio_thresh", 2.0)),
        deltas=tuple(float(delta) for delta in ember.get("deltas", DEFAULT_DELTAS)),
        explicit_delta=(None if ember.get("explicit_delta") is None else float(ember["explicit_delta"])),
        eval_json=(None if evaluation.get("data_json") is None else resolve_path(evaluation["data_json"])),
        device=str(lment["model_device"]),
        dtype=str(lment["dtype"]),
        reuse_features=bool(feature["reuse"]),
        fitting_device=str(feature["fitting_device"]),
        concept_json=resolve_path(data["concept_json"]),
        neutral_json=resolve_path(data["neutral_json"]),
        feature_max_iterations=int(feature.get("max_iterations", 20_000)),
        feature_g_sparsity=float(feature.get("g_sparsity", 0.01)),
        feature_k_proj=int(feature.get("k_proj", 30)),
        output_dir=(None if lment.get("output_dir") is None else resolve_path(lment["output_dir"])),
        full_save=bool(save.get("full_model", False)),
        selection_mode=str(selection.get("mode", "judge")),
        feature_ratio_threshold=(None if selection.get("feature_ratio_threshold") is None else float(selection["feature_ratio_threshold"])),
        judge_confidence_threshold=float(selection.get("judge_confidence_threshold", 0.85)),
        judge_top_k=int(selection.get("judge_top_k", 20)),
        alpaca_eval=bool(evaluation.get("alpaca", False)),
        alpaca_split=str(evaluation.get("alpaca_split", "test")),
        alpaca_max_items=(None if evaluation.get("alpaca_max_items") is None else int(evaluation["alpaca_max_items"])),
        judge_model=(None if judge.get("model") is None else str(judge["model"])),
        judge_revision=(None if judge.get("revision") is None else str(judge["revision"])),
        judge_device=str(judge.get("device", "cuda")),
        judge_max_new_tokens=int(judge.get("max_new_tokens", 256)),
        judge_local_files_only=bool(judge.get("local_files_only", False)),
        judge_cache_dir=(None if judge.get("cache_dir") is None else resolve_path(judge["cache_dir"])),
        activate_script=(None if execution.get("activate_script") is None else resolve_path(execution["activate_script"])),
        slurm=(slurm or None),
        config_path=config_path,
    )
    if not config.model_key.strip():
        raise ValueError("model_key must be non-empty")
    if config.rank <= 0:
        raise ValueError("rank must be positive")
    if config.feature_max_iterations <= 0:
        raise ValueError("feature_max_iterations must be positive")
    if not (0.0 < config.feature_g_sparsity <= 1.0):
        raise ValueError("feature_g_sparsity must be in (0, 1]")
    if config.feature_k_proj <= 0:
        raise ValueError("feature_k_proj must be positive")
    if config.device not in {"auto", "cpu", "cuda"}:
        raise ValueError("device must be one of: auto, cpu, cuda")
    if config.fitting_device not in {"cpu", "cuda"}:
        raise ValueError("lment.features.fitting_device must be cpu or cuda")
    if config.selection_mode not in {"judge", "threshold"}:
        raise ValueError("selection_mode must be 'judge' or 'threshold'")
    if config.selection_mode == "threshold" and config.feature_ratio_threshold is None:
        raise ValueError("feature_ratio_threshold is required in threshold mode")
    if not 0.0 <= config.judge_confidence_threshold <= 1.0:
        raise ValueError("judge_confidence_threshold must be in [0, 1]")
    if config.judge_top_k <= 0:
        raise ValueError("judge_top_k must be positive")
    if config.alpaca_split not in {"train", "test"}:
        raise ValueError("alpaca_split must be 'train' or 'test'")
    if config.alpaca_max_items is not None and config.alpaca_max_items <= 0:
        raise ValueError("alpaca_max_items must be positive")
    if config.judge_max_new_tokens <= 0:
        raise ValueError("lment.judge.max_new_tokens must be positive")
    if config.selection_mode == "judge" and not config.judge_model:
        raise ValueError("lment.judge.model is required in judge selection mode")
    model_loader.pick_dtype(config.dtype)
    return config


@dataclass(frozen=True)
class DeltaSearchResult:
    chosen_delta: float
    candidates: List[Dict[str, Any]]


def _retention_fraction(edited_accuracy: float, baseline_accuracy: float) -> float:
    denominator = float(baseline_accuracy) - 0.25
    if denominator <= 0.0:
        raise ValueError(
            "Automatic delta selection requires baseline accuracy above chance (0.25)")
    return min(1.0, max(0.0, (float(edited_accuracy) - 0.25) / denominator))


def _erasure_metrics(
        baseline: Mapping[str, Mapping[str, float]],
        edited: Mapping[str, Mapping[str, float]],
) -> Dict[str, float]:
    qa_retention = _retention_fraction(
        edited["qa"]["accuracy"], baseline["qa"]["accuracy"])
    simdom_retention = _retention_fraction(
        edited["simdom"]["accuracy"], baseline["simdom"]["accuracy"])
    efficacy = 1.0 - qa_retention
    specificity = simdom_retention
    return {
        "qa_retention": qa_retention,
        "simdom_retention": simdom_retention,
        "efficacy": efficacy,
        "specificity": specificity,
        "objective": harmonic_mean([efficacy, specificity]),
    }


def search_deltas(
        *,
        deltas: Sequence[float],
        baseline: Mapping[str, Mapping[str, float]],
        restore_pristine: Callable[[], None],
        apply_delta: Callable[[float], Mapping[str, Any]],
        evaluate_train: Callable[[], Mapping[str, Mapping[str, float]]],
) -> DeltaSearchResult:
    """Evaluate a delta grid, restoring pristine embeddings before every cell."""
    if not deltas:
        raise ValueError("Delta grid is empty")
    candidates: List[Dict[str, Any]] = []
    try:
        for raw_delta in deltas:
            delta = float(raw_delta)
            restore_pristine()
            edit_info = dict(apply_delta(delta))
            evaluation = {
                key: dict(value) for key, value in evaluate_train().items()
            }
            metrics = _erasure_metrics(baseline, evaluation)
            candidates.append({
                "delta": delta,
                "edit": edit_info,
                "evaluation": evaluation,
                **metrics,
            })
    finally:
        restore_pristine()

    best = max(
        candidates,
        key=lambda candidate: (candidate["objective"], -candidate["delta"]),
    )
    return DeltaSearchResult(
        chosen_delta=float(best["delta"]),
        candidates=candidates,
    )


def _embedding_artifact_paths(
        config: LMEntRunConfig, concept: str, *, require: bool = True) -> tuple[Path, Path]:
    base = Path(config.features_root) / _safe_model_name(config.model_key)
    concept_key = _safe_concept(concept)
    common = Path(f"rank{config.rank}") / f"seed{config.seed}" / concept_key / "embedding"
    artifact_path = base / "pickles" / common / "embedding.pkl"
    potential_path = base / "interpretations" / common / "potential_features.csv"
    if require and not artifact_path.is_file():
        raise FileNotFoundError(f"Embedding artifact not found: {artifact_path}")
    if require and not potential_path.is_file():
        raise FileNotFoundError(f"Potential-features CSV not found: {potential_path}")
    return artifact_path, potential_path


def _embedding_training_paths(config: LMEntRunConfig,
                              concept: str) -> tuple[Path, Path, Path]:
    artifact, _ = _embedding_artifact_paths(config, concept, require=False)
    base = Path(config.features_root) / _safe_model_name(config.model_key)
    common = (
        Path(f"rank{config.rank}") / f"seed{config.seed}"
        / _safe_concept(concept) / "embedding")
    return (
        artifact,
        base / "csvs" / common / "stats_embed.csv",
        base / "csvs" / common / "token_features.csv",
    )


def ensure_factor_artifact(config: LMEntRunConfig, concept: str) -> None:
    """Create the embedding factorization and statistics; never invoke a judge."""
    artifact_path, stats_path, tokens_path = _embedding_training_paths(config, concept)
    missing = [path for path in (artifact_path, stats_path, tokens_path) if not path.is_file()]
    if not missing:
        return
    if config.reuse_features:
        raise FileNotFoundError(
            "Feature artifacts are missing while lment.features.reuse is true: "
            + ", ".join(str(path) for path in missing))
    if config.concept_json is None or config.neutral_json is None:
        raise ValueError(
            "concept_json and neutral_json are required to create features")
    if config.fitting_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "Feature fitting requested CUDA, but torch.cuda.is_available() is false. "
            "Set lment.features.fitting_device: cpu to run on CPU.")

    project_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (
        str(project_root),
        str(project_root / "external" / "snmf"),
        env.get("PYTHONPATH", ""),
    )))

    subprocess.run([
        sys.executable, "-m", "ember.train_mf_features",
        "--concepts", concept,
        "--ranks", str(config.rank),
        "--seed", str(config.seed),
        "--model-name", str(config.model_path),
        "--model-key", config.model_key,
        "--model-device", config.device,
        "--fitting-device", config.fitting_device,
        "--concept-json", str(config.concept_json),
        "--neutral-json", str(config.neutral_json),
        "--skip-mlp",
        "--max-iterations", str(config.feature_max_iterations),
        "--g-sparsity", str(config.feature_g_sparsity),
        "--k-proj", str(config.feature_k_proj),
        "--outdir", str(config.features_root),
    ], cwd=project_root, env=env, check=True)
    missing_after = [
        path for path in (artifact_path, stats_path, tokens_path) if not path.is_file()]
    if missing_after:
        raise RuntimeError(
            "Feature extraction completed without expected files: "
            + ", ".join(str(path) for path in missing_after))


def ensure_feature_artifacts(config: LMEntRunConfig,
                             concepts: Sequence[str]) -> None:
    """Compatibility wrapper: require factors and an existing selection CSV."""
    for concept in concepts:
        ensure_factor_artifact(config, concept)
        _embedding_artifact_paths(config, concept, require=True)


def _tensor_hash(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy()
    digest = hashlib.sha256()
    digest.update(memoryview(raw))
    return digest.hexdigest()


def _state_hashes(model: torch.nn.Module) -> Dict[str, str]:
    return {name: _tensor_hash(tensor) for name, tensor in model.state_dict().items()}


@torch.no_grad()
def _changed_embedding_rows(
        current: torch.Tensor,
        pristine: torch.Tensor,
        *,
        chunk_rows: int = 2048,
) -> List[int]:
    """Find changed rows without retaining a second full CPU embedding copy."""
    if tuple(current.shape) != tuple(pristine.shape):
        raise ValueError("Current and pristine embedding shapes do not match")
    changed: List[int] = []
    for start in range(0, int(current.shape[0]), chunk_rows):
        end = min(start + chunk_rows, int(current.shape[0]))
        current_chunk = current[start:end].detach().to(
            device="cpu", dtype=pristine.dtype)
        local_rows = torch.where(
            (current_chunk != pristine[start:end]).any(dim=1)
        )[0].tolist()
        changed.extend(start + int(row) for row in local_rows)
    return changed


def _collect_model_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _input_embedding_name(model: torch.nn.Module) -> str:
    target = model.get_input_embeddings().weight
    for name, parameter in model.named_parameters():
        if parameter is target:
            return name
    raise ValueError("Could not identify input embedding parameter name")


def _model_signature(model: torch.nn.Module) -> Dict[str, Any]:
    config = model.config
    keys = (
        "architectures", "model_type", "vocab_size", "hidden_size",
        "max_position_embeddings", "pad_token_id", "bos_token_id",
        "eos_token_id", "use_cache", "tie_word_embeddings",
    )
    return {key: getattr(config, key, None) for key in keys}


def _tokenizer_signature(tokenizer: Any) -> Dict[str, Any]:
    return {
        "size": len(tokenizer),
        "pad_token_id": tokenizer.pad_token_id,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "unk_token_id": tokenizer.unk_token_id,
        "model_max_length": tokenizer.model_max_length,
    }


@torch.no_grad()
def _probe_logits(model: torch.nn.Module, tokenizer: Any,
                  edited_token_ids: Sequence[int]) -> torch.Tensor:
    first = tokenizer.bos_token_id
    if first is None or not (0 <= int(first) < len(tokenizer)):
        first = tokenizer.eos_token_id
    if first is None:
        first = 0
    second = next((int(tid) for tid in edited_token_ids if tid < len(tokenizer)), int(first))
    device = model.get_input_embeddings().weight.device
    input_ids = torch.tensor([[int(first), second]], dtype=torch.long, device=device)
    return model(input_ids=input_ids).logits.detach().cpu()


def _load_eval_pair(config: LMEntRunConfig, concept: str,
                    split: str) -> Optional[Dict[str, Any]]:
    if config.eval_json is None:
        return None
    json_path = Path(config.eval_json)
    qa = prepare_mc_items(
        load_mc_qa_items(f"qa_{split}", concept=concept, json_path=json_path),
        seed=config.seed,
    )
    simdom = prepare_mc_items(
        load_mc_qa_items(f"simdom_{split}", concept=concept, json_path=json_path),
        seed=config.seed,
    )
    if not qa and not simdom:
        return None
    if not qa or not simdom:
        raise ValueError(
            f"Evaluation JSON must contain both QA_{split} and SimdomQA_{split} "
            f"for concept {concept!r}")
    return {"qa": qa, "simdom": simdom}


def _evaluate_pair(model: torch.nn.Module, tokenizer: Any,
                   items: Mapping[str, Any], *, include_records: bool) -> Dict[str, Any]:
    return {
        subset: evaluate_causal_mc(model, tokenizer, subset_items).to_dict(
            include_records=include_records)
        for subset, subset_items in items.items()
    }


def _metrics_or_error(baseline: Mapping[str, Mapping[str, float]],
                      edited: Mapping[str, Mapping[str, float]]) -> Dict[str, Any]:
    try:
        return _erasure_metrics(baseline, edited)
    except ValueError as error:
        return {"unavailable": str(error)}


def _paired_margin_report(
        baseline: Mapping[str, Mapping[str, Any]],
        edited: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    """Pair pristine and edited margins for the same held-out questions."""
    if set(baseline) != set(edited):
        raise ValueError("Baseline and edited evaluation subsets do not match")

    report: Dict[str, Any] = {}
    for subset in baseline:
        baseline_records = baseline[subset].get("records")
        edited_records = edited[subset].get("records")
        if baseline_records is None or edited_records is None:
            raise ValueError(f"Margin records are missing for {subset}")
        if len(baseline_records) != len(edited_records):
            raise ValueError(f"Margin record count differs for {subset}")

        records: List[Dict[str, Any]] = []
        for baseline_record, edited_record in zip(
                baseline_records, edited_records):
            identity = (
                baseline_record["question"],
                baseline_record["correct_letter"],
            )
            edited_identity = (
                edited_record["question"],
                edited_record["correct_letter"],
            )
            if identity != edited_identity:
                raise ValueError(f"Margin records are misaligned for {subset}")
            baseline_margin = float(baseline_record["margin"])
            edited_margin = float(edited_record["margin"])
            records.append({
                "question": identity[0],
                "correct_letter": identity[1],
                "baseline_margin": baseline_margin,
                "edited_margin": edited_margin,
                "delta": edited_margin - baseline_margin,
            })

        report[subset] = {
            "n": len(records),
            "baseline_mean_margin": (
                sum(record["baseline_margin"] for record in records) / len(records)
                if records else 0.0
            ),
            "edited_mean_margin": (
                sum(record["edited_margin"] for record in records) / len(records)
                if records else 0.0
            ),
            "mean_delta": (
                sum(record["delta"] for record in records) / len(records)
                if records else 0.0
            ),
            "records": records,
        }
    return report


def run_concept(config: LMEntRunConfig, *, concept: str,
                selection: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """Erase one concept from a pristine local checkpoint and export it."""
    if not concept.strip():
        raise ValueError("Concept must be non-empty")
    if config.explicit_delta is None and config.eval_json is None:
        raise ValueError("An explicit delta is required when no evaluation JSON is supplied")

    if config.output_dir is None:
        raise ValueError("LMEnt run has not been prepared with an output directory")
    output_dir = Path(config.output_dir).resolve()
    checkpoint_dir = output_dir / "model"
    source_dir = Path(config.model_path).resolve()
    if (output_dir == source_dir
            or source_dir in output_dir.parents
            or output_dir in source_dir.parents):
        raise ValueError(
            "The erased output must not overlap the source model directory: "
            f"source={source_dir}, output={output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    occupied = [
        output_dir / "report.json",
        output_dir / "erased_embeddings.safetensors",
        checkpoint_dir,
    ]
    if any(path.exists() for path in occupied):
        raise FileExistsError(
            f"Erasure output already exists under prepared run: {output_dir}")

    artifact_path, potential_path = _embedding_artifact_paths(config, concept)
    artifact = features.load_embedding_artifact(artifact_path)
    feature_ids = features.select_embed_feature_ids(
        pd.read_csv(potential_path), ratio_thresh=None)
    if not feature_ids:
        raise ValueError(f"No embedding features selected for concept {concept!r}")

    train_items = _load_eval_pair(config, concept, "train")
    test_items = _load_eval_pair(config, concept, "test")
    if config.explicit_delta is None:
        if train_items is None:
            raise ValueError(
                f"No train evaluation data found for {concept!r}; provide an explicit delta")
        if test_items is None:
            raise ValueError(
                f"Automatic delta selection for {concept!r} requires held-out test data")

    model, tokenizer = model_loader.load_local_causal_lm(
        config.model_path,
        dtype=model_loader.pick_dtype(config.dtype),
        device=config.device,
    )
    embedding_name = _input_embedding_name(model)
    actual_model_device = str(model.get_input_embeddings().weight.device)
    pristine_embedding = embed_edit.snapshot(model)
    pristine_hashes = _state_hashes(model)
    model_signature = _model_signature(model)
    tokenizer_signature = _tokenizer_signature(tokenizer)

    baseline_train = (
        _evaluate_pair(model, tokenizer, train_items, include_records=False)
        if train_items is not None else None)
    chosen_delta = (
        float(config.explicit_delta) if config.explicit_delta is not None else None)
    delta_search: Optional[DeltaSearchResult] = None
    if chosen_delta is None:
        delta_search = search_deltas(
            deltas=config.deltas,
            baseline=baseline_train,
            restore_pristine=lambda: embed_edit.restore(model, pristine_embedding),
            apply_delta=lambda delta: embed_edit.apply_embedding_artifact(
                model=model,
                tokenizer=tokenizer,
                artifact=artifact,
                feature_ids=feature_ids,
                delta=delta,
                model_key=config.model_key,
            ),
            evaluate_train=lambda: _evaluate_pair(
                model, tokenizer, train_items, include_records=False),
        )
        chosen_delta = delta_search.chosen_delta

    baseline_test = (
        _evaluate_pair(model, tokenizer, test_items, include_records=True)
        if test_items is not None else None)

    embed_edit.restore(model, pristine_embedding)
    edit_info = embed_edit.apply_embedding_artifact(
        model=model,
        tokenizer=tokenizer,
        artifact=artifact,
        feature_ids=feature_ids,
        delta=chosen_delta,
        model_key=config.model_key,
    )
    edited_train = None
    if train_items is not None:
        if delta_search is not None:
            selected = next(
                candidate for candidate in delta_search.candidates
                if candidate["delta"] == chosen_delta)
            edited_train = selected["evaluation"]
        else:
            edited_train = _evaluate_pair(
                model, tokenizer, train_items, include_records=False)
    edited_test = (
        _evaluate_pair(model, tokenizer, test_items, include_records=True)
        if test_items is not None else None)
    changed_rows = _changed_embedding_rows(
        model.get_input_embeddings().weight, pristine_embedding)
    edited_hashes = _state_hashes(model)
    changed_tensors = sorted(
        name for name in pristine_hashes if pristine_hashes[name] != edited_hashes[name])
    isolation_passed = (
        set(changed_tensors).issubset({embedding_name})
        and set(changed_rows).issubset(set(edit_info["edited_token_ids"]))
    )

    final_logits = _probe_logits(model, tokenizer, edit_info["edited_token_ids"])
    final_hashes = edited_hashes
    checkpoint_path: Optional[Path] = None
    embedding_artifact_path: Optional[Path] = None
    if config.full_save:
        checkpoint_dir.mkdir(parents=True, exist_ok=False)
        model.save_pretrained(checkpoint_dir, safe_serialization=True)
        tokenizer.save_pretrained(checkpoint_dir)
        checkpoint_path = checkpoint_dir
    else:
        embedding_artifact_path = save_erased_embedding(
            model=model,
            output_dir=output_dir,
            base_model_path=source_dir,
            model_key=config.model_key,
            tensor_name=embedding_name,
            base_embedding_sha256=pristine_hashes[embedding_name],
            edited_token_ids=edit_info["edited_token_ids"],
        )

    del pristine_embedding
    del model
    _collect_model_memory()

    if config.full_save:
        reloaded, reloaded_tokenizer = model_loader.load_local_causal_lm(
            checkpoint_dir,
            dtype=model_loader.pick_dtype(config.dtype),
            device=config.device,
        )
    else:
        reloaded, reloaded_tokenizer = load_lment_with_erased_embeddings(
            source_dir,
            embedding_artifact_path,
            dtype=model_loader.pick_dtype(config.dtype),
            device=config.device,
        )
    reloaded_hashes = _state_hashes(reloaded)
    reload_state_match = reloaded_hashes == final_hashes
    reload_logits = _probe_logits(
        reloaded, reloaded_tokenizer, edit_info["edited_token_ids"])
    reload_logits_match = torch.equal(reload_logits, final_logits)
    config_preserved = _model_signature(reloaded) == model_signature
    tokenizer_preserved = _tokenizer_signature(reloaded_tokenizer) == tokenizer_signature
    integrity_passed = all((
        isolation_passed,
        reload_state_match,
        reload_logits_match,
        config_preserved,
        tokenizer_preserved,
    ))

    report: Dict[str, Any] = {
        "schema_version": 1,
        "model_path": str(Path(config.model_path)),
        "model_key": config.model_key,
        "concept": concept,
        "rank": config.rank,
        "seed": config.seed,
        "ratio_thresh": config.ratio_thresh,
        "artifact": {
            "path": str(artifact_path),
            "version": artifact.version,
            "selected_feature_ids": feature_ids,
        },
        "feature_selection": dict(selection) if selection is not None else None,
        "chosen_delta": chosen_delta,
        "delta_search": (None if delta_search is None else {
            "chosen_delta": delta_search.chosen_delta,
            "candidates": delta_search.candidates,
        }),
        "edit": edit_info,
        "devices": {
            "requested_model_device": config.device,
            "actual_model_device": actual_model_device,
            "requested_fitting_device": config.fitting_device,
        },
        "evaluation": ({
            "train": {
                "baseline": baseline_train,
                "edited": edited_train,
                "metrics": _metrics_or_error(baseline_train, edited_train),
            } if baseline_train is not None and edited_train is not None else None,
            "test": {
                "baseline": baseline_test,
                "edited": edited_test,
                "metrics": _metrics_or_error(baseline_test, edited_test),
                "paired_margins": _paired_margin_report(baseline_test, edited_test),
            } if baseline_test is not None and edited_test is not None else None,
        } if baseline_train is not None or baseline_test is not None else None),
        "save": {
            "mode": "full_model" if config.full_save else "embedding_only",
            "output_dir": str(output_dir),
            "checkpoint_path": (
                str(checkpoint_path) if checkpoint_path is not None else None),
            "erased_embeddings_path": (
                str(embedding_artifact_path)
                if embedding_artifact_path is not None else None),
        },
        "checkpoint_path": (
            str(checkpoint_path) if checkpoint_path is not None else None),
        "erased_embeddings_path": (
            str(embedding_artifact_path)
            if embedding_artifact_path is not None else None),
        "integrity": {
            "passed": integrity_passed,
            "input_embedding_name": embedding_name,
            "changed_tensor_names": changed_tensors,
            "changed_embedding_rows": changed_rows,
            "reload_state_match": reload_state_match,
            "reload_logits_match": reload_logits_match,
            "config_preserved": config_preserved,
            "tokenizer_preserved": tokenizer_preserved,
            "pristine_embedding_sha256": pristine_hashes[embedding_name],
            "edited_embedding_sha256": edited_hashes[embedding_name],
        },
    }
    io.save_json_atomic(output_dir / "report.json", report)

    del reloaded
    _collect_model_memory()
    if not integrity_passed:
        raise RuntimeError(f"Export integrity audit failed for concept {concept!r}")
    return report


def run_lment_pipeline(
        config: LMEntRunConfig,
        *,
        concept: str,
        describe_callback: Optional[TextCallback] = None,
        classify_callback: Optional[TextCallback] = None,
        alpaca_relevance_callback: Optional[TextCallback] = None,
        alpaca_fluency_callback: Optional[TextCallback] = None,
) -> Dict[str, Any]:
    """Prepare/select features, erase one concept, export, and optionally evaluate."""
    ensure_factor_artifact(config, concept)
    selection: FeatureSelectionResult
    if config.selection_mode == "threshold":
        if config.feature_ratio_threshold is None:
            raise ValueError("feature_ratio_threshold is required in threshold mode")
        selection = select_by_threshold(
            features_root=config.features_root,
            model_key=config.model_key,
            concept=concept,
            rank=config.rank,
            seed=config.seed,
            threshold=config.feature_ratio_threshold,
        )
    else:
        selection = select_with_judge(
            features_root=config.features_root,
            model_key=config.model_key,
            concept=concept,
            rank=config.rank,
            seed=config.seed,
            prefilter_threshold=config.ratio_thresh,
            confidence_threshold=config.judge_confidence_threshold,
            top_k=config.judge_top_k,
            describe_callback=describe_callback or judge_callbacks.describe_feature,
            classify_callback=classify_callback or judge_callbacks.classify_feature,
        )

    report = run_concept(
        config, concept=concept, selection=selection.to_dict())
    if not config.alpaca_eval:
        return report

    evaluation_dtype = model_loader.pick_dtype(config.dtype)
    if config.full_save:
        evaluation_model, evaluation_tokenizer = model_loader.load_local_causal_lm(
            report["checkpoint_path"], dtype=evaluation_dtype, device="cuda")
    else:
        evaluation_model, evaluation_tokenizer = load_lment_with_erased_embeddings(
            config.model_path,
            report["erased_embeddings_path"],
            dtype=evaluation_dtype,
            device="cuda",
        )
    try:
        from ember.evals.lment_alpaca import evaluate_lment_alpaca
        alpaca_report = evaluate_lment_alpaca(
            model=evaluation_model,
            tokenizer=evaluation_tokenizer,
            split=config.alpaca_split,
            relevance_callback=(
                alpaca_relevance_callback or judge_callbacks.score_alpaca_relevance),
            fluency_callback=(
                alpaca_fluency_callback or judge_callbacks.score_alpaca_fluency),
            max_items=config.alpaca_max_items,
        )
    finally:
        del evaluation_model
        del evaluation_tokenizer
        _collect_model_memory()
    report["alpaca"] = alpaca_report
    io.save_json_atomic(Path(report["save"]["output_dir"]) / "report.json", report)
    return report


__all__ = [
    "DEFAULT_DELTAS", "LMEntRunConfig", "DeltaSearchResult",
    "load_lment_config", "ensure_factor_artifact", "ensure_feature_artifacts",
    "search_deltas", "run_concept", "run_lment_pipeline",
]
