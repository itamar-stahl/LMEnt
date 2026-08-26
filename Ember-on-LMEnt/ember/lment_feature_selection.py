"""Strict threshold and callback-based feature selection for LMEnt EMBER."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List

import pandas as pd

from ember.interpret_features import (
    MEMBERSHIP_PROMPT,
    TOKENS_DESC_SYS,
    TOKENS_DESC_USER,
    _parse_list_cell,
)
from ember.utils import _safe_concept, _safe_model_name


TextCallback = Callable[[str], str]


@dataclass(frozen=True)
class FeatureSelectionResult:
    mode: str
    threshold: float
    selected_feature_ids: List[int]
    potential_features_path: str
    judge_trace_path: str | None = None

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _paths(features_root: Path, model_key: str, concept: str,
           rank: int, seed: int) -> tuple[Path, Path, Path]:
    common = (
        Path(f"rank{rank}") / f"seed{seed}" / _safe_concept(concept) / "embedding")
    base = Path(features_root) / _safe_model_name(model_key)
    return (
        base / "csvs" / common / "stats_embed.csv",
        base / "csvs" / common / "token_features.csv",
        base / "interpretations" / common,
    )


def _eligible_stats(stats_path: Path, threshold: float) -> pd.DataFrame:
    if not stats_path.is_file():
        raise FileNotFoundError(f"Embedding statistics not found: {stats_path}")
    stats = pd.read_csv(stats_path)
    required = {"feature", "ratio_abs"}
    missing = required - set(stats.columns)
    if missing:
        raise ValueError(f"{stats_path} is missing columns: {sorted(missing)}")
    ratios = pd.to_numeric(stats["ratio_abs"], errors="coerce")
    selected = stats[ratios.notna() & (ratios >= float(threshold))].copy()
    if selected.empty:
        raise ValueError(
            f"No embedding feature has ratio_abs >= {float(threshold):g}; "
            "lower the threshold or use an LLM judge.")
    selected["feature"] = pd.to_numeric(selected["feature"], errors="raise").astype(int)
    selected["metric_score"] = pd.to_numeric(
        selected["ratio_abs"], errors="raise").astype(float)
    return selected.sort_values("feature").drop_duplicates("feature")


def select_by_threshold(*, features_root: Path, model_key: str, concept: str,
                        rank: int, seed: int, threshold: float) -> FeatureSelectionResult:
    """Select every embedding feature whose ``ratio_abs`` is at least p."""
    stats_path, _, interpretation_dir = _paths(
        features_root, model_key, concept, rank, seed)
    selected = _eligible_stats(stats_path, threshold)
    output = interpretation_dir / "potential_features.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        column for column in (
            "feature", "metric_score", "mean_abs_concept", "mean_abs_neutral",
            "num_concept", "num_neutral", "num_both",
        ) if column in selected.columns
    ]
    payload = selected[columns].copy()
    payload["selection_mode"] = "threshold"
    payload.to_csv(output, index=False)
    return FeatureSelectionResult(
        mode="threshold",
        threshold=float(threshold),
        selected_feature_ids=payload["feature"].astype(int).tolist(),
        potential_features_path=str(output.resolve()),
    )


def _parse_membership(response: str) -> tuple[bool, float]:
    try:
        payload = json.loads(response)
    except json.JSONDecodeError as error:
        raise ValueError(f"Feature classifier returned invalid JSON: {response!r}") from error
    if not isinstance(payload, dict):
        raise ValueError("Feature classifier response must be a JSON object")
    is_member = payload.get("is_member")
    confidence = payload.get("confidence")
    if not isinstance(is_member, bool):
        raise ValueError("Feature classifier 'is_member' must be a JSON boolean")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError("Feature classifier 'confidence' must be a number")
    confidence = float(confidence)
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("Feature classifier 'confidence' must be in [0, 1]")
    return is_member, confidence


def select_with_judge(*, features_root: Path, model_key: str, concept: str,
                      rank: int, seed: int, prefilter_threshold: float,
                      confidence_threshold: float, top_k: int,
                      describe_callback: TextCallback,
                      classify_callback: TextCallback) -> FeatureSelectionResult:
    """Run the documented two-prompt callback flow and save selected features."""
    stats_path, tokens_path, interpretation_dir = _paths(
        features_root, model_key, concept, rank, seed)
    eligible = _eligible_stats(stats_path, prefilter_threshold)
    if not tokens_path.is_file():
        raise FileNotFoundError(f"Embedding token features not found: {tokens_path}")
    tokens = pd.read_csv(tokens_path)
    merged = eligible.merge(tokens, on="feature", how="inner", suffixes=("", "_tokens"))
    if merged.empty:
        raise ValueError("No token rows match the judge-prefiltered embedding features")

    traces: List[Dict[str, object]] = []
    selected_rows: List[Dict[str, object]] = []
    for _, row in merged.sort_values("feature").iterrows():
        feature_id = int(row["feature"])
        feature_tokens = _parse_list_cell(row.get("activating_tokens"))[:int(top_k)]
        if not feature_tokens:
            raise ValueError(f"Feature {feature_id} has no activation tokens for judging")
        description_prompt = (
            f"{TOKENS_DESC_SYS.strip()}\n\n"
            f"{TOKENS_DESC_USER.format(tokens=feature_tokens).strip()}")
        description = describe_callback(description_prompt)
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"Description callback returned no text for feature {feature_id}")
        description = description.strip()
        classification_prompt = MEMBERSHIP_PROMPT.format(
            concept=concept, desc=description).strip()
        classification_response = classify_callback(classification_prompt)
        if not isinstance(classification_response, str):
            raise ValueError("Classification callback must return a JSON string")
        is_member, confidence = _parse_membership(classification_response)
        trace = {
            "feature": feature_id,
            "description_prompt": description_prompt,
            "description_response": description,
            "classification_prompt": classification_prompt,
            "classification_response": classification_response,
            "is_member": is_member,
            "confidence": confidence,
        }
        traces.append(trace)
        if is_member and confidence >= float(confidence_threshold):
            selected_rows.append({
                "feature": feature_id,
                "metric_score": float(row["metric_score"]),
                "description": description,
                "is_member": True,
                "confidence": confidence,
                "selection_mode": "judge",
            })

    interpretation_dir.mkdir(parents=True, exist_ok=True)
    trace_path = interpretation_dir / "judge_trace.json"
    trace_path.write_text(json.dumps(traces, indent=2, ensure_ascii=False), encoding="utf-8")
    if not selected_rows:
        raise ValueError(
            "The feature judge selected no features above the confidence threshold; "
            f"trace saved to {trace_path}")
    output = interpretation_dir / "potential_features.csv"
    pd.DataFrame(selected_rows).to_csv(output, index=False)
    return FeatureSelectionResult(
        mode="judge",
        threshold=float(prefilter_threshold),
        selected_feature_ids=sorted(row["feature"] for row in selected_rows),
        potential_features_path=str(output.resolve()),
        judge_trace_path=str(trace_path.resolve()),
    )


__all__ = [
    "FeatureSelectionResult", "TextCallback", "select_by_threshold",
    "select_with_judge",
]
