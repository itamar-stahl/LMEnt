"""Embedding-space erasure primitives.

Subtracts the SparseMatrixFactorization concept contribution from each
eligible concept token's embedding:

    c_i = F'[:, k'] @ G'[i, k']   (concept contribution for token i)
    e_i_new = e_i - delta * c_i

F (shape [d_model, K]) is the dense set of erasure directions;
G (shape [|V'|, K]) is the WTA-sparse signed per-token activation scores.
Only tokens labeled as concept-only are edited (intersection with nonzero G).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from ember.utils import _safe_tokens, get_special_token_ids
from ember.erasure import features, log
from ember.erasure.features import EmbeddingArtifact


# ========================================================================== #
# Embedding weight helpers                                                    #
# ========================================================================== #

def _embedding_weight(model: torch.nn.Module) -> torch.Tensor:
    emb = model.get_input_embeddings()
    if emb is None or getattr(emb, "weight", None) is None:
        raise ValueError("HF model has no input embedding weight.")
    return emb.weight


def snapshot(model: torch.nn.Module) -> torch.Tensor:
    """Return a CPU copy of the input-embedding weight tensor."""
    return _embedding_weight(model).data.detach().cpu().clone()


def restore(model: torch.nn.Module, snap: torch.Tensor) -> None:
    """Overwrite the model's input-embedding weight from a saved snapshot."""
    W = _embedding_weight(model)
    with torch.no_grad():
        W.data.copy_(snap.to(device=W.device, dtype=W.dtype))


def _embedding_scale(model: torch.nn.Module, model_name: str) -> float:
    """Read-side scale to bring W_E into the factorization's scaled space.

    HF Gemma stores W_E unscaled; the forward pass applies sqrt(d_model) at
    runtime. We multiply by sqrt(d_model) on read and divide on write so the
    math operates in the same scaled space as the factorization.
    Llama has no embedding scale, so returns 1.0.
    """
    if "gemma" not in model_name.lower():
        return 1.0
    W = _embedding_weight(model)
    return float(np.sqrt(int(W.shape[1])))


# ========================================================================== #
# Concept token set                                                           #
# ========================================================================== #

def _build_concept_token_set(model_name: str, concept_name: str,
                              rank: int, seed: int,
                              vprime_token_ids: List[int],
                              tokenizer: Any = None) -> Optional[set]:
    """Set of token-ids labeled exactly as concept_name in the SNMF token CSV.

    Returns None when the token-label CSV is missing.
    """
    str_to_label = features.load_token_label_map(model_name, concept_name, rank, seed)
    if not str_to_label:
        return None

    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    hf_strs = tokenizer.convert_ids_to_tokens(vprime_token_ids)
    cleaned = [
        s.replace("▁", " ").replace("Ġ", " ") if isinstance(s, str) else ""
        for s in hf_strs
    ]
    id_to_safe = {tid: s for tid, s in zip(vprime_token_ids, _safe_tokens(cleaned))}
    return {tid for tid in vprime_token_ids
            if str_to_label.get(id_to_safe.get(tid, ""), "") == concept_name}


# ========================================================================== #
# Core erasure                                                                #
# ========================================================================== #

def _eligible_tids_rows(vprime_token_ids: List[int], G_prime: torch.Tensor,
                         concept_token_ids: set) -> Tuple[List[int], List[int]]:
    """Eligible (token_id, row) pairs for the edit.

    A token is eligible when it is a concept token, lives in V', and has a
    nonzero G value on at least one selected feature. ``G_prime`` is G already
    restricted to the selected feature columns (shape ``[|V'|, k']``).
    """
    tid_to_row = {tid: i for i, tid in enumerate(vprime_token_ids)}
    tids: List[int] = []
    rows: List[int] = []
    for tid in sorted(concept_token_ids):
        if tid not in tid_to_row:
            continue
        row = tid_to_row[tid]
        if float(G_prime[row, :].abs().max()) > 0.0:
            tids.append(tid)
            rows.append(row)
    return tids, rows


def _erase_factored(
        hf_model: torch.nn.Module,
        model_name: str,
        F_dir: torch.Tensor,
        G_tok: torch.Tensor,
        vprime_token_ids: List[int],
        feature_ids: List[int],
        delta_embed: float,
        concept_token_ids: set,
) -> List[int]:
    """Apply the factored embedding edit. Returns edited token IDs.

    For each eligible token i (concept-only ∩ in V' ∩ nonzero G across the
    selected features):

        c_i = F'[:, feature_ids] @ G_tok[row_i, feature_ids]
        e_i ← e_i - delta * c_i
    """
    if not concept_token_ids or not feature_ids:
        return []

    W_E = _embedding_weight(hf_model)
    device, dtype = W_E.device, W_E.dtype
    scale = _embedding_scale(hf_model, model_name)

    F_prime = torch.as_tensor(F_dir[:, feature_ids]).to(device=device, dtype=dtype)  # [d, k']
    G_prime = torch.as_tensor(G_tok[:, feature_ids])                                  # [|V'|, k'] cpu

    eligible_tids, eligible_rows = _eligible_tids_rows(
        vprime_token_ids, G_prime, concept_token_ids)
    if not eligible_tids:
        return []

    tids = torch.tensor(eligible_tids, device=device, dtype=torch.long)
    G_rows = G_prime[eligible_rows, :].to(device=device, dtype=dtype)  # [n_tok, k']
    C = (F_prime @ G_rows.T).T                                          # [n_tok, d]

    with torch.no_grad():
        E = W_E.data.index_select(0, tids) * scale
        W_E.data.index_copy_(0, tids, (E - float(delta_embed) * C) / scale)

    return eligible_tids


def _validate_artifact_for_model(
        model: torch.nn.Module,
        tokenizer: Any,
        artifact: EmbeddingArtifact,
        model_key: Optional[str],
) -> None:
    W_E = _embedding_weight(model)
    if artifact.F_dir.shape[0] != W_E.shape[1]:
        raise ValueError(
            f"Artifact hidden size {artifact.F_dir.shape[0]} does not match "
            f"model hidden size {W_E.shape[1]}")
    if artifact.G_tok.shape[0] != len(artifact.vprime_token_ids):
        raise ValueError("Artifact G rows do not match vprime_token_ids")
    if artifact.version < 2 or artifact.token_roles is None:
        raise ValueError("ID-native editing requires a version 2 embedding artifact")
    if model_key is not None and artifact.model_key != model_key:
        raise ValueError(
            f"Artifact model_key {artifact.model_key!r} does not match {model_key!r}")
    if artifact.tokenizer_size != len(tokenizer):
        raise ValueError(
            f"Artifact tokenizer_size {artifact.tokenizer_size} does not match {len(tokenizer)}")
    if artifact.embedding_vocab_size != int(W_E.shape[0]):
        raise ValueError(
            f"Artifact embedding_vocab_size {artifact.embedding_vocab_size} "
            f"does not match {W_E.shape[0]}")
    if artifact.embedding_dim != int(W_E.shape[1]):
        raise ValueError(
            f"Artifact embedding_dim {artifact.embedding_dim} does not match {W_E.shape[1]}")


def apply_embedding_artifact(
        *,
        model: torch.nn.Module,
        tokenizer: Any,
        artifact: EmbeddingArtifact,
        feature_ids: List[int],
        delta: float,
        model_key: Optional[str] = None,
) -> Dict[str, Any]:
    """Apply one validated, ID-native EMBER embedding artifact."""
    _validate_artifact_for_model(model, tokenizer, artifact, model_key)
    feature_ids = sorted({int(feature_id) for feature_id in feature_ids})
    rank = int(artifact.F_dir.shape[1])
    if any(feature_id < 0 or feature_id >= rank for feature_id in feature_ids):
        raise ValueError(f"Feature IDs must be in [0, {rank}), got {feature_ids}")

    if float(delta) == 0.0 or not feature_ids:
        return {
            "delta_embed": float(delta),
            "k_features_embed": len(feature_ids),
            "n_tokens_edited": 0,
            "edited_token_ids": [],
        }

    W_E = _embedding_weight(model)
    special_ids = get_special_token_ids(tokenizer)
    concept_token_ids = {
        tid for tid in artifact.concept_token_ids
        if (0 <= tid < len(tokenizer)
            and tid < W_E.shape[0]
            and tid not in special_ids)
    }
    edited_token_ids = _erase_factored(
        hf_model=model,
        model_name=model_key or artifact.model_key or "",
        F_dir=artifact.F_dir,
        G_tok=artifact.G_tok,
        vprime_token_ids=artifact.vprime_token_ids,
        feature_ids=feature_ids,
        delta_embed=float(delta),
        concept_token_ids=concept_token_ids,
    )
    return {
        "delta_embed": float(delta),
        "k_features_embed": len(feature_ids),
        "n_tokens_edited": len(edited_token_ids),
        "edited_token_ids": edited_token_ids,
    }


# ========================================================================== #
# Public API                                                                  #
# ========================================================================== #

def apply_concept_embed_edit_factored(
        hf_model: torch.nn.Module,
        model_name: str,
        concept_name: str,
        delta_embed: float,
        rank: int,
        seed: int,
        ratio_thresh: Optional[float] = 2.0,
        tokenizer: Any = None,
) -> Dict[str, Any]:
    """Apply the factored embedding edit for one concept.

    Returns a dict with keys delta_embed, k_features_embed, n_tokens_edited.
    delta_embed=0.0 is a no-op and returns zeros.
    """
    if float(delta_embed) == 0.0:
        return {"delta_embed": 0.0, "k_features_embed": 0,
                "n_tokens_edited": 0, "edited_token_ids": []}

    pkl_path, pot_csv = features._embedding_paths(model_name, concept_name, rank, seed)
    artifact = features.load_embedding_artifact(pkl_path)
    feat_ids = features.select_embed_feature_ids(pd.read_csv(pot_csv),
                                                 ratio_thresh=ratio_thresh)

    if artifact.token_roles is not None:
        if tokenizer is None:
            raise ValueError("A loaded tokenizer is required for ID-native embedding artifacts")
        return apply_embedding_artifact(
            model=hf_model,
            tokenizer=tokenizer,
            artifact=artifact,
            feature_ids=feat_ids,
            delta=float(delta_embed),
            model_key=model_name,
        )

    concept_token_ids = _build_concept_token_set(
        model_name, concept_name, rank, seed, artifact.vprime_token_ids, tokenizer
    )
    if concept_token_ids is None:
        log.warning("embed edit: no token CSV for %r; skipping", concept_name)
        return {"delta_embed": float(delta_embed),
                "k_features_embed": int(len(feat_ids)),
                "n_tokens_edited": 0,
                "edited_token_ids": []}
    if not concept_token_ids:
        log.warning("embed edit: empty concept set for %r; skipping", concept_name)
        return {"delta_embed": float(delta_embed),
                "k_features_embed": int(len(feat_ids)),
                "n_tokens_edited": 0,
                "edited_token_ids": []}

    edited_token_ids = _erase_factored(
        hf_model=hf_model,
        model_name=model_name,
        F_dir=artifact.F_dir,
        G_tok=artifact.G_tok,
        vprime_token_ids=artifact.vprime_token_ids,
        feature_ids=feat_ids,
        delta_embed=float(delta_embed),
        concept_token_ids=concept_token_ids,
    )
    return {
        "delta_embed": float(delta_embed),
        "k_features_embed": int(len(feat_ids)),
        "n_tokens_edited": len(edited_token_ids),
        "edited_token_ids": edited_token_ids,
    }


def concept_edit_tokens(model_name: str, concept_name: str, rank: int, seed: int,
                        ratio_thresh: Optional[float] = 2.0,
                        tokenizer: Any = None) -> List[Tuple[str, float]]:
    """``(token, edit_magnitude)`` pairs for the tokens the EMBER edit modifies.

    Same eligibility as the edit itself (concept tokens in V' with nonzero G on a
    selected feature), so this is exactly the set of edited tokens. The magnitude
    is ``||c_i||`` where ``c_i = F'[:, feat] @ G'[i, feat]`` is the vector
    subtracted from token i's embedding (delta and the embedding scale are
    constant across tokens, so this norm ranks how strongly each token is edited).
    Returned sorted from most to least edited. Empty when the token-label CSV or
    concept set is missing.
    """
    pkl_path, pot_csv = features._embedding_paths(model_name, concept_name, rank, seed)
    artifact = features.load_embedding_artifact(pkl_path)
    F_dir, G_tok, vprime_ids = (
        artifact.F_dir, artifact.G_tok, artifact.vprime_token_ids)
    feat_ids = features.select_embed_feature_ids(pd.read_csv(pot_csv),
                                                 ratio_thresh=ratio_thresh)
    concept_token_ids = (artifact.concept_token_ids if artifact.token_roles is not None
                         else _build_concept_token_set(
                             model_name, concept_name, rank, seed, vprime_ids, tokenizer))
    if not concept_token_ids or not feat_ids:
        return []

    G_prime = torch.as_tensor(G_tok[:, feat_ids])
    tids, rows = _eligible_tids_rows(vprime_ids, G_prime, concept_token_ids)
    if not tids:
        return []

    F_prime = torch.as_tensor(F_dir[:, feat_ids]).float()       # [d, k']
    G_rows = G_prime[rows, :].float()                            # [n, k']
    mags = (F_prime @ G_rows.T).norm(dim=0)                      # [n] = ||c_i||
    order = torch.argsort(mags, descending=True).tolist()

    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    toks = tokenizer.convert_ids_to_tokens([tids[i] for i in order])
    cleaned = [t.replace("▁", " ").replace("Ġ", " ") if isinstance(t, str) else ""
               for t in toks]
    return [(tok, float(mags[i])) for tok, i in zip(cleaned, order)]


__all__ = [
    "snapshot", "restore",
    "apply_embedding_artifact",
    "apply_concept_embed_edit_factored",
    "concept_edit_tokens",
]
