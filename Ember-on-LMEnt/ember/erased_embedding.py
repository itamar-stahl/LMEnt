"""Save and load LMEnt models as a base checkpoint plus one erased embedding."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from ember.erasure.model_loader import load_local_causal_lm


ARTIFACT_FILENAME = "erased_embeddings.safetensors"
SCHEMA_VERSION = 1


def tensor_sha256(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy()
    digest = hashlib.sha256()
    digest.update(memoryview(raw))
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_sha256(model_path: Path) -> str:
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Base model config not found: {config_path}")
    return _file_sha256(config_path)


def _source_tensor(model_path: Path, tensor_name: str) -> torch.Tensor:
    index_path = model_path / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        shard_name = index.get("weight_map", {}).get(tensor_name)
        if not shard_name:
            raise ValueError(f"Base checkpoint index has no tensor {tensor_name!r}")
        tensor_file = model_path / shard_name
    else:
        tensor_file = model_path / "model.safetensors"
    if not tensor_file.is_file():
        raise FileNotFoundError(f"Base model tensor file not found: {tensor_file}")
    with safe_open(tensor_file, framework="pt", device="cpu") as handle:
        if tensor_name not in handle.keys():
            raise ValueError(f"Base checkpoint has no tensor {tensor_name!r}")
        return handle.get_tensor(tensor_name)


def save_erased_embedding(*, model: torch.nn.Module, output_dir: Path,
                          base_model_path: Path, model_key: str,
                          tensor_name: str, base_embedding_sha256: str,
                          edited_token_ids: Sequence[int]) -> Path:
    """Write only the edited input-embedding tensor with strict base metadata."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    base_model_path = Path(base_model_path).resolve()
    weight = model.get_input_embeddings().weight.detach().cpu().contiguous()
    metadata = {
        "schema_version": str(SCHEMA_VERSION),
        "model_key": str(model_key),
        "base_model_path": str(base_model_path),
        "base_config_sha256": _config_sha256(base_model_path),
        "base_embedding_sha256": str(base_embedding_sha256),
        "erased_embedding_sha256": tensor_sha256(weight),
        "tensor_name": str(tensor_name),
        "shape": json.dumps(list(weight.shape)),
        "dtype": str(weight.dtype),
        "edited_token_ids": json.dumps([int(token_id) for token_id in edited_token_ids]),
    }
    artifact_path = output_dir / ARTIFACT_FILENAME
    save_file({tensor_name: weight}, artifact_path, metadata=metadata)
    return artifact_path


def read_embedding_metadata(erased_embeddings_path: str | Path) -> Dict[str, str]:
    path = Path(erased_embeddings_path)
    with safe_open(path, framework="pt", device="cpu") as handle:
        return dict(handle.metadata() or {})


def load_lment_with_erased_embeddings(
        base_model_path: str | Path,
        erased_embeddings_path: str | Path,
        *,
        device: str = "auto",
        dtype: torch.dtype = torch.float32,
) -> Tuple[Any, Any]:
    """Load a pristine local LMEnt checkpoint, validate, then replace W_E in memory."""
    base_path = Path(base_model_path).resolve()
    artifact_path = Path(erased_embeddings_path).resolve()
    if not artifact_path.is_file():
        raise FileNotFoundError(f"Erased embedding artifact not found: {artifact_path}")

    with safe_open(artifact_path, framework="pt", device="cpu") as handle:
        metadata = dict(handle.metadata() or {})
        if metadata.get("schema_version") != str(SCHEMA_VERSION):
            raise ValueError(
                f"Unsupported erased embedding schema: {metadata.get('schema_version')!r}")
        tensor_name = metadata.get("tensor_name", "")
        keys = list(handle.keys())
        if keys != [tensor_name]:
            raise ValueError(
                f"Erased embedding artifact must contain only {tensor_name!r}; got {keys}")
        erased_weight = handle.get_tensor(tensor_name)

    if metadata.get("base_config_sha256") != _config_sha256(base_path):
        raise ValueError("Erased embedding artifact does not match the base model config")
    source_weight = _source_tensor(base_path, tensor_name)
    if tensor_sha256(source_weight) != metadata.get("base_embedding_sha256"):
        raise ValueError("Erased embedding artifact does not match the base embedding weights")
    if list(erased_weight.shape) != json.loads(metadata.get("shape", "[]")):
        raise ValueError("Erased embedding tensor shape does not match its metadata")
    if tensor_sha256(erased_weight) != metadata.get("erased_embedding_sha256"):
        raise ValueError("Erased embedding tensor hash does not match its metadata")

    model, tokenizer = load_local_causal_lm(base_path, dtype=dtype, device=device)
    target = model.get_input_embeddings().weight
    if tuple(target.shape) != tuple(erased_weight.shape):
        raise ValueError(
            f"Erased embedding shape {tuple(erased_weight.shape)} does not match "
            f"base model shape {tuple(target.shape)}")
    with torch.no_grad():
        target.copy_(erased_weight.to(device=target.device, dtype=target.dtype))
    model.eval()
    return model, tokenizer


__all__ = [
    "ARTIFACT_FILENAME", "SCHEMA_VERSION", "load_lment_with_erased_embeddings",
    "read_embedding_metadata", "save_erased_embedding", "tensor_sha256",
]
