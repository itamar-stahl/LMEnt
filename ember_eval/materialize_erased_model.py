#!/usr/bin/env python
"""Turn EMBER's embedding-only erasure output into a full, scorable model dir.

`run_lment_ember` with `save.full_model: false` writes just
`erased_embeddings.safetensors` -- one tensor, `model.embed_tokens.weight`. Every
instrument in this directory (`completion_eval/evaluate_completion.py`,
`heldout_ppl/`, `cross_concept_null.py`) takes `--model <dir>` and loads a whole
HF model, so the erased weights have to be written back beside the 200 tensors
that did not change.

WHY THIS IS NOT A COPY WITH A FILE SWAPPED
------------------------------------------
`model.embed_tokens.weight` shares shard 1 with 199 other tensors, so the shard
has to be rewritten rather than replaced. And `lm_head.weight` is a SEPARATE,
UNTIED tensor in shard 2: EMBER edits the input embedding only, so the output
head must be carried across untouched. Getting that wrong would silently change
what is being measured.

WHAT IS VERIFIED, AND WHY EACH CHECK EARNS ITS PLACE
----------------------------------------------------
The erasure artifact carries provenance in its safetensors metadata, and all of
it is checked before anything is written:

  base_config_sha256      the erasure ran against THIS control, not the retired
                          Pornography pair's `lment-1b-control-2e`. The two
                          differ and are not weight-comparable.
  base_embedding_sha256   the base's embedding is bit-identical to the one the
                          erasure started from -- so the edit is being replayed
                          onto the same weights, not onto a drifted copy.
  erased_embedding_sha256 the tensor we are about to write is the one the run
                          produced.

Afterwards the written model is re-read and compared against the base, and the
set of differing rows must equal `edited_token_ids` EXACTLY -- not merely have
the right size. That is the check that catches a mis-scattered row, and it is
the reason to prefer this over `cp` plus a hand edit.

    python ember_eval/materialize_erased_model.py \\
        --base /home/dcor/galbarak2/hf-models/lment-1b-control-2e-b131k \\
        --erased .../outputs/erased_embeddings.safetensors \\
        --out /home/dcor/galbarak2/hf-models/lment-1b-rome-erased-b131k
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

TENSOR = "model.embed_tokens.weight"


def tensor_sha256(t: torch.Tensor) -> str:
    """Byte-identical to ember.lment_pipeline._tensor_hash."""
    raw = t.detach().cpu().contiguous().view(torch.uint8).numpy()
    d = hashlib.sha256()
    d.update(memoryview(raw))
    return d.hexdigest()


def file_sha256(p: Path) -> str:
    d = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            d.update(chunk)
    return d.hexdigest()


def die(msg: str) -> None:
    raise SystemExit(f"REFUSING: {msg}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="the pristine control model dir")
    ap.add_argument("--erased", required=True, help="erased_embeddings.safetensors")
    ap.add_argument("--out", required=True)
    ap.add_argument("--force", action="store_true", help="overwrite an existing --out")
    args = ap.parse_args()

    base, erased_path, out = Path(args.base), Path(args.erased), Path(args.out)
    if out.exists() and not args.force:
        die(f"{out} already exists (pass --force to replace it)")

    with safe_open(str(erased_path), "pt") as fh:
        meta = fh.metadata() or {}
        keys = list(fh.keys())
    if keys != [TENSOR]:
        die(f"{erased_path} holds {keys}, expected exactly [{TENSOR!r}]")

    print(f"erasure artifact: {erased_path}")
    print(f"  base_model_path   {meta.get('base_model_path')}")
    print(f"  model_key         {meta.get('model_key')}")

    got = file_sha256(base / "config.json")
    if meta.get("base_config_sha256") != got:
        die("base_config_sha256 mismatch -- this erasure was produced against a "
            f"DIFFERENT model.\n  artifact {meta.get('base_config_sha256')}\n  {base} {got}")
    print("  base_config_sha256      OK")

    index = json.loads((base / "model.safetensors.index.json").read_text())
    shard_name = index["weight_map"][TENSOR]
    head_shard = index["weight_map"].get("lm_head.weight")
    print(f"  {TENSOR} lives in {shard_name}; lm_head.weight in {head_shard}")

    shard = load_file(str(base / shard_name))
    base_embed = shard[TENSOR]
    if meta.get("base_embedding_sha256") != tensor_sha256(base_embed):
        die("base_embedding_sha256 mismatch -- the base model's embedding is not "
            "the one the erasure started from")
    print("  base_embedding_sha256   OK")

    new_embed = load_file(str(erased_path))[TENSOR]
    if meta.get("erased_embedding_sha256") != tensor_sha256(new_embed):
        die("erased_embedding_sha256 mismatch -- the artifact is corrupt")
    print("  erased_embedding_sha256 OK")
    if new_embed.shape != base_embed.shape or new_embed.dtype != base_embed.dtype:
        die(f"shape/dtype mismatch: {tuple(new_embed.shape)}/{new_embed.dtype} "
            f"vs base {tuple(base_embed.shape)}/{base_embed.dtype}")

    expected_rows = sorted(json.loads(meta["edited_token_ids"]))
    diff_before = torch.nonzero((new_embed != base_embed).any(dim=1)).flatten().tolist()
    if sorted(diff_before) != expected_rows:
        die(f"the artifact differs from the base on {len(diff_before)} rows but "
            f"edited_token_ids lists {len(expected_rows)}; they are not the same set")
    print(f"  edited rows agree with edited_token_ids   {len(expected_rows)} rows")

    out.mkdir(parents=True, exist_ok=True)
    for item in sorted(base.iterdir()):
        if item.name == shard_name or item.is_dir():
            continue
        shutil.copy2(item, out / item.name)
        print(f"  copied {item.name}")

    shard[TENSOR] = new_embed
    save_file(shard, str(out / shard_name), metadata={"format": "pt"})
    print(f"  wrote  {shard_name} ({len(shard)} tensors, {TENSOR} replaced)")

    # Re-read what was written; trusting the in-memory object would not catch a
    # bad serialisation, which is the failure this is guarding against.
    check = load_file(str(out / shard_name))
    if set(check) != set(shard):
        die("the written shard does not hold the same tensor names as the base")
    if tensor_sha256(check[TENSOR]) != meta["erased_embedding_sha256"]:
        die("the written embedding does not hash to the erased tensor")
    rows = torch.nonzero((check[TENSOR] != base_embed).any(dim=1)).flatten().tolist()
    if sorted(rows) != expected_rows:
        die(f"the written model differs from the base on the wrong rows")
    unchanged = [n for n in shard if n != TENSOR
                 and not torch.equal(check[n], shard[n])]
    if unchanged:
        die(f"tensors that must not change did: {unchanged[:5]}")
    print(f"\nverified: {out}")
    print(f"  exactly {len(rows)} embedding rows differ from {base.name}, "
          f"and they are exactly edited_token_ids")
    print(f"  all other {len(shard) - 1} tensors in {shard_name} bit-identical")
    print(f"  {head_shard} (lm_head.weight, untied) copied unchanged")


if __name__ == "__main__":
    main()
