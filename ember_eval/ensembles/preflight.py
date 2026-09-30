#!/usr/bin/env python
"""Phase 1: assert the ensemble composition is clean before building anything.

EMBER runs with `save.full_model: false` and edits only the input embedding
matrix, so an EMBER-erased checkpoint must have BIT-IDENTICAL MLP weights to the
control. Everything downstream depends on that: it is what makes "apply an MLP
method on top of EMBER" a composition rather than two interacting edits. The
post-EMBER scripts assert it rather than assume it, and so does this.

Checked per concept, against the control:
    MLP tensors          bit-identical   (the weights the MLP methods will edit)
    attention + norms    bit-identical   (nothing else moved)
    lm_head              bit-identical   (EMBER edits the input embedding only;
                                          OLMo-2 has tie_word_embeddings: false,
                                          so the edit must not reach the head)
    embed_tokens         DIFFERENT       (EMBER did something)

Reads the safetensors header and stream-hashes each tensor's raw byte range in
fixed-size chunks: exact rather than a tolerance, no torch, and bounded memory
regardless of tensor size. An earlier version materialised tensors through
safetensors and was reaped by the harness for RAM -- hence the streaming.

    sbatch ember_eval/ensembles/slurm/preflight.slurm     (~20 GB of NFS reads)
"""
from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path
from typing import Dict, Iterator, Tuple

CONTROL = Path("/home/dcor/galbarak2/hf-models/lment-1b-control-2e-b131k")
CANDIDATES = Path("/home/morg/NLP_2526b/galbarak2/runs/nll_kl/candidates")
CHUNK = 8 << 20

# The SELECTED EMBER winners (Appendix A, Table 7) -- NOT the released
# `lment-1b-*-erased-b131k` models the post_ember scripts point at. For AI those
# are different checkpoints entirely; see ENSEMBLE_GRID.md.
EMBER = {
    "Ancient Rome": ("ember_rome_d200", CANDIDATES / "ember_rome/d200/model"),
    "Baseball": ("ember_baseball_d10", CANDIDATES / "ember_baseball/d10/model"),
    "Artificial intelligence": ("ember_ai_d500", CANDIDATES / "ember_ai/d500/model"),
}


def shard_tensors(path: Path) -> Iterator[Tuple[str, int, int]]:
    """(name, absolute start, absolute end) for every tensor in one shard."""
    with open(path, "rb") as fh:
        (header_len,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(header_len))
    base = 8 + header_len
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        begin, end = meta["data_offsets"]
        yield name, base + begin, base + end


def tensor_hashes(model_dir: Path) -> Dict[str, str]:
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    out: Dict[str, str] = {}
    for shard in sorted(set(index.values())):
        path = model_dir / shard
        with open(path, "rb") as fh:
            for name, start, end in sorted(shard_tensors(path), key=lambda t: t[1]):
                fh.seek(start)
                h = hashlib.sha256()
                remaining = end - start
                while remaining:
                    block = fh.read(min(CHUNK, remaining))
                    if not block:
                        raise SystemExit(f"{path}: truncated at {name}")
                    h.update(block)
                    remaining -= len(block)
                out[name] = h.hexdigest()
    return out


def family(name: str) -> str:
    if name == "model.embed_tokens.weight":
        return "embed_tokens"
    if name == "lm_head.weight":
        return "lm_head"
    if ".mlp." in name:
        return "mlp"
    if ".self_attn." in name:
        return "attention"
    return "norms"


EXPECT_IDENTICAL = ("mlp", "attention", "norms", "lm_head")


def main() -> None:
    print(f"control: {CONTROL}", flush=True)
    ctrl = tensor_hashes(CONTROL)
    print(f"  {len(ctrl)} tensors hashed\n", flush=True)

    groups: Dict[str, list] = {}
    for name in ctrl:
        groups.setdefault(family(name), []).append(name)

    failures = 0
    for concept, (label, path) in EMBER.items():
        print(f"{concept}  ({label})\n  {path}", flush=True)
        if not path.exists():
            print("  [FAIL] checkpoint missing\n")
            failures += 1
            continue
        other = tensor_hashes(path)
        if set(other) != set(ctrl):
            print(f"  [FAIL] tensor set differs from the control "
                  f"({len(set(other) ^ set(ctrl))} not shared)\n")
            failures += 1
            continue
        for fam in EXPECT_IDENTICAL:
            names = groups[fam]
            diff = [n for n in names if ctrl[n] != other[n]]
            failures += bool(diff)
            print(f"  [{'FAIL' if diff else 'PASS'}] {fam:11s} bit-identical to control "
                  f"({len(names)} tensors"
                  f"{f'; {len(diff)} DIFFER: {diff[:3]}' if diff else ''})", flush=True)
        emb = groups["embed_tokens"][0]
        changed = ctrl[emb] != other[emb]
        failures += not changed
        print(f"  [{'PASS' if changed else 'FAIL'}] embed_tokens differs from control "
              f"({'EMBER edited it' if changed else 'UNCHANGED -- not an erased model'})\n",
              flush=True)

    if failures:
        print(f"{failures} check(s) FAILED -- do not build ensembles on these checkpoints.",
              file=sys.stderr)
        raise SystemExit(1)
    print("All preflight checks passed. The composition is clean: the MLP methods "
          "will edit exactly the weights they edited on the control.")


if __name__ == "__main__":
    main()
