#!/usr/bin/env python
"""Materialise an EMBER erasure at several deltas from an EXISTING feature artifact.

The delta grid for the answer-NLL/KL selection needs one erased model per
delta. The pipeline (`ember/lment_pipeline.py::run_concept`) only ever writes
the single delta it chose, and re-running it would refit the sparse
factorisation -- which does not reproduce across GPU models at a fixed seed,
so a refit is a *different* feature set wearing the same name. This script
therefore never fits anything: it loads the artifact the finished run
recorded in its `report.json`, applies it at each requested delta exactly the
way the pipeline does (`embed_edit.apply_embedding_artifact`), writes the
edited embedding with the same provenance metadata
(`erased_embedding.save_erased_embedding`), and hands each one to
`ember_eval/materialize_erased_model.py`, which rebuilds a full HF directory
and verifies the changed rows equal `edited_token_ids` exactly.

Reproduction check, always on: the run's own `chosen_delta` is included in
the grid and the embedding this script produces for it must be byte-identical
to the run's shipped `erased_embeddings.safetensors`. If it is not, the
artifact, the base model or the edit code has drifted and NO delta from this
script is trusted.

    python ember_eval/nll_kl/ember_deltas.py \
        --report <run>/outputs/report.json \
        --deltas 1 2 5 10 20 50 100 200 500 \
        --out-root /home/morg/.../candidates/ember_rome

Writes `<out-root>/d<delta>/erased_embeddings.safetensors` and
`<out-root>/d<delta>/model/` (full HF dir), plus `<out-root>/deltas.json`.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FORK = REPO / "Ember-on-LMEnt"
sys.path.insert(0, str(FORK))
sys.path.insert(1, str(FORK / "external" / "snmf"))   # ember.utils imports factorization.seminmf

from ember.erasure import embed_edit, features, model_loader  # noqa: E402
from ember.erased_embedding import save_erased_embedding  # noqa: E402
from ember.lment_pipeline import (  # noqa: E402
    _input_embedding_name, _tensor_hash,
)

MATERIALIZE = REPO / "ember_eval" / "materialize_erased_model.py"


def slug(delta: float) -> str:
    return f"d{delta:g}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True, help="a finished run's outputs/report.json")
    ap.add_argument("--deltas", type=float, nargs="+", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--base", help="override report['model_path'] (the pristine control)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--no-materialize", action="store_true",
                    help="write the embeddings only, skip the full HF dirs")
    a = ap.parse_args()

    report = json.loads(Path(a.report).read_text())
    base = Path(a.base or report["model_path"]).resolve()
    artifact_path = Path(report["artifact"]["path"])
    feature_ids = [int(f) for f in report["feature_selection"]["selected_feature_ids"]]
    model_key = report["model_key"]
    chosen = float(report["chosen_delta"])
    shipped = Path(report["erased_embeddings_path"])
    concept = report["concept"]
    for p in (base / "config.json", artifact_path, shipped):
        if not p.exists():
            raise SystemExit(f"missing: {p}")

    deltas = sorted({float(d) for d in a.deltas} | {chosen})
    out_root = Path(a.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"concept={concept!r} base={base} features={feature_ids} chosen_delta={chosen}")
    print(f"deltas: {deltas}")

    artifact = features.load_embedding_artifact(artifact_path)
    model, tokenizer = model_loader.load_local_causal_lm(
        base, dtype=torch.float32, device=a.device)
    embedding_name = _input_embedding_name(model)
    pristine = embed_edit.snapshot(model)
    pristine_sha = _tensor_hash(model.get_input_embeddings().weight)

    from safetensors.torch import load_file
    shipped_weight = load_file(str(shipped))[embedding_name]

    manifest = {"concept": concept, "base": str(base), "artifact": str(artifact_path),
                "feature_ids": feature_ids, "chosen_delta": chosen,
                "shipped": str(shipped), "cells": {}}
    reproduced = False
    for delta in deltas:
        embed_edit.restore(model, pristine)
        info = embed_edit.apply_embedding_artifact(
            model=model, tokenizer=tokenizer, artifact=artifact,
            feature_ids=feature_ids, delta=delta, model_key=model_key)
        cell = out_root / slug(delta)
        cell.mkdir(parents=True, exist_ok=True)
        path = save_erased_embedding(
            model=model, output_dir=cell, base_model_path=base, model_key=model_key,
            tensor_name=embedding_name, base_embedding_sha256=pristine_sha,
            edited_token_ids=info["edited_token_ids"])
        sha = _tensor_hash(model.get_input_embeddings().weight)
        entry = {"delta": delta, "n_tokens_edited": info["n_tokens_edited"],
                 "erased_embeddings": str(path), "embedding_sha256": sha}
        if delta == chosen:
            same = torch.equal(model.get_input_embeddings().weight.detach().cpu(),
                               shipped_weight)
            entry["reproduces_shipped"] = same
            reproduced = same
            print(f"  delta={delta:g}: REPRODUCTION CHECK {'PASSED' if same else 'FAILED'}")
            if not same:
                raise SystemExit(
                    f"delta {delta:g} does not reproduce {shipped}; refusing to continue")
        print(f"  delta={delta:g}: {info['n_tokens_edited']} rows edited -> {path}")
        manifest["cells"][slug(delta)] = entry

    del model
    torch.cuda.empty_cache()
    assert reproduced, "chosen delta was never checked"

    if not a.no_materialize:
        for delta in deltas:
            cell = out_root / slug(delta)
            out = cell / "model"
            cmd = [sys.executable, str(MATERIALIZE), "--base", str(base),
                   "--erased", str(cell / "erased_embeddings.safetensors"),
                   "--out", str(out)]
            print("  materialize:", " ".join(cmd))
            subprocess.run(cmd, check=True)
            manifest["cells"][slug(delta)]["model"] = str(out)

    (out_root / "deltas.json").write_text(json.dumps(manifest, indent=2))
    print("wrote", out_root / "deltas.json")


if __name__ == "__main__":
    main()
