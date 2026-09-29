#!/usr/bin/env python
"""Did the input embeddings learn anything, or only shrink?

The 1B twins trained before commit 885d87c drew `embeddings.weight` from N(0,1)
instead of trunc_normal(0, 0.02) -- 50x too large -- and the consequence was not
a crash but a silence: the model trains, the loss falls, and every
embedding-space method finds nothing, because the learned signal never grows
large enough to dominate the initial noise.

`check_embedding_init` catches that at step 0. This script asks the same
question at the *other* end of the run, which the guard cannot see: given that
the draw was correct, did the embeddings move for any reason other than weight
decay pulling them toward zero?

THE TEST. AdamW's decoupled weight decay multiplies every parameter by
(1 - lr_t * wd) each step, independent of the gradient. Applying that product
over the whole LR schedule predicts the row-norm a matrix would reach if it
learned *nothing*:

    predicted = initial_row_norm * prod_t (1 - lr_t * wd)

A ratio of observed/predicted near 1.00 is the failure the docstring of
`check_embedding_init` records for the pre-fix twins: 0.5492 observed against
0.5471 predicted, i.e. nothing learned survived. A ratio meaningfully above 1
means gradient updates outran the decay and there is real structure to read.

Reads only `model.embeddings.weight`, never the optimizer state, so it touches
~800 MB of a 15 GB checkpoint.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import torch


def load_embeddings(step_dir: str) -> torch.Tensor:
    """The input embedding matrix from one OLMo-core .distcp checkpoint."""
    from torch.distributed.checkpoint import FileSystemReader

    mao = os.path.join(step_dir, "model_and_optim")
    if not os.path.isdir(mao):
        raise SystemExit(f"no model_and_optim under {step_dir}")

    shards = [f for f in os.listdir(mao) if f.startswith("__")]
    if len(shards) != 16:
        raise SystemExit(
            f"refusing: {step_dir} has {len(shards)} proper shards, expected 16 "
            "(OPERATIONS.md: latest_checkpoint_dir selects incomplete checkpoints)"
        )

    reader = FileSystemReader(mao)
    key = "model.embeddings.weight"

    # Shape comes from the checkpoint's own metadata rather than the config, so
    # a vocab or d_model change cannot silently produce a wrong-shaped read.
    md = reader.read_metadata()
    if key not in md.state_dict_metadata:
        raise SystemExit(
            f"{key} not in {step_dir}; keys look like: "
            + ", ".join(list(md.state_dict_metadata)[:5])
        )
    shape = tuple(md.state_dict_metadata[key].size)
    state = {key: torch.empty(shape, dtype=torch.float32)}

    # Single-process read of a 16-shard checkpoint. torch 2.6 removed `no_dist`
    # from the public `load()` and it survives only on the deprecated
    # `load_state_dict()`; without it, DCP tries to reach a process group that
    # was never initialised. `_load_state_dict_from_keys` does not exist in this
    # build at all, so this is the one path that works here.
    import torch.distributed.checkpoint as dcp

    dcp.load_state_dict(
        state_dict=state,
        storage_reader=reader,
        no_dist=True,
    )
    w = state[key].detach().float()
    if not torch.isfinite(w).all():
        raise SystemExit(f"non-finite values in {key} from {step_dir}")
    return w


def lr_at(step: int, peak: float, warmup: int, t_max: int, alpha_f: float) -> float:
    """olmo_core CosWithWarmup: linear warmup, then cosine to alpha_f * peak."""
    if step < warmup:
        return peak * step / warmup
    frac = (step - warmup) / max(t_max - warmup, 1)
    frac = min(frac, 1.0)
    eta_min = peak * alpha_f
    return eta_min + (peak - eta_min) * (1 + math.cos(math.pi * frac)) / 2


def decay_factor(steps: int, peak: float, wd: float, warmup: int, t_max: int, alpha_f: float) -> float:
    """prod_t (1 - lr_t * wd) over the schedule actually run."""
    logf = 0.0
    for t in range(steps):
        logf += math.log1p(-lr_at(t, peak, warmup, t_max, alpha_f) * wd)
    return math.exp(logf)


def describe(name: str, w: torch.Tensor) -> dict:
    norms = w.norm(dim=1)
    d = {
        "shape": tuple(w.shape),
        "std": float(w.std()),
        "mean_row_norm": float(norms.mean()),
        "median_row_norm": float(norms.median()),
        "frac_exact_zero_rows": float((norms == 0).float().mean()),
    }
    print(f"[{name}] shape={d['shape']} std={d['std']:.6f} "
          f"row_norm mean={d['mean_row_norm']:.4f} median={d['median_row_norm']:.4f} "
          f"zero_rows={d['frac_exact_zero_rows']:.4%}", flush=True)
    return d


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--init", required=True, help="step0 checkpoint dir")
    p.add_argument("--final", required=True, help="final checkpoint dir")
    p.add_argument("--steps", type=int, required=True, help="steps actually trained")
    p.add_argument("--peak-lr", type=float, default=0.0004)
    p.add_argument("--wd", type=float, default=0.05, help="embeddings' weight decay (group override)")
    p.add_argument("--warmup", type=int, default=2000)
    p.add_argument("--t-max", type=int, default=54832)
    p.add_argument("--alpha-f", type=float, default=0.1)
    p.add_argument("--json-out", default=None)
    a = p.parse_args()

    print(f"[emb] init  = {a.init}", flush=True)
    print(f"[emb] final = {a.final}", flush=True)

    w0 = load_embeddings(a.init)
    d0 = describe("init ", w0)
    w1 = load_embeddings(a.final)
    d1 = describe("final", w1)

    f = decay_factor(a.steps, a.peak_lr, a.wd, a.warmup, a.t_max, a.alpha_f)
    predicted = d0["mean_row_norm"] * f
    ratio = d1["mean_row_norm"] / predicted if predicted else float("nan")

    # How far the matrix moved beyond pure shrinkage: compare against the
    # decayed initial draw itself, not against the raw draw.
    resid = (w1 - w0 * f)
    rel_move = float(resid.norm() / (w0 * f).norm())
    cos = torch.nn.functional.cosine_similarity(w1, w0, dim=1)

    print("", flush=True)
    print(f"[emb] decay factor over {a.steps} steps (wd={a.wd}) = {f:.6f}", flush=True)
    print(f"[emb] predicted final row-norm from decay alone     = {predicted:.4f}", flush=True)
    print(f"[emb] observed  final row-norm                      = {d1['mean_row_norm']:.4f}", flush=True)
    print(f"[emb] RATIO observed/predicted                      = {ratio:.4f}", flush=True)
    print(f"[emb] relative movement beyond decay                = {rel_move:.4f}", flush=True)
    print(f"[emb] row cosine(init, final): mean={float(cos.mean()):.4f} "
          f"median={float(cos.median()):.4f} min={float(cos.min()):.4f}", flush=True)
    print("", flush=True)
    if ratio < 1.02:
        print("[emb] VERDICT: ratio ~1.00 -- embeddings only shrank. This is the "
              "pre-fix failure mode; embedding-space methods will find nothing.", flush=True)
    else:
        print(f"[emb] VERDICT: ratio {ratio:.2f} -- gradient updates outran the decay, "
              "so there is learned structure in embedding space.", flush=True)

    if a.json_out:
        import json
        json.dump({"init": d0, "final": d1, "steps": a.steps, "decay_factor": f,
                   "predicted_row_norm": predicted, "ratio": ratio,
                   "rel_move_beyond_decay": rel_move,
                   "row_cos_mean": float(cos.mean()), "row_cos_median": float(cos.median())},
                  open(a.json_out, "w"), indent=2)
        print(f"[emb] wrote {a.json_out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
