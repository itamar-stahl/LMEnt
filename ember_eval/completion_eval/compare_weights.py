"""Compare an erased model with the base and never-learned models.

The directional measurements assume that the base and never-learned models
used the same initialization and training setup.

    python compare_weights.py \
        --base <base> --never <never> --erased <erased> --out weights.json

Let D_erase = W_erased - W_base and D_target = W_never - W_base. The second is
where a model that never learned the concept actually sits, so it is the
displacement erasure is trying to reproduce. The script reports:

    rel_edit_size          |D_erase| / |W_base|      how large the edit is
    rel_target_size        |D_target| / |W_base|     how large the gap is
    cosine                 whether the edit points the right way
    progress_along_target  how far along that direction it got, 1.0 = all of it
    residual               |D_erase - D_target| / |D_target|, how much is left

Read cosine and progress together. A tiny edit in exactly the right direction
has cosine near 1 and progress near 0; a large edit in an unrelated direction
has progress near 0 and cosine near 0. Only the pair distinguishes them.

Three changes from the version in nlp_evaluation/:

  1. Models load on CPU. The original passed device_map="auto" to three 1B
     models at once; when they do not all fit, parameters land on different
     devices and `erased - base` raises a device-mismatch error. Nothing here
     needs a GPU, and running on CPU also keeps it off the training queue.
  2. Tensors present in one model but not another, or with mismatched shapes,
     are counted and reported rather than silently skipped.
  3. Per-tensor directional stats are marked undefined when the never-learned
     displacement for that tensor is numerically zero, instead of dividing by
     the 1e-12 clamp and printing noise.
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

TINY = 1e-12


def load(name, cache_dir=None, dtype=torch.float32, device="cpu"):
    m = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=dtype, cache_dir=cache_dir)
    m.config.use_cache = False
    m.eval()
    return m.to(device)


def named_targets(model, device="cpu"):
    """All floating-point parameters, keyed by name (buffers excluded).

    Everything is pulled onto one device so that differences between two
    models can be taken regardless of how either was loaded.
    """
    return {name: value.detach().float().to(device)
            for name, value in model.named_parameters() if value.is_floating_point()}


def directional_stats(W_base, W_erased, W_never):
    """How far, and in what direction, erasure moved relative to never-learned."""
    d_erase = (W_erased - W_base).flatten()
    d_target = (W_never - W_base).flatten()
    nt = d_target.norm()
    ne = d_erase.norm()
    nb = W_base.norm().clamp(min=TINY)
    out = {"rel_edit_size": (ne / nb).item(), "rel_target_size": (nt / nb).item()}
    if nt.item() < TINY or ne.item() < TINY:
        # this tensor did not move in one of the two comparisons, so the
        # direction it moved in is not defined
        out.update({"cosine": None, "progress_along_target": None, "residual": None})
        return out
    dot = torch.dot(d_erase, d_target)
    out.update({
        "cosine": (dot / (ne * nt)).item(),
        "progress_along_target": (dot / (nt * nt)).item(),
        "residual": ((d_erase - d_target).norm() / nt).item(),
    })
    return out


def global_stats(base, erased, never):
    """The same quantities accumulated over the complete parameter vector."""
    base_sq = erase_sq = target_sq = residual_sq = dot = 0.0
    n_parameters = 0
    n_tensors = 0
    n_skipped = 0
    for name, b in base.items():
        if name not in erased or name not in never:
            n_skipped += 1
            continue
        e, n = erased[name], never[name]
        if b.shape != e.shape or b.shape != n.shape:
            n_skipped += 1
            continue
        de, dt = e - b, n - b
        base_sq += b.pow(2).sum().item()
        erase_sq += de.pow(2).sum().item()
        target_sq += dt.pow(2).sum().item()
        residual_sq += (de - dt).pow(2).sum().item()
        dot += (de * dt).sum().item()
        n_parameters += b.numel()
        n_tensors += 1
    eps = 1e-24
    base_norm = max(base_sq, eps) ** 0.5
    erase_norm = max(erase_sq, eps) ** 0.5
    target_norm = max(target_sq, eps) ** 0.5
    return {
        "n_tensors": n_tensors,
        "n_skipped": n_skipped,
        "n_parameters": n_parameters,
        "rel_edit_size": erase_norm / base_norm,
        "rel_target_size": target_norm / base_norm,
        "cosine": dot / (erase_norm * target_norm),
        "progress_along_target": dot / max(target_sq, eps),
        "residual": residual_sq ** 0.5 / target_norm,
    }


def cmd_weights(a):
    base = load(a.base, a.cache_dir, device=a.device)
    erased = load(a.erased, a.cache_dir, device=a.device)
    never = load(a.never, a.cache_dir, device=a.device)
    B = named_targets(base, a.device)
    E = named_targets(erased, a.device)
    N = named_targets(never, a.device)

    global_row = global_stats(B, E, N)
    print("global:", json.dumps(global_row, indent=2))
    if global_row["n_skipped"]:
        print(f"warning: {global_row['n_skipped']} tensors skipped (missing from one "
              "model, or shape mismatch)")

    rows = {}
    for k in B:
        if k not in E or k not in N or B[k].shape != N[k].shape or B[k].shape != E[k].shape:
            continue
        rows[k] = directional_stats(B[k], E[k], N[k])

    moved = {k: v for k, v in rows.items() if v["rel_edit_size"] > 1e-8}
    print(f"{len(moved)}/{len(rows)} matrices changed by erasure")
    for k, v in sorted(moved.items(), key=lambda x: -x[1]["rel_edit_size"])[:20]:
        cos = "  n/a " if v["cosine"] is None else f"{v['cosine']:+.3f}"
        prog = "  n/a " if v["progress_along_target"] is None \
            else f"{v['progress_along_target']:+.3f}"
        print(f"{k:55s} size={v['rel_edit_size']:.2e} cos={cos} progress={prog}")

    emb_key = next((k for k in B if k.endswith("embed_tokens.weight")), None)
    if emb_key and (E[emb_key] - B[emb_key]).abs().sum() > 0:
        delta = (E[emb_key] - B[emb_key]).norm(dim=1)
        nz = (delta > 0).sum().item()
        top = delta.topk(min(30, delta.numel()))
        tok = AutoTokenizer.from_pretrained(a.base, cache_dir=a.cache_dir)
        print(f"\nembedding rows changed: {nz} ({100*nz/delta.numel():.3f}% of vocab)")
        print("largest:", [tok.decode([i]) for i in top.indices.tolist()])
    elif emb_key:
        print("\nembeddings unchanged (expected for RMU and SNMF)")

    if a.out:
        report = {"global": global_row, "parameters": rows}
        Path(a.out).write_text(json.dumps(report, indent=2))
        print("wrote", a.out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--never", required=True)
    ap.add_argument("--erased", required=True)
    ap.add_argument("--cache-dir")
    ap.add_argument("--device", default="cpu",
                    help="three 1B models in fp32 is roughly 13 GB; CPU is the "
                         "default so this never competes for a GPU")
    ap.add_argument("--out")
    a = ap.parse_args()
    cmd_weights(a)


if __name__ == "__main__":
    main()
