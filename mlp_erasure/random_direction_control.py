"""Are the SNMF features special, or is any perturbation of that size as damaging?

Job 893308 found that large sign-preserving amplification (delta -19/-99, i.e.
x20/x100) drops concept accuracy 10-20 points -- but only for the 44- and
58-feature sets. The 6-feature set does nothing even at x100. That pattern
(damage tracks FEATURE COUNT, not concept) suggests generic capacity damage
rather than concept-specific erasure.

This tests it directly. For each condition the SAME ablate_layer machinery is
applied at the SAME delta to the SAME number of directions in the SAME layers;
only the identity of the neurons changes:

  real          the actual SNMF-selected features
  perm-seed-N   each selected feature's entries randomly PERMUTED across
                neurons -- identical values, identical support size, identical
                magnitude distribution, different neurons

A permutation control is the right null here rather than Gaussian noise: it
holds sparsity and scale exactly fixed, so the only thing that varies is
whether these particular neurons carry the concept.

Read: if the permuted conditions drop concept accuracy as much as `real`, the
SNMF features are not special and the large-delta degradation is capacity
damage. If `real` drops substantially more, the features do carry something.
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Ember-on-LMEnt"))

import snmf                      # noqa: E402
import sweep_delta as sd         # noqa: E402


def permute_columns(Z, feature_ids, seed):
    """Shuffle each selected feature's entries across neurons, values intact."""
    g = torch.Generator().manual_seed(seed)
    if isinstance(Z, torch.Tensor):
        Zp = Z.clone()
        n = Z.shape[0]
        for f in feature_ids:
            Zp[:, f] = Z[:, f][torch.randperm(n, generator=g)]
        return Zp
    # numpy fallback
    import numpy as np
    rng = np.random.default_rng(seed)
    Zp = Z.copy()
    n = Z.shape[0]
    for f in feature_ids:
        Zp[:, f] = Z[:, f][rng.permutation(n)]
    return Zp


def apply_all_layers(model, blob, selected, delta, gamma, Z_for_layer):
    """Ablate every layer holding a selected feature, on BOTH sides."""
    total = 0
    for layer in blob["layers"]:
        feats = selected.get(str(layer), [])
        if not feats:
            continue
        n, _ = snmf.ablate_layer(model, layer, Z_for_layer(layer), feats,
                                 delta, delta, gamma)
        total += n
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--snmf-out", required=True)
    ap.add_argument("--concept", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--deltas", type=float, nargs="+", default=[-19.0, -99.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--subset", default="QA_train")
    ap.add_argument("--simdom-subset", default="SimdomQA_train")
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--dtype", default="fp32")
    ap.add_argument("--device")
    ap.add_argument("--out")
    a = ap.parse_args()

    out_dir = Path(a.snmf_out)
    blob = pickle.load(open(out_dir / "features.pkl", "rb"))
    selected = json.load(open(out_dir / "selected.json"))
    layers_used = sorted(int(k) for k, v in selected.items() if v)
    n_feats = sum(len(v) for v in selected.values())
    print(f"{n_feats} selected features across layers {layers_used}")

    qa = sd.load_questions(a.questions, a.concept, a.subset)
    simdom = sd.load_questions(a.questions, a.concept, a.simdom_subset)

    device = snmf.resolve_device(a.device)
    model, tok = snmf.load_model(a.model, None, snmf.DTYPES[a.dtype], device)

    base_acc, _ = sd.score(model, tok, qa, device)
    base_sim, _ = sd.score(model, tok, simdom, device)
    print(f"\ncontrol: concept {base_acc:.1%}  simdom {base_sim:.1%}  "
          f"(chance 25.0%)\n")

    snap = sd.snapshot(model, layers_used)
    rows = []

    for delta in a.deltas:
        scale = snmf.erasure_scale(delta)
        conditions = [("real", None)] + [(f"perm-seed{s}", s) for s in a.seeds]
        for name, seed in conditions:
            if seed is None:
                zf = lambda layer: blob["results"][layer]["Z"]
            else:
                cache = {
                    layer: permute_columns(blob["results"][layer]["Z"],
                                           selected.get(str(layer), []), seed)
                    for layer in layers_used
                }
                zf = lambda layer: cache[layer]

            edited = apply_all_layers(model, blob, selected, delta, a.gamma, zf)
            acc, _ = sd.score(model, tok, qa, device)
            sim, _ = sd.score(model, tok, simdom, device)
            sd.restore(model, snap)

            rows.append({
                "delta": delta, "scale": scale, "condition": name,
                "features_edited": edited,
                "concept_acc": round(acc, 4), "simdom_acc": round(sim, 4),
                "concept_drop": round(base_acc - acc, 4),
                "simdom_drop": round(base_sim - sim, 4),
            })
            print(f"delta {delta:>6} (x{scale:.0f})  {name:<12} "
                  f"edited={edited:<3} concept {acc:.1%} ({base_acc-acc:+.1%})  "
                  f"simdom {sim:.1%} ({base_sim-sim:+.1%})")

        # rollback integrity: the baseline must come back exactly
        chk, _ = sd.score(model, tok, qa, device)
        if abs(chk - base_acc) > 1e-9:
            raise SystemExit(
                f"restore failed after delta {delta}: {chk:.4f} vs baseline "
                f"{base_acc:.4f}; every row above is suspect")
        print()

    if a.out:
        import csv
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {len(rows)} rows -> {a.out}")

    print("\nREAD: if perm-seed* drop concept as much as `real`, these features "
          "are not special and the large-delta effect is capacity damage.")


if __name__ == "__main__":
    main()
