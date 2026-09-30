"""Sweep SNMF's delta and score every cell on the real questions.

This is the step the pipeline is missing. Appendix C.3 tunes
delta_in, delta_out in {1,4,7,10} crossed with three layer ranges per side --
144 cells -- and selects one on downstream evaluation. `snmf.py erase` applies
ONE cell and saves it, which is how a checkpoint at delta 4 (component x3) got
written and read back as an erasure.

Run this on the cluster, against the real control twin, and pick delta from the
table it prints instead of from a rule.

    python sweep_delta.py \
        --model /home/dcor/galbarak2/hf-models/lment-1b-control-2e-b131k \
        --snmf-out /home/dcor/galbarak2/runs/mlp_erasure/snmf_rome_871547 \
        --concept "Ancient Rome" \
        --questions Ember-on-LMEnt/data/mc_questions.json \
        --out sweep_rome.csv

Scoring is ember.evals.causal_mc: options are scored as continuations of
"Question: ...\nAnswer:" by summed log-likelihood per character. EVALUATION.md
explains why letter parsing cannot be used on these base models.

The model is loaded ONCE. Each cell is applied to a snapshot of the two edited
matrices and rolled back afterwards, so cells cannot contaminate each other --
verified by re-scoring the restored model against the baseline at the end.
"""
import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Ember-on-LMEnt"))

import snmf                                              # noqa: E402
from ember.evals.causal_mc import continuation_logprobs   # noqa: E402

PUBLISHED_DELTAS = [1.0, 4.0, 7.0, 10.0]


# ------------------------------------------------------------------ #
def load_questions(path, concept, subset):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if concept not in data:
        raise SystemExit(f"{concept!r} not in {path}; have {sorted(data)[:8]}")
    items = data[concept][subset]
    for it in items:
        if it["correct_answer"] not in it["options"]:
            raise SystemExit(f"correct_answer missing from options: {it['q']!r}")
    return items


@torch.no_grad()
def score(model, tokenizer, items, device):
    """Accuracy and mean margin, per ember.evals.causal_mc's rule."""
    correct, margins = 0, []
    for it in items:
        conts = [f" {o}" for o in it["options"]]
        raw = continuation_logprobs(model, tokenizer,
                                    f"Question: {it['q']}\nAnswer:",
                                    conts, device=device)
        per_char = [s / max(len(c), 1) for (s, _), c in zip(raw, conts)]
        best = max(range(len(conts)), key=per_char.__getitem__)
        gold = it["options"].index(it["correct_answer"])
        correct += (best == gold)
        margins.append(per_char[gold] - max(
            v for i, v in enumerate(per_char) if i != gold))
    return correct / len(items), sum(margins) / len(margins)


def edited_layers(blob, selected, lo_in, hi_in, lo_out, hi_out):
    """Layers that hold selected features and fall inside at least one range."""
    out = []
    for layer in blob["layers"]:
        feats = selected.get(str(layer), [])
        if feats and (lo_in <= layer <= hi_in or lo_out <= layer <= hi_out):
            out.append(layer)
    return out


def snapshot(model, layers):
    snap = {}
    for layer in layers:
        mlp = snmf.mlp_of(snmf.get_layers(model)[layer])
        snap[layer] = (mlp.up_proj.weight.data.detach().clone(),
                       mlp.down_proj.weight.data.detach().clone())
    return snap


def restore(model, snap):
    with torch.no_grad():
        for layer, (up, down) in snap.items():
            mlp = snmf.mlp_of(snmf.get_layers(model)[layer])
            mlp.up_proj.weight.data.copy_(up)
            mlp.down_proj.weight.data.copy_(down)


# ------------------------------------------------------------------ #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--snmf-out", required=True,
                    help="directory holding features.pkl and selected.json")
    ap.add_argument("--concept", required=True)
    ap.add_argument("--questions", required=True)
    ap.add_argument("--out", default="sweep_delta.csv")
    ap.add_argument("--subset", default="QA_train",
                    help="QA_train while choosing delta; keep QA_test held out")
    ap.add_argument("--simdom-subset", default="SimdomQA_train")
    ap.add_argument("--deltas", type=float, nargs="+", default=PUBLISHED_DELTAS)
    ap.add_argument("--range-cells", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--gamma", type=float, default=0.95)
    ap.add_argument("--dtype", choices=sorted(snmf.DTYPES), default="fp32")
    ap.add_argument("--device")
    a = ap.parse_args()

    out_dir = Path(a.snmf_out)
    blob = pickle.load(open(out_dir / "features.pkl", "rb"))
    selected = json.load(open(out_dir / "selected.json"))
    n_selected = sum(len(v) for v in selected.values())
    if n_selected == 0:
        raise SystemExit("selected.json is empty; run select first")
    print(f"{n_selected} selected features across layers "
          f"{sorted(int(k) for k, v in selected.items() if v)}")

    qa = load_questions(a.questions, a.concept, a.subset)
    simdom = load_questions(a.questions, a.concept, a.simdom_subset)
    print(f"{len(qa)} concept questions, {len(simdom)} similar-domain "
          f"({a.subset} / {a.simdom_subset})")

    device = snmf.resolve_device(a.device)
    model, tok = snmf.load_model(a.model, None, snmf.DTYPES[a.dtype], device)
    n_layers = len(snmf.get_layers(model))

    base_acc, base_margin = score(model, tok, qa, device)
    base_sim, base_sim_margin = score(model, tok, simdom, device)
    print(f"\ncontrol: concept {base_acc:.1%} (margin {base_margin:+.4f}), "
          f"simdom {base_sim:.1%} (margin {base_sim_margin:+.4f})")
    print("chance is 25.0%; a successful erasure drives concept toward chance "
          "while holding simdom\n")

    rows = []
    for cell in a.range_cells:
        (lo_in, hi_in), (lo_out, hi_out) = snmf.default_layer_ranges(n_layers, cell)
        layers = edited_layers(blob, selected, lo_in, hi_in, lo_out, hi_out)
        if not layers:
            print(f"range cell {cell}: in [{lo_in},{hi_in}] out [{lo_out},{hi_out}] "
                  f"-- no selected feature falls inside; skipped")
            continue
        snap = snapshot(model, layers)

        for delta in a.deltas:
            for layer in layers:
                feats = selected.get(str(layer), [])
                d_in = delta if lo_in <= layer <= hi_in else 0.0
                d_out = delta if lo_out <= layer <= hi_out else 0.0
                if feats and (d_in or d_out):
                    snmf.ablate_layer(model, layer, blob["results"][layer]["Z"],
                                      feats, d_in, d_out, a.gamma)

            acc, margin = score(model, tok, qa, device)
            sim, sim_margin = score(model, tok, simdom, device)
            scale = snmf.erasure_scale(delta)
            row = {
                "range_cell": cell,
                "layers_in": f"[{lo_in},{hi_in}]", "layers_out": f"[{lo_out},{hi_out}]",
                "delta": delta, "erasure_scale": scale,
                "concept_acc": round(acc, 4), "concept_margin": round(margin, 4),
                "simdom_acc": round(sim, 4), "simdom_margin": round(sim_margin, 4),
                "concept_drop": round(base_acc - acc, 4),
                "simdom_drop": round(base_sim - sim, 4),
                "selectivity": round((base_acc - acc) - (base_sim - sim), 4),
                "amplifies": scale >= 1.0,
            }
            rows.append(row)
            flag = "  <-- AMPLIFIES, not an erasure" if scale >= 1.0 else ""
            print(f"cell {cell} delta {delta:>5} (x{scale:.1f})  "
                  f"concept {acc:.1%} ({base_acc - acc:+.1%})  "
                  f"simdom {sim:.1%} ({base_sim - sim:+.1%})  "
                  f"selectivity {row['selectivity']:+.1%}{flag}")

            restore(model, snap)

        # cells must not contaminate each other
        check_acc, _ = score(model, tok, qa, device)
        if abs(check_acc - base_acc) > 1e-9:
            raise SystemExit(
                f"restore failed: concept accuracy is {check_acc:.4f} after "
                f"rollback but the baseline was {base_acc:.4f}. Every number "
                f"above cell {cell} is suspect.")

    if not rows:
        raise SystemExit("no cell was evaluated")

    with open(a.out, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {len(rows)} cells -> {a.out}")

    erasing = [r for r in rows if not r["amplifies"]]
    if erasing:
        best = max(erasing, key=lambda r: r["selectivity"])
        print(f"\nbest selectivity among non-amplifying cells: "
              f"cell {best['range_cell']} delta {best['delta']} -- "
              f"concept {best['concept_drop']:+.1%}, "
              f"simdom {best['simdom_drop']:+.1%}")
    print("\nChoose the cell from this table and WRITE THE RULE DOWN before "
          "looking at QA_test. Selecting on the test split is the failure "
          "ERASURE_RESULTS.md retracted two acc_raw claims for.")


if __name__ == "__main__":
    main()
