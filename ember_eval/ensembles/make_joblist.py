#!/usr/bin/env python
"""Emit the frozen ensemble grid as a work list, one cell per line.

The grid and the tie-break order are fixed in ENSEMBLE_GRID.md, committed before
any candidate existed. This script only renders that file's contents as TSV; it
decides nothing. A file rather than an argument so 43 entries cannot be
truncated, re-split or mangled by quoting, and so the exact work list is
recoverable afterwards -- the same convention as the selection joblists.

Columns:  kind  label  topic  concept  base_model  p1  p2  p3

    rmu     ...  layer  band      alpha
    snmfv1  ...  fitdir delta_in  delta_out      (frozen control-derived fit)
    snmfv2  ...  REFIT  delta_in  delta_out      (fit re-derived in prep)

    python ember_eval/ensembles/make_joblist.py --out <run>/joblist_ensembles.tsv
"""
from __future__ import annotations

import argparse
from pathlib import Path

CANDIDATES = Path("/home/morg/NLP_2526b/galbarak2/runs/nll_kl/candidates")
MLP_RUNS = Path("/home/dcor/galbarak2/runs/mlp_erasure")

CONCEPTS = [
    # slug, concept string, EMBER base (the SELECTED winner, not the released model)
    ("rome", "Ancient Rome", CANDIDATES / "ember_rome/d200/model"),
    ("baseball", "Baseball", CANDIDATES / "ember_baseball/d10/model"),
    ("ai", "Artificial intelligence", CANDIDATES / "ember_ai/d500/model"),
]

# RMU: the standalone grid exactly -- 2 layers x 2 bands x 2 alphas.
# `band` picks the steering scale out of the per-concept probe run on the
# EMBER-erased model: mid = mean_residual_norm, hi = steering_high (10x that).
# It is never inherited from the control.
RMU_LAYERS = (5, 6)
RMU_BANDS = ("mid", "hi")
RMU_ALPHAS = (10, 100)

# SNMF side -> (delta_in, delta_out); 1.0 removes the component exactly.
SIDES = {"in": (1.0, 0.0), "out": (0.0, 1.0), "both": (1.0, 1.0)}

# Variant 1 reuses the factorizations already derived on the control.
V1_FITS = {
    ("rome", "ratio"): MLP_RUNS / "snmf_rome_883188",
    ("rome", "judge"): MLP_RUNS / "snmf_romeband_871493",
    ("baseball", "ratio"): MLP_RUNS / "snmf_baseball_887805",
    ("ai", "ratio"): MLP_RUNS / "snmf_ai_887806",
}
# Per ENSEMBLE_GRID.md: ratio x {in,out,both} everywhere, plus Rome's judge_both
# for Variant 1 only (Variant 2 is ratio-only, --skip-llm).
V1_CELLS = [("ratio", s) for s in SIDES] + [("judge", "both")]
V2_CELLS = [("ratio", s) for s in SIDES]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    lines = []
    for slug, concept, base in CONCEPTS:
        for layer in RMU_LAYERS:
            for band in RMU_BANDS:
                for alpha in RMU_ALPHAS:
                    label = f"rmuember_{slug}_L{layer}{band}_a{alpha}"
                    lines.append(("rmu", label, slug, concept, str(base),
                                  str(layer), band, str(alpha)))
        for select, side in V2_CELLS:
            din, dout = SIDES[side]
            label = f"snmfv2_{slug}_{select}_{side}"
            lines.append(("snmfv2", label, slug, concept, str(base),
                          "REFIT", str(din), str(dout)))
        for select, side in V1_CELLS:
            fit = V1_FITS.get((slug, select))
            if fit is None:
                continue          # only Rome has a judged control fit
            din, dout = SIDES[side]
            label = f"snmfv1_{slug}_{select}_{side}"
            lines.append(("snmfv1", label, slug, concept, str(base),
                          str(fit), str(din), str(dout)))

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join("\t".join(r) + "\n" for r in lines))
    by_kind: dict[str, int] = {}
    for r in lines:
        by_kind[r[0]] = by_kind.get(r[0], 0) + 1
    print(f"wrote {len(lines)} cells to {out}")
    for k, n in sorted(by_kind.items()):
        print(f"  {k:8s} {n}")


if __name__ == "__main__":
    main()
