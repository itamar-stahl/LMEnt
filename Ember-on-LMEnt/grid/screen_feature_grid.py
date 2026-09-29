#!/usr/bin/env python3
"""Rank the feature grid by how DISTINCTIVELY Roman each cell's features are.

Stage 2 of the grid. Stage 1 (`run_feature_grid.slurm`) builds features for
every (rank, g_sparsity, seed) cell in about 6.5s each. The Gemma judge costs
~45 minutes per cell, so it cannot be run on all of them; this picks the few
worth paying for.

WHY NOT SCREEN ON THE NUMBERS ALREADY IN THE OUTPUT
---------------------------------------------------
`stats_embed.csv` has `ratio_abs` (mean |G| on concept tokens over neutral
tokens) and `token_features.csv` has `num_concept_related`. Both look like
ready-made specificity scores. Neither is, and job 858233 proves it:

    feature 22: ratio_abs 7.97, 45 of 51 tokens "concept related" (88%)
                Gemma read the tokens: "words associated with negation,
                limitation, scarcity, and restriction"          -> rejected
    feature 45: ratio_abs 6.07, 41 of 51 (80%)
                "food, culinary items, and dining"              -> rejected

All 24 features clearing `ratio_abs >= 2` were rejected, at confidence
0.95-1.0. Measured over all 100 features, corr(ratio_abs, concept fraction) =
0.84 -- they are one signal, and that signal is **"appears in Roman prose"**.

A token counts as "concept related" if it occurred in any concept sentence, and
Roman Wikipedia articles are written in ordinary English. So `not`, `without`,
`food` and `organizations` are all "Rome related", and ~88% of any feature's
tokens qualify. The statistic cannot distinguish a feature about Rome from a
feature about negation that happens to appear in Rome articles.

WHAT THIS USES INSTEAD
----------------------
Rome-*distinctive* vocabulary: the `ROME_MARKERS` alternation already used by
`archive/ember_eval/run_rome_heldout.slurm` to prove chunk_id -> Roman text before that
measurement was trusted. `caesar`, `legion*`, `aqueduct*`, `denarius`,
`carthag*` and so on -- words that are about Rome rather than merely nearby.

Per cell it reports the best feature's marker share and how many features clear
a marker threshold. A cell where some feature is half Roman vocabulary is worth
the judge; a cell whose best feature is 4% Roman vocabulary is not, and the
858233 result says the judge will reject it.

The screen is a PROXY for the judge, not a replacement. It reads word lists, not
meanings; a high score means "worth 45 minutes", never "this is a Rome feature".
Only the judge decides that.

WHAT THIS DELIBERATELY CANNOT SEE
---------------------------------
The ablated twin, any accuracy, any erasure, any delta. Every number here comes
from the control model's own embedding matrix. So choosing a cell on this screen
cannot be influenced by the size of the effect the erasure is meant to show --
which is the selection-on-the-outcome failure that retracted two `acc_raw`
claims (`ember_eval/EVALUATION.md`) and that
`completion_eval/metric_bakeoff.py` exists to avoid. Fix the cell here, then
run the erasure once; that result is then a real test.

    python grid/screen_feature_grid.py --grid-root /home/dcor/galbarak2/lment-ember-grid/features
"""
from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

import pandas as pd

# Copied verbatim from archive/ember_eval/run_rome_heldout.slurm, where it gated the
# held-out perplexity measurement. Kept identical on purpose: if the two ever
# disagree about what counts as Roman vocabulary, the audit that validated the
# held-out numbers no longer describes this screen.
ROME_MARKERS = re.compile(
    r"\b(ancient rome|roman\w*|rome|caesar|augustus|cicero|pompey|nero|caligula|"
    r"hadrian|trajan|constantine|legion\w*|centurion\w*|consul\w*|praetor\w*|"
    r"plebeian\w*|patrician\w*|gladiator\w*|colosseum|aqueduct\w*|punic|carthag\w*|"
    r"latium|etruscan\w*|senatus|imperator|denarius|denarii)\b",
    re.IGNORECASE,
)

# The dolma2 tokenizer writes a leading space as 'G with a dot above' and some
# exports use the sentencepiece underscore. Neither is part of the word.
LEADING = ("Ġ", "▁")


def clean(tok: str) -> str:
    t = str(tok)
    for mark in LEADING:
        t = t.replace(mark, " ")
    return t.strip()


def marker_share(cell: str) -> float:
    """Fraction of a feature's token list that is distinctively Roman."""
    try:
        toks = ast.literal_eval(cell) if isinstance(cell, str) else list(cell)
    except (ValueError, SyntaxError):
        return 0.0
    toks = [clean(t) for t in toks]
    toks = [t for t in toks if t]
    if not toks:
        return 0.0
    return sum(1 for t in toks if ROME_MARKERS.search(t)) / len(toks)


def read_cell(tokens_csv: Path, stats_csv: Path) -> pd.DataFrame | None:
    """One row per feature, with its marker share and its ratio_abs."""
    if not tokens_csv.is_file():
        return None
    tok = pd.read_csv(tokens_csv)
    if "activating_tokens" not in tok.columns:
        return None
    tok["marker_share"] = tok["activating_tokens"].map(marker_share)
    # `projection_top_tokens` is what the feature direction WRITES to the
    # vocabulary, as opposed to what activates it. A genuine concept direction
    # should look Roman on both sides, so both are reported.
    if "projection_top_tokens" in tok.columns:
        tok["proj_marker_share"] = tok["projection_top_tokens"].map(marker_share)
    else:
        tok["proj_marker_share"] = float("nan")
    if stats_csv.is_file():
        st = pd.read_csv(stats_csv)
        if {"feature", "ratio_abs"} <= set(st.columns):
            tok = tok.merge(st[["feature", "ratio_abs"]], on="feature", how="left")
    if "ratio_abs" not in tok.columns:
        tok["ratio_abs"] = float("nan")
    return tok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-root", required=True,
                    help="the features/ directory run_feature_grid.slurm wrote")
    ap.add_argument("--concept-dir", default="Ancient_Rome")
    ap.add_argument("--marker-thresh", type=float, default=0.25,
                    help="a feature counts as Roman-looking above this share")
    ap.add_argument("--ratio-thresh", type=float, default=2.0,
                    help="the prefilter the judge would apply, for context")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    root = Path(args.grid_root)
    tokens = sorted(root.glob(
        f"sp*_seed*/**/csvs/rank*/seed*/{args.concept_dir}/embedding/token_features.csv"))
    if not tokens:
        raise SystemExit(f"no token_features.csv under {root}")

    rows = []
    for tokens_csv in tokens:
        emb_dir = tokens_csv.parent
        stats_csv = emb_dir / "stats_embed.csv"
        # .../sp0.01_seed42/<model-key>/csvs/rank100/seed42/<concept>/embedding
        # Found by name rather than by a fixed parents[] index: the depth
        # depends on the model key, and an off-by-one here would silently
        # mislabel every cell's g_sparsity.
        sp_seed = next((p.name for p in tokens_csv.parents
                        if re.fullmatch(r"sp[0-9.]+_seed\d+", p.name)), None)
        if sp_seed is None:
            print(f"  (cell dir not named sp<x>_seed<n>, skipping) {tokens_csv}")
            continue
        rank = int(emb_dir.parents[2].name.replace("rank", ""))
        sp = float(sp_seed.split("_")[0].replace("sp", ""))
        seed = int(sp_seed.split("seed")[1])

        df = read_cell(tokens_csv, stats_csv)
        if df is None or df.empty:
            print(f"  (unreadable, skipping) {tokens_csv}")
            continue

        eligible = df[df["ratio_abs"] >= args.ratio_thresh]
        roman = df[df["marker_share"] >= args.marker_thresh]
        # The cell only helps if a Roman-looking feature ALSO clears the ratio
        # prefilter -- `select_with_judge` applies that before the judge, so a
        # feature below it is never shown to the judge at all.
        both = df[(df["marker_share"] >= args.marker_thresh)
                  & (df["ratio_abs"] >= args.ratio_thresh)]
        best = df.loc[df["marker_share"].idxmax()]
        rows.append({
            "rank": rank, "g_sparsity": sp, "seed": seed,
            "n_features": len(df),
            "tokens_per_feature": int(df["num_activating_tokens_all"].median())
            if "num_activating_tokens_all" in df.columns else -1,
            "n_eligible": len(eligible),
            "n_roman": len(roman),
            "n_roman_and_eligible": len(both),
            "best_marker_share": float(best["marker_share"]),
            "best_feature": int(best["feature"]),
            "best_feature_ratio": float(best["ratio_abs"]),
            "best_proj_marker_share": float(best["proj_marker_share"]),
            "max_ratio": float(df["ratio_abs"].max()),
        })

    grid = pd.DataFrame(rows).sort_values(
        ["n_roman_and_eligible", "best_marker_share"], ascending=False)

    print(f"\nscreened {len(grid)} cells from {root}")
    print(f"a feature is 'Roman-looking' at marker_share >= {args.marker_thresh:.2f}; "
          f"the judge's prefilter is ratio_abs >= {args.ratio_thresh:g}\n")
    print("=" * 100)
    cols = ["rank", "g_sparsity", "seed", "tokens_per_feature", "n_eligible",
            "n_roman", "n_roman_and_eligible", "best_marker_share",
            "best_feature", "best_feature_ratio", "max_ratio"]
    print(grid[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("=" * 100)

    # --- seed variance, the reason seed is an axis ------------------------- #
    print("\nseed spread within each (rank, g_sparsity) -- 858233 plateaued at a")
    print("fixed loss from iter ~150, so a bad cell and a bad init look alike:")
    for (rank, sp), g in grid.groupby(["rank", "g_sparsity"]):
        if len(g) < 2:
            continue
        b = g["best_marker_share"]
        print(f"  rank {rank:>4} sp {sp:<6} best_marker_share "
              f"{b.min():.3f} .. {b.max():.3f}  (spread {b.max() - b.min():.3f})")

    # --- the verdict ------------------------------------------------------- #
    top = grid.iloc[0]
    print("\n" + "=" * 100)
    if top["n_roman_and_eligible"] > 0:
        print(f"JUDGE THESE: rank {int(top['rank'])}, g_sparsity {top['g_sparsity']}, "
              f"seed {int(top['seed'])} has {int(top['n_roman_and_eligible'])} "
              f"feature(s) both Roman-looking and past the prefilter.")
        print("Run stage 2 on the top cells only. The screen reads word lists, not")
        print("meanings -- it says 'worth 45 minutes', never 'this is a Rome feature'.")
    else:
        print("NO CELL has a feature that is both Roman-looking and past the")
        print("prefilter. If that holds across the grid, the finding is that a sparse")
        print("factorization of THIS model's embedding matrix does not isolate an")
        print("Ancient Rome direction at any of these settings -- which is a real")
        print("result about 1B embedding geometry, and much stronger than the single")
        print("failed setting job 858233 gave. Before concluding it, note the")
        print("concept sentences are EMBER's stock Rome Wikipedia text, NOT the 56")
        print("QIDs / 65,844 chunks actually held out; the sampling seam is the")
        print("untested axis (Ember-on-LMEnt/README.md).")
    print("=" * 100)

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps({"marker_thresh": args.marker_thresh,
                        "ratio_thresh": args.ratio_thresh,
                        "markers": ROME_MARKERS.pattern,
                        "cells": grid.to_dict(orient="records")}, indent=2),
            encoding="utf-8")
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
