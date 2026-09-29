#!/usr/bin/env python3
"""Score grid cells by how DISTINCTIVE their features' tokens are, from the data.

A second, independent screen over the same 27 cells as
`screen_feature_grid.py`, built because that one cannot be trusted to rank.

WHY A SECOND SCREEN
-------------------
`screen_feature_grid.py` matches a hand-written list of Roman words
(`ROME_MARKERS`, borrowed from `archive/ember_eval/run_rome_heldout.slurm`). That list
was written to audit paragraphs of prose, and at the level of subword tokens it
demonstrably undercounts. Its own top-scoring feature in the grid (rank 300,
g_sparsity 0.005, seed 42, feature 254) reads:

    Roman, Roman, Romans, Rome, gens, if, uet, Marcus, Western, theon, erva,
    Luc, Gaul, Wes, Wel, legion, Julius, aka, Kerala, pra, Loss, -shaped, IF,
    -wide, ios, yo

It counted 5 of 26. It missed `Marcus`, `Julius`, `Gaul`, `gens`, and plausibly
`theon` (Pantheon) and `erva` (Minerva/Nerva) -- the list has no Roman personal
names. So its ordering of the 27 cells cannot be relied on, and using it to
pick which cells to pay the judge for would be circular.

WHAT THIS USES INSTEAD
----------------------
Nothing hand-written. For every token in V', compare how often it occurs in the
concept sentences against the neutral sentences, and score a feature by the mean
log rate ratio of its tokens:

    score(feature) = mean over its tokens of  log( (p_concept + a) / (p_neutral + a) )

`Marcus`, `Julius` and `Gaul` are caught automatically because they are in fact
far commoner in Roman text; `not` and `food` are not, because they are not.
There is no threshold to tune -- the score is continuous, and the only constant
is additive smoothing.

THIS IS NOT `ratio_abs`, AND NOT `num_concept_related`
------------------------------------------------------
Three different quantities, easy to confuse:

  num_concept_related  did this token OCCUR in a concept sentence? Useless:
                       Roman articles are ordinary English, so ~88% of any
                       feature's tokens qualify.
  ratio_abs            how much more does the FACTORIZATION activate on
                       concept tokens than neutral ones? Correlates 0.84 with
                       the above, so it carries the same "appears in Roman
                       prose" signal -- and it can blow up: one grid cell
                       reports ratio_abs 248,106 because `mean_abs_neutral`
                       is exactly 0 for a near-dead component.
  THIS                 is this token's VOCABULARY distinctive of Roman text?

The first two ask about occurrence and activation. This asks about the words.

VALIDATION, NOT ASSERTION
-------------------------
Run with no arguments it prints the most and least distinctive tokens, so the
statistic can be inspected before it is trusted -- if `Caesar` does not outrank
`the`, it is wrong and should be discarded rather than tuned.

It is still a proxy for the judge, and the plan is to test it: judge cells
spanning its range and see whether the judge agrees with its ordering. It reads
vocabulary, not meaning.

Like every other stage-1/2 statistic here it never touches the ablated twin, an
accuracy, an erasure or a delta.

    python grid/token_distinctiveness.py --grid-root /home/dcor/galbarak2/lment-ember-grid/features
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import re
from collections import Counter
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

MODEL = "/home/dcor/galbarak2/hf-models/lment-1b-control-2e-b131k"
DATA = Path(__file__).resolve().parent.parent / "data"
LEADING = ("Ġ", "▁")


def clean(tok: str) -> str:
    t = str(tok)
    for mark in LEADING:
        t = t.replace(mark, " ")
    return t.strip()


def token_rates(concept: str, smoothing: float):
    """{token_string: log rate ratio}, from the same sentences the run used."""
    tok = AutoTokenizer.from_pretrained(MODEL)
    con = json.loads((DATA / "concept_sentences.json").read_text(encoding="utf-8"))
    neu = json.loads((DATA / "neutral_sentences.json").read_text(encoding="utf-8"))
    sents = next(r["sentences"] for r in con if r["concept"] == concept)
    neutrals = [r["sentence"] for r in neu if r.get("sentence")]

    def count(texts):
        """Counts keyed on the CLEANED token string, summed across ids.

        Several ids share one cleaned form -- most importantly the
        leading-space variant and the bare one ('Ġthe' and 'the'). Keying the
        output dict on the cleaned string while counting per id silently let
        the last id processed overwrite the others, which put 'the' at +3.607,
        exactly level with 'Caesar', because the rare sentence-initial variant
        happens to be concept-only. Summing here is also the right level of
        aggregation: the feature token lists this scores are strings, not ids.
        """
        c = Counter()
        for s in texts:
            # No special tokens: they are excluded from V' by
            # build_token_label_codes, so counting them would skew the rates.
            ids = tok(s, add_special_tokens=False)["input_ids"]
            c.update(clean(t) for t in tok.convert_ids_to_tokens(ids))
        return c

    cc, nc = count(sents), count(neutrals)
    n_con, n_neu = max(sum(cc.values()), 1), max(sum(nc.values()), 1)
    out = {}
    for form in set(cc) | set(nc):
        p_con = cc.get(form, 0) / n_con
        p_neu = nc.get(form, 0) / n_neu
        out[form] = math.log((p_con + smoothing) / (p_neu + smoothing))
    return out, len(sents), len(neutrals)


def feature_score(cell: str, rates: dict, default: float) -> float:
    try:
        toks = ast.literal_eval(cell) if isinstance(cell, str) else list(cell)
    except (ValueError, SyntaxError):
        return float("nan")
    vals = [rates.get(clean(t), default) for t in toks if clean(t)]
    return sum(vals) / len(vals) if vals else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid-root", required=True)
    ap.add_argument("--concept", default="Ancient Rome")
    ap.add_argument("--concept-dir", default="Ancient_Rome")
    ap.add_argument("--smoothing", type=float, default=1e-5)
    ap.add_argument("--show-tokens", type=int, default=15)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    rates, n_con, n_neu = token_rates(args.concept, args.smoothing)
    # A token absent from both sets gets the neutral score, 0.0, rather than
    # being dropped: a feature padded with unseen tokens is diluted, and that
    # is a real property of the feature, not missing data.
    default = 0.0

    print(f"token distinctiveness for {args.concept!r}: "
          f"{n_con} concept sentences vs {n_neu} neutral, "
          f"{len(rates)} distinct tokens, smoothing {args.smoothing:g}")
    ranked = sorted(rates.items(), key=lambda kv: kv[1], reverse=True)
    print(f"\nMOST distinctive (sanity check -- Roman vocabulary should be here):")
    print("  " + ", ".join(f"{t!r}" for t, _ in ranked[:args.show_tokens]))
    print(f"LEAST distinctive (ordinary or neutral-domain words):")
    print("  " + ", ".join(f"{t!r}" for t, _ in ranked[-args.show_tokens:]))
    for probe in ("Roman", "Caesar", "Marcus", "Julius", "Gaul", "legion",
                  "the", "not", "food", "organizations"):
        v = rates.get(probe)
        print(f"    {probe:<14} {v:+.3f}" if v is not None
              else f"    {probe:<14} (not in V')")

    root = Path(args.grid_root)
    rows = []
    for tokens_csv in sorted(root.glob(
            f"sp*_seed*/**/csvs/rank*/seed*/{args.concept_dir}/embedding/token_features.csv")):
        emb = tokens_csv.parent
        sp_seed = next((p.name for p in tokens_csv.parents
                        if re.fullmatch(r"sp[0-9.]+_seed\d+", p.name)), None)
        if sp_seed is None:
            continue
        df = pd.read_csv(tokens_csv)
        if "activating_tokens" not in df.columns:
            continue
        df["distinct"] = df["activating_tokens"].map(
            lambda c: feature_score(c, rates, default))
        stats = emb / "stats_embed.csv"
        if stats.is_file():
            st = pd.read_csv(stats)
            if {"feature", "ratio_abs"} <= set(st.columns):
                df = df.merge(st[["feature", "ratio_abs"]], on="feature", how="left")
        if "ratio_abs" not in df.columns:
            df["ratio_abs"] = float("nan")
        # Only features the judge would actually be shown: select_with_judge
        # applies the ratio prefilter first.
        elig = df[df["ratio_abs"] >= 2.0]
        pool = elig if not elig.empty else df
        best = pool.loc[pool["distinct"].idxmax()]
        rows.append({
            "rank": int(emb.parents[2].name.replace("rank", "")),
            "g_sparsity": float(sp_seed.split("_")[0].replace("sp", "")),
            "seed": int(sp_seed.split("seed")[1]),
            "n_eligible": int(len(elig)),
            "best_distinct": float(best["distinct"]),
            "best_feature": int(best["feature"]),
            "mean_distinct": float(pool["distinct"].mean()),
        })

    grid = pd.DataFrame(rows).sort_values("best_distinct", ascending=False)
    print(f"\n{'=' * 88}\ncells ranked by their best eligible feature's mean token "
          f"distinctiveness\n{'=' * 88}")
    print(grid.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print("=" * 88)

    if args.json_out:
        Path(args.json_out).write_text(json.dumps({
            "concept": args.concept, "smoothing": args.smoothing,
            "most_distinctive": [t for t, _ in ranked[:50]],
            "cells": grid.to_dict(orient="records")}, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
