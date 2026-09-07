#!/usr/bin/env python3
"""Judge named grid cells and STOP. No erasure, no delta, no evaluation.

Stage 3. Two screens rank the 27 cells and they disagree, so the judge is the
arbiter -- but it must not be allowed to run on to an erasure.

WHY JUDGE-ONLY MATTERS
----------------------
`run_lment_ember` goes features -> selection -> erasure -> evaluation in one
flow. Pointing it at several cells would produce several erasure results, and
choosing among those would be choosing a hyperparameter by the size of the
effect it produces. That is the failure that retracted two `acc_raw` claims
(`ember_eval/EVALUATION.md`) and that `completion_eval/metric_bakeoff.py` was
written to avoid.

So this calls `select_with_judge` and returns. Nothing here reads the ablated
twin, an accuracy, an erasure or a delta. Once a cell is chosen on what the
judge says about the CONTROL model's features, the erasure runs once, and that
result is then a real test.

WHICH CELLS, AND WHY NOT "THE TOP THREE"
----------------------------------------
Picking the top three of one screen would mean trusting a ranking neither screen
has earned. Both are proxies that read tokens, not meaning, and each has a
demonstrated blind spot:

  screen_feature_grid.py   matches a hand-written Roman word list borrowed from
                           run_rome_heldout.slurm. Written for prose, it has no
                           Roman personal names, so it scored its own champion
                           5/26 while missing Marcus, Julius, Gaul and gens.

  token_distinctiveness.py derives distinctiveness from concept-vs-neutral token
                           rates, catching those automatically. But it averages
                           over a feature's tokens, so a feature that is
                           intensely Roman in six tokens and junk in twenty
                           scores mediocre -- it penalises exactly the
                           concentrated case we are looking for.

They nominate different cells, and the two candidates fail in opposite
directions:

  rank 100 / 0.005 / 42, feature 36   king, kings, monarchy, emperor, Empire,
                                      dynasty, ruler, temple, reign, dictator,
                                      tyranny... clean and coherent, but the
                                      concept is monarchy, not Rome.
  rank 300 / 0.005 / 42, feature 254  Roman, Romans, Rome, Marcus, Gaul,
                                      legion, Julius... the right concept,
                                      diluted with Kerala, yo, -shaped, IF.

Judging both plus a cell both screens rate poorly tests the screens as well as
the features. If the judge rejects the best candidate under two independent
definitions of "best", the negative result is robust rather than one setting's
bad luck. If it accepts one, we have an erasure target and we also learn which
screen to trust.

    python grid/judge_cells.py --config configs/ember_lment_rome_slurm.yaml \\
        --grid-root /home/dcor/galbarak2/lment-ember-grid/features \\
        --cell 100:0.005:42 --cell 300:0.005:42 --cell 500:0.02:44
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ember.lment_feature_selection import select_with_judge  # noqa: E402
from ember.lment_pipeline import load_lment_config  # noqa: E402
from ember.lment_worker import build_judge  # noqa: E402


def parse_cell(text: str) -> tuple[int, str, int]:
    """"rank:g_sparsity:seed", e.g. 300:0.005:42."""
    try:
        rank, sparsity, seed = text.split(":")
        return int(rank), sparsity, int(seed)
    except ValueError as exc:
        raise SystemExit(f"--cell must be rank:g_sparsity:seed, got {text!r}") from exc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    help="the run config, read for judge settings only")
    ap.add_argument("--grid-root", required=True)
    ap.add_argument("--cell", action="append", required=True,
                    help="rank:g_sparsity:seed; repeatable")
    ap.add_argument("--concept", default="Ancient Rome")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    config = load_lment_config(Path(args.config))
    cells = [parse_cell(c) for c in args.cell]

    # One judge for every cell. The 23 GB load dominates a cell's cost, so
    # loading it per cell would multiply the only expensive part of the run.
    print(f"loading the judge ({config.judge_model}, executor "
          f"{config.judge_executor}) once for {len(cells)} cell(s)", flush=True)
    judge = build_judge(config)
    if judge is None:
        raise SystemExit("the config declares no judge model")

    results = []

    def flush() -> None:
        """Write after every cell, not at the end.

        A cell costs minutes and the 23 GB judge load costs ~20, so a job that
        hits its wall clock must not throw away the cells it already finished.
        Judging is deterministic given a cell's features, so a resubmission can
        skip whatever is already recorded here.
        """
        if args.out:
            Path(args.out).write_text(json.dumps(
                {"concept": args.concept, "cells": results}, indent=2),
                encoding="utf-8")

    try:
        for rank, sparsity, seed in cells:
            features_root = Path(args.grid_root) / f"sp{sparsity}_seed{seed}"
            if not features_root.is_dir():
                print(f"\n!! no such cell: {features_root}", flush=True)
                results.append({"rank": rank, "g_sparsity": sparsity,
                                "seed": seed, "error": "cell directory missing"})
                flush()
                continue
            print(f"\n{'=' * 78}\nrank {rank}  g_sparsity {sparsity}  seed {seed}"
                  f"\n{'=' * 78}", flush=True)
            try:
                sel = select_with_judge(
                    features_root=features_root,
                    model_key=config.model_key,
                    concept=args.concept,
                    rank=rank,
                    seed=seed,
                    prefilter_threshold=config.ratio_thresh,
                    confidence_threshold=config.judge_confidence_threshold,
                    top_k=config.judge_top_k,
                    describe_callback=judge.describe_feature,
                    classify_callback=judge.classify_feature,
                )
            except ValueError as exc:
                # "selected no features above the confidence threshold" is the
                # expected negative result, not a crash -- record and continue.
                print(f"  REJECTED: {exc}", flush=True)
                results.append({"rank": rank, "g_sparsity": sparsity, "seed": seed,
                                "accepted": 0, "reason": str(exc)})
                flush()
                continue
            ids = list(sel.selected_feature_ids)
            print(f"  ACCEPTED {len(ids)} feature(s): {ids}", flush=True)
            print(f"  csv: {sel.potential_features_path}", flush=True)
            results.append({"rank": rank, "g_sparsity": sparsity, "seed": seed,
                            "accepted": len(ids), "feature_ids": ids,
                            "potential_features_path": sel.potential_features_path})
            flush()
    finally:
        close = getattr(judge, "close", None)
        if callable(close):
            close()

    print(f"\n{'=' * 78}")
    any_accepted = any(r.get("accepted") for r in results)
    for r in results:
        state = (f"{r['accepted']} accepted" if r.get("accepted")
                 else "none accepted" if "accepted" in r else r.get("error", "?"))
        print(f"  rank {r['rank']:>4} sp {r['g_sparsity']:<6} seed {r['seed']}  ->  {state}")
    if any_accepted:
        print("\nA cell has an erasure target. Choose ONE, set rank/g_sparsity/seed")
        print("and features.reuse in the run config, and run run_lment_ember once.")
        print("Do not judge more cells afterwards to find a better erasure result.")
    else:
        print("\nNo cell accepted, under two independent screens' best candidates.")
        print("That is a result about this model's embedding geometry, not a")
        print("configuration problem: EMBER assumes a concept has a findable")
        print("embedding subspace, and at 1B that assumption may not hold. The")
        print("untested axis is the concept sentences -- EMBER's stock Rome")
        print("Wikipedia text, not the 56 QIDs / 65,844 chunks actually ablated.")
    print("=" * 78)

    flush()
    if args.out:
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
