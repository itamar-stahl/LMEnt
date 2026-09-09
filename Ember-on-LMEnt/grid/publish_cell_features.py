#!/usr/bin/env python3
"""Stamp one grid cell's factorization as a reusable feature cache.

WHY THIS IS NEEDED
------------------
`run_lment_ember` refits the factorization as part of its own flow, and the fit
does NOT reproduce across GPU models at a fixed seed (see JUDGE_RESULTS.md: the
same cell gave different features on an a6000 and a 3090, and the judge accepted
in one and rejected in the other). So the erasure must be handed the exact
factorization the judge accepted, via `lment.features.reuse: true` and a
`cache_root` pointing at the chosen cell -- otherwise it could erase a feature
nobody judged.

But `validate_feature_bundle` refuses a cache that lacks its provenance files,
and correctly so: `get_pipeline_path` keys its directories on rank and seed but
NOT on g_sparsity, so two cells that differ only in sparsity land on identical
paths. `feature_manifest.json` is the only thing that records g_sparsity, which
makes it the guard that stops the wrong cell being reused. The grid ran
`train_mf_features` directly rather than through `prepare_lment_run`, so its
cells have the three artifact branches but none of that provenance.

This writes it -- by calling the repo's own `add_feature_provenance`, the same
function `publish_run_features` uses, so what is written cannot drift from what
`validate_feature_bundle` checks. It then validates, so a mistake is caught here
rather than inside the erasure job.

Nothing here reads the ablated twin, an accuracy, an erasure or a delta.

    python grid/publish_cell_features.py \\
        --config configs/ember_lment_rome_slurm.yaml \\
        --cell-root /home/dcor/galbarak2/lment-ember-grid/features/sp0.02_seed44 \\
        --rank 100 --seed 44 --g-sparsity 0.02
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# external/snmf too: ember.utils imports `factorization.seminmf` from it, and
# ensure_factor_artifact adds exactly this pair to PYTHONPATH for its subprocess.
for _p in (ROOT, ROOT / "external" / "snmf"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from ember.lment_pipeline import load_lment_config  # noqa: E402
from ember.lment_runs import (  # noqa: E402
    add_feature_provenance, feature_concept_dirs, validate_feature_bundle)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cell-root", required=True,
                    help="the grid cell, e.g. .../features/sp0.02_seed44")
    ap.add_argument("--rank", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--g-sparsity", type=float, required=True,
                    help="must match the cell directory; recorded in the manifest")
    ap.add_argument("--concept", default="Ancient Rome")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    root = Path(args.cell_root)
    # The sparsity is not in the artifact paths, so a mismatch between the flag
    # and the directory name would be recorded silently in the manifest and
    # would then VALIDATE against the wrong cell. Check it against the name.
    expected = f"sp{args.g_sparsity:g}_seed{args.seed}"
    if root.name != expected:
        raise SystemExit(
            f"--g-sparsity/--seed say {expected!r} but the cell directory is "
            f"{root.name!r}. Refusing: the manifest would claim the wrong cell.")

    config = load_lment_config(Path(args.config))
    config = replace(config, rank=args.rank, seed=args.seed,
                     feature_g_sparsity=args.g_sparsity)

    dirs = feature_concept_dirs(root, config, args.concept)
    for branch, directory in dirs.items():
        print(f"  {branch:<16} {directory}"
              f"{'' if directory.is_dir() else '   <-- MISSING'}")

    if args.dry_run:
        print("\n--dry-run: nothing written")
        return

    add_feature_provenance(root, config, args.concept)
    validate_feature_bundle(root, config, args.concept)

    manifest = json.loads(
        (dirs["csvs"] / "feature_manifest.json").read_text(encoding="utf-8"))
    print("\nvalidated. feature_manifest.json:")
    for key, value in manifest.items():
        print(f"  {key:<22} {value}")
    print(f"\nSet in the run config:\n"
          f"  rank: {args.rank}\n  seed: {args.seed}\n"
          f"  lment.features.g_sparsity: {args.g_sparsity}\n"
          f"  lment.features.reuse: true\n"
          f"  lment.features.cache_root: {root}")


if __name__ == "__main__":
    main()
