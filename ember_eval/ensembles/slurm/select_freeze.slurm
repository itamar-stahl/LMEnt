#!/bin/bash
#SBATCH --job-name=ens-select
#SBATCH --output=/home/dcor/galbarak2/runs/ensembles/%x_%j.out
#SBATCH --error=/home/dcor/galbarak2/runs/ensembles/%x_%j.err
#SBATCH --account=gpu-research
#SBATCH --partition=killable
#SBATCH --exclude=n-102,n-202,n-203,n-204,n-205,n-602,rack-bgw-dgx1,rack-gww-dgx1,rack-omerl-g01
#SBATCH --time=30
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8000
#SBATCH --cpus-per-task=2
#SBATCH --gpus=0
#
# Phase 3: rank the ensemble candidates on the 50 SELECTION questions and freeze
# the winners. No GPU -- this only reads per-question CSVs.
#
# The twin is not opened here and neither is any _test split; select_accuracy.py
# enforces that by construction, and this job adds the one check it cannot: that
# every cell of the frozen grid actually produced a candidate. A silently short
# grid would let a winner be "best of four" where the standalone round chose best
# of eight, and that difference is invisible in the output. Missing cells fail
# the job rather than quietly shrinking the search.
set -uo pipefail
: "${ROOT:?}"; : "${ENS:?}"
PY=/home/morg/NLP_2526b/stahli/anaconda3/envs/lment/bin/python

"$PY" - "$ENS" <<'PYEOF'
import json, sys
from pathlib import Path
ENS = Path(sys.argv[1])
METHOD = {"rmu": "RMU+EMBER", "snmfv2": "SNMF+EMBER", "snmfv1": "SNMF+EMBER-V1"}
inv, expected = [], {}
for ln in open(ENS / "joblist_ensembles.tsv"):
    kind, label, topic, concept, base, *_ = ln.rstrip("\n").split("\t")
    m = METHOD[kind]
    inv.append([topic, m, label, str(ENS / "candidates" / label / "model")])
    expected[f"{topic}/{m}"] = expected.get(f"{topic}/{m}", 0) + 1
json.dump(inv, open(ENS / "inventory_ensembles.json", "w"), indent=1)
json.dump(expected, open(ENS / "expected_candidates.json", "w"), indent=1)
print(f"inventory: {len(inv)} candidates across {len(expected)} cells")
PYEOF

"$PY" "$ROOT/ember_eval/acc_selection/select_accuracy.py" \
  --scored "$ENS/scored_selection" \
  --inventory "$ENS/inventory_ensembles.json" \
  --full-label FULL_lment-1b-control-2e-b131k \
  --out "$ENS/results_ensembles" || exit 1

"$PY" - "$ENS" <<'PYEOF'
import csv, json, sys
from pathlib import Path
ENS = Path(sys.argv[1])
exp = json.load(open(ENS / "expected_candidates.json"))
rows = list(csv.DictReader(open(ENS / "results_ensembles" / "rankings.csv", encoding="utf-8")))
obs = {}
for r in rows:
    k = f"{r['topic']}/{r['method']}"
    obs[k] = obs.get(k, 0) + 1
bad = [f"  {k}: expected {v}, ranked {obs.get(k, 0)}" for k, v in sorted(exp.items())
       if obs.get(k, 0) != v]
print("candidates per cell, expected vs ranked:")
for k, v in sorted(exp.items()):
    print(f"  {k:26s} {v:2d} -> {obs.get(k, 0):2d} {'OK' if obs.get(k,0)==v else 'SHORT'}")
json.dump({"expected": exp, "observed": obs, "short_cells": bad},
          open(ENS / "results_ensembles" / "grid_completeness.json", "w"), indent=1)
if bad:
    print("\nFAIL: the grid is short -- a winner would be chosen from fewer candidates "
          "than the standalone round used:\n" + "\n".join(bad), file=sys.stderr)
    sys.exit(1)
win = json.load(open(ENS / "results_ensembles" / "selected_checkpoints.json"))
print(f"\nFROZEN {win['n_winners']} winners:")
for w in win["winners"]:
    print(f"  {w['topic']:9s} {w['method']:14s} {w['model_label']:30s} H={w['H_selection']}")
PYEOF
rc=$?
echo "select exit $rc"
exit $rc
