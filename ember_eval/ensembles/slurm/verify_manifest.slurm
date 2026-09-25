#!/bin/bash
#SBATCH --job-name=ens-verify
#SBATCH --output=/home/dcor/galbarak2/runs/ensembles/%x_%j.out
#SBATCH --error=/home/dcor/galbarak2/runs/ensembles/%x_%j.err
#SBATCH --account=gpu-research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --exclude=n-102,n-202,n-203,n-204,n-205,n-602,rack-bgw-dgx1,rack-gww-dgx1,rack-omerl-g01
#
# Prove the combined manifest is the merged tree, before it decides anything.
#
# The original round scored target+neighbour and SciQ separately and merged the
# output trees. This pipeline scores one manifest that already carries the SciQ
# rows under each topic -- half the checkpoint I/O. That should be identical,
# because scoring is deterministic and score_mc.py does nothing with --topic but
# filter. "Should be" is not evidence, so this re-scores a candidate the original
# round already scored and diffs every per-question row against it.
#
# Exits non-zero on any mismatch, and the whole pipeline hangs off afterok.
set -uo pipefail
: "${ROOT:?}"; : "${ENS:?}"
PY=/home/morg/NLP_2526b/stahli/anaconda3/envs/lment/bin/python
REF=/home/morg/NLP_2526b/galbarak2/runs/acc_selection/scored_merged
LABEL=ember_rome_d200
DIR=/home/morg/NLP_2526b/galbarak2/runs/nll_kl/candidates/ember_rome/d200/model
export HF_HOME=/home/dcor/galbarak2/hf_cache TOKENIZERS_PARALLELISM=false
export TMPDIR=/home/dcor/galbarak2/tmp

echo "verify on $(hostname) at $(date +%F_%H:%M:%S)"
"$PY" -u "$ROOT/ember_eval/acc_selection/score_mc.py" \
  --model "$DIR" --manifest "$ENS/manifest_selection.csv" \
  --topic rome --label "$LABEL" --out "$ENS/verify" || exit 1

"$PY" - "$ENS/verify/$LABEL/rome/per_question.csv" "$REF/$LABEL/rome/per_question.csv" <<'PYEOF'
import csv, sys
new, ref = (list(csv.DictReader(open(p, encoding="utf-8"))) for p in sys.argv[1:3])
kf = lambda r: (r["question_group"], r["question_id"])
a, b = {kf(r): r for r in new}, {kf(r): r for r in ref}
if set(a) != set(b):
    print(f"FAIL: question sets differ ({len(set(a) ^ set(b))} not shared)"); sys.exit(1)
cols = ["is_correct", "gold_margin", "n_ties_at_top", "n_nonfinite"]
bad = 0
for k in sorted(a):
    for c in cols:
        x, y = a[k].get(c), b[k].get(c)
        if c == "gold_margin":
            if abs(float(x) - float(y)) > 1e-9: bad += 1; print(f"  {k} {c}: {x} vs {y}")
        elif x != y:
            bad += 1; print(f"  {k} {c}: {x} vs {y}")
print(f"compared {len(a)} questions x {len(cols)} columns")
print("VERIFIED: the combined manifest reproduces the merged tree exactly" if not bad
      else f"FAIL: {bad} mismatching cells")
sys.exit(1 if bad else 0)
PYEOF
rc=$?
echo "verify exit $rc"
exit $rc
