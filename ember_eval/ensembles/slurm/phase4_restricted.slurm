#!/bin/bash
#SBATCH --job-name=ens-restr
#SBATCH --output=/home/dcor/galbarak2/runs/ensembles/%x_%j.out
#SBATCH --error=/home/dcor/galbarak2/runs/ensembles/%x_%j.err
#SBATCH --account=gpu-research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=48000
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#
# Held-out evaluation for ONE gate-restricted RMU+EMBER cell, by the identical
# code path every other cell used. The checkpoint is named explicitly rather
# than read from selected_checkpoints.json, because this condition is frozen in
# RESTRICTED_CONDITION.md instead -- committed before this job could run.
#
# One cell per job, deliberately: a 5 GB read from /home/morg collapses with two
# concurrent readers on a node (measured: 638 kB/s with two, minutes with one),
# so these are submitted with excludes that keep them apart.
set -uo pipefail
: "${ROOT:?}"; : "${ENS:?}"; : "${TOPIC:?}"; : "${LABEL:?}"
PY=/home/morg/NLP_2526b/stahli/anaconda3/envs/lment/bin/python
CHECK="$ROOT/ember_eval/nll_kl/check_alloc.py"
export HF_HOME=/home/dcor/galbarak2/hf_cache TOKENIZERS_PARALLELISM=false
export TMPDIR=/home/dcor/galbarak2/tmp

dir="$ENS/candidates/$LABEL/model"
echo "restricted $TOPIC / $LABEL on $(hostname) at $(date +%F_%H:%M:%S)"
"$PY" "$CHECK" "$dir" || { echo "partially-written checkpoint" >&2; exit 1; }
rc=0

if [ -f "$ENS/scored_test/$LABEL/$TOPIC/per_question.csv" ]; then
  echo "accuracy: already scored"
else
  "$PY" -u "$ROOT/ember_eval/acc_selection/score_mc.py" \
    --model "$dir" --manifest "$ENS/manifest_test.csv" \
    --topic "$TOPIC" --label "$LABEL" --out "$ENS/scored_test" || rc=1
fi
if [ -f "$ENS/scored_nllkl/$LABEL/$TOPIC/records.json" ]; then
  echo "NLL/KL: already scored"
else
  "$PY" -u "$ROOT/ember_eval/nll_kl/score_model.py" \
    --model "$dir" --sets "$ROOT/ember_eval/nll_kl/sets/$TOPIC.json" \
    --out "$ENS/scored_nllkl/$LABEL" || rc=1
fi
echo "restricted $TOPIC done at $(date +%F_%H:%M:%S), rc=$rc"
exit $rc
