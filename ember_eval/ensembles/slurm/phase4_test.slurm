#!/bin/bash
#SBATCH --job-name=ens-test
#SBATCH --output=/home/dcor/galbarak2/runs/ensembles/%x_%j.out
#SBATCH --error=/home/dcor/galbarak2/runs/ensembles/%x_%j.err
#SBATCH --account=gpu-research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=48000
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --exclude=n-102,n-202,n-203,n-204,n-205,n-602,rack-bgw-dgx1,rack-gww-dgx1,rack-omerl-g01
#
# Phase 4, one topic: the frozen winners through the held-out evaluation, by the
# identical code path the existing nine cells travelled.
#
#   score_mc.py    on the TEST manifest  -> accuracy and H_test   (Panel A)
#   score_model.py on sets/<topic>.json  -> answer NLL and the full-vocabulary
#                                           next-token distributions for KL
#                                           (Panel B); test-phase items only
#
# Reads the winners at runtime rather than taking them as arguments: this job is
# submitted before they exist, and hard-coding them would be a second place for
# the frozen list to drift from results_ensembles/selected_checkpoints.json.
#
# Sequential on purpose. Each winner is a 5 GB load and the scoring itself is
# seconds; parallel readers on one node each get 1/N of its link. Restartable --
# anything already written is skipped.
set -uo pipefail
: "${ROOT:?}"; : "${ENS:?}"; : "${TOPIC:?}"
PY=/home/morg/NLP_2526b/stahli/anaconda3/envs/lment/bin/python
SETS="$ROOT/ember_eval/nll_kl/sets"
CHECK="$ROOT/ember_eval/nll_kl/check_alloc.py"
export HF_HOME=/home/dcor/galbarak2/hf_cache TOKENIZERS_PARALLELISM=false
export TMPDIR=/home/dcor/galbarak2/tmp

echo "phase4 $TOPIC on $(hostname) at $(date +%F_%H:%M:%S)"
nvidia-smi --query-gpu=name --format=csv,noheader | head -1 || true

mapfile -t WINNERS < <("$PY" -c "
import json,sys
w=json.load(open('$ENS/results_ensembles/selected_checkpoints.json'))['winners']
for x in w:
    if x['topic']=='$TOPIC': print(x['model_label'])
")
echo "winners for $TOPIC: ${WINNERS[*]:-none}"
[ "${#WINNERS[@]}" -gt 0 ] || { echo "no winners for $TOPIC" >&2; exit 1; }

failed=0
for label in "${WINNERS[@]}"; do
  dir="$ENS/candidates/$label/model"
  if ! "$PY" "$CHECK" "$dir"; then
    echo "SKIP $label: partially-written checkpoint" >&2; failed=$((failed+1)); continue
  fi
  if [ -f "$ENS/scored_test/$label/$TOPIC/per_question.csv" ]; then
    echo "=== $label accuracy: already scored"
  else
    echo "=== $label accuracy  $(date +%H:%M:%S)"
    "$PY" -u "$ROOT/ember_eval/acc_selection/score_mc.py" \
      --model "$dir" --manifest "$ENS/manifest_test.csv" \
      --topic "$TOPIC" --label "$label" --out "$ENS/scored_test" \
      || { echo "FAILED accuracy $label" >&2; failed=$((failed+1)); }
  fi
  if [ -f "$ENS/scored_nllkl/$label/$TOPIC/records.json" ]; then
    echo "=== $label NLL/KL: already scored"
  else
    echo "=== $label NLL/KL  $(date +%H:%M:%S)"
    "$PY" -u "$ROOT/ember_eval/nll_kl/score_model.py" \
      --model "$dir" --sets "$SETS/$TOPIC.json" --out "$ENS/scored_nllkl/$label" \
      || { echo "FAILED nllkl $label" >&2; failed=$((failed+1)); }
  fi
done

echo "phase4 $TOPIC done at $(date +%F_%H:%M:%S), failed=$failed"
exit $((failed > 0 ? 1 : 0))
