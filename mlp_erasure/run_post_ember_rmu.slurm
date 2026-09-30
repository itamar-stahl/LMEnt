#! /bin/sh
#SBATCH --job-name=postember-rmu
#SBATCH --output=/home/dcor/galbarak2/runs/mlp_erasure/%x_%j.out
#SBATCH --error=/home/dcor/galbarak2/runs/mlp_erasure/%x_%j.err
#SBATCH --account=gpu-research
#SBATCH --partition=killable
#SBATCH --exclude=rack-bgw-dgx1,rack-gww-dgx1,rack-omerl-g01,n-202,n-203,n-204,n-205,n-602
#SBATCH --time=120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32000
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#
# VARIANT 3: RMU on top of an EMBER-erased model.
#
# RMU is the other MLP method and works completely differently from SNMF --
# it trains down_proj to point forget-set activations at a random control
# vector, rather than projecting a direction out of the weights. Including it
# is what makes this a picture of "MLP methods after EMBER" rather than "SNMF
# after EMBER".
#
# Probes the erased model's OWN residual norms first: EMBER changed the
# embeddings, so activation scale may differ from the control's
# 72.7/83.2/93.8 even though the MLP weights are identical. Then runs the two
# cells that were the only ones ever to clear rotation_is_substantial on the
# control (L5-hi, L6-hi), at this model's own probed steering_high.
#
# Reading it: the sanity gates decide whether the run counts at all. A flat
# QA number on a cell that failed rotation_is_substantial says nothing --
# that is the trap Daniela's model fell into (job 892663/893171).

# ROOT: the checkout whose code this job runs. Walk up from the directory you
# submitted from, so a second checkout runs its own code rather than whichever
# tree was hard-coded here; export ROOT=<checkout> to pin one deliberately.
if [ -z "${ROOT:-}" ]; then
  ROOT=${SLURM_SUBMIT_DIR:-$PWD}
  while [ "$ROOT" != "/" ] && [ ! -f "$ROOT/mlp_erasure/env.sh" ]; do
    ROOT=$(dirname "$ROOT")
  done
fi
[ -f "$ROOT/mlp_erasure/env.sh" ] || { echo "not an LMEnt checkout: ${SLURM_SUBMIT_DIR:-$PWD}; export ROOT=<checkout>" >&2; exit 1; }
. "$ROOT/mlp_erasure/env.sh"
HF=/home/dcor/galbarak2/hf-models

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 || echo "no nvidia-smi"

# tag | model | concept
printf '%s\n' \
  "romeerased|$HF/lment-1b-rome-erased-b131k|Ancient Rome" \
  "bberased|$HF/lment-1b-baseball-erased-b131k|Baseball" \
| while IFS='|' read -r tag model concept; do
  PROBE="$RUNS/rmu_postember_${tag}_probe_${SLURM_JOB_ID}"
  echo
  echo "================ $tag ($concept): probe ================"
  "$PY" "$ROOT/mlp_erasure/rmu.py" \
    --model "$model" --concept "$concept" \
    --concept-sentences "$DATA/concept_sentences.json" \
    --neutral-sentences "$DATA/neutral_sentences.json" \
    --dtype fp32 --out "$PROBE" --probe
  echo "probe exit $?"
  cat "$PROBE/probe.json" 2>/dev/null

  for LID in 5 6; do
    case "$LID" in
      5) LIDS="3 4 5" ;;
      6) LIDS="4 5 6" ;;
    esac
    STEER=$("$PY" -c "
import json
d = json.load(open('$PROBE/probe.json'))
print(d['layers']['$LID']['steering_high'])
")
    if [ -z "$STEER" ]; then
      echo "FATAL: no steering for $tag L$LID (probe failed?)" >&2
      continue
    fi
    OUT="$RUNS/rmu_postember_${tag}_L${LID}hi_${SLURM_JOB_ID}"
    echo
    echo "--- $tag L${LID}-hi steering=$STEER ---"
    "$PY" "$ROOT/mlp_erasure/rmu.py" \
      --model "$model" --concept "$concept" \
      --concept-sentences "$DATA/concept_sentences.json" \
      --neutral-sentences "$DATA/neutral_sentences.json" \
      --dtype fp32 --lr 1e-4 --alpha 100 --steering "$STEER" \
      --layer-id "$LID" --layer-ids $LIDS \
      --batch-size 4 --max-num-batches 150 \
      --out "$OUT" --sanity
    echo "rmu exit $? (1 just means a sanity gate failed; eval still runs)"

    if [ -d "$OUT/model" ]; then
      "$PY" "$ROOT/ember_eval/score_ember_mc.py" \
        --model "$OUT/model" --subfolder "" \
        --ember-data "$DATA" \
        --concepts "$concept" --sets qa_test simdom_test \
        --out "$OUT/eval.json"
      echo "score exit $?"
      rm -rf "$OUT/model"
      echo "checkpoint deleted"
    else
      echo "NO MODEL SAVED for $tag L$LID"
    fi
  done
done

echo DONE
