#! /bin/sh
#SBATCH --job-name=rmu-dani-steer
#SBATCH --output=/home/dcor/galbarak2/runs/mlp_erasure/%x_%j.out
#SBATCH --error=/home/dcor/galbarak2/runs/mlp_erasure/%x_%j.err
#SBATCH --account=gpu-research
#SBATCH --partition=killable
#SBATCH --exclude=rack-bgw-dgx1,rack-gww-dgx1,rack-omerl-g01,n-202,n-203,n-204,n-205,n-602
#SBATCH --time=60
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32000
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#
# RMU steering sweep on Daniela's LMEnt-1B-2E, Ancient Rome.
#
# WHY: job 892663 ran L5-hi/L6-hi at her own probed steering_high (norm x10)
# and both failed even the loose forget_rotated gate -- cos_forget_end 0.034
# / 0.038, against 0.314 / 0.319 for the same cells on Gal's control. So RMU
# was never actually exercised on her model; the flat QA there proves
# nothing. Her residual norms are ~2.4x smaller (27.4/33.8/41.8 vs
# 72.7/83.2/93.8), so norm-relative steering does not transfer.
#
# This pushes steering to 30x/60x/100x her probed norm to find a setting that
# clears rotation_is_substantial (>= 0.30), then asks whether QA actually
# falls -- and, critically, whether it falls SELECTIVELY (concept down,
# simdom held) rather than as general damage.
#
# COMPARISON POINTS ALREADY IN HAND, on Gal's control (null-result table):
#   L5 steer=2000 (~24x norm): QA_test 44.0%, simdom 50.0%, rotation 0.339
#   L5 steer=5000 (~60x norm): QA_test 48.0%, simdom 52.0%, rotation 0.366
# i.e. our control DOES reach real rotation at high steering and still
# shows no erasure. If hers does the same, the null is closed on both.

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

MODEL_PATH=/home/dcor/galbarak2/hf_cache/hub/models--dhgottesman--LMEnt-1B-2E/snapshots/d0ce274557365c983032af9d7f8ae59851df18d8/step219344
CONCEPT="Ancient Rome"
PROBE="$RUNS/rmu_daniela2e_probe_892663/probe.json"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 || echo "no nvidia-smi"
echo "reusing probe: $PROBE"

# layer_id:layer_ids:multiplier
for spec in "5:3 4 5:30" "5:3 4 5:60" "5:3 4 5:100" "6:4 5 6:100"; do
  LID=$(echo "$spec" | cut -d: -f1)
  LIDS=$(echo "$spec" | cut -d: -f2)
  MULT=$(echo "$spec" | cut -d: -f3)

  STEER=$("$PY" -c "
import json
d = json.load(open('$PROBE'))
norm = d['layers']['$LID']['mean_residual_norm']
print(norm * $MULT)
")
  if [ -z "$STEER" ]; then
    echo "FATAL: could not compute steering for L$LID x$MULT" >&2
    continue
  fi

  OUT="$RUNS/rmu_dani_L${LID}x${MULT}_${SLURM_JOB_ID}"
  echo
  echo "================ L$LID x${MULT} norm : steering=$STEER ================"
  "$PY" "$ROOT/mlp_erasure/rmu.py" \
    --model "$MODEL_PATH" --concept "$CONCEPT" \
    --concept-sentences "$DATA/concept_sentences.json" \
    --neutral-sentences "$DATA/neutral_sentences.json" \
    --dtype fp32 --lr 1e-4 --alpha 100 --steering "$STEER" \
    --layer-id "$LID" --layer-ids $LIDS \
    --batch-size 4 --max-num-batches 150 \
    --out "$OUT" --sanity
  echo "rmu exit $? (exit 1 just means a sanity gate failed; eval still runs)"

  if [ -d "$OUT/model" ]; then
    "$PY" "$ROOT/ember_eval/score_ember_mc.py" \
      --model "$OUT/model" --subfolder "" \
      --ember-data "$DATA" \
      --concepts "$CONCEPT" --sets qa_test simdom_test \
      --out "$OUT/eval.json"
    echo "score exit $?"
    rm -rf "$OUT/model"
    echo "checkpoint deleted for L$LID x$MULT"
  else
    echo "NO MODEL SAVED for L$LID x$MULT"
  fi
done

echo DONE
