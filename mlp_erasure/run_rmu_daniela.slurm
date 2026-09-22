#! /bin/sh
#SBATCH --job-name=rmu-daniela
#SBATCH --output=/home/dcor/galbarak2/runs/mlp_erasure/%x_%j.out
#SBATCH --error=/home/dcor/galbarak2/runs/mlp_erasure/%x_%j.err
#SBATCH --account=gpu-research
#SBATCH --partition=gpu-h100-killable
#SBATCH --exclude=n-102
#SBATCH --time=90
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32000
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#
# RMU erasure, Ancient Rome, on Daniela Gottesman's LMEnt-1B-2E checkpoint.
# Probes this model's own residual norms first (they need not match Gal's
# control), then runs the two depth/steering cells that were the only ones
# to clear the new rotation_is_substantial sanity gate on Gal's control
# (L5-hi, L6-hi) -- using THIS model's own probed steering_high, not the
# control's numbers.

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
PROBE_OUT="$RUNS/rmu_daniela2e_probe_${SLURM_JOB_ID}"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 || echo "no nvidia-smi"

echo "=== probe ==="
"$PY" "$ROOT/mlp_erasure/rmu.py" \
  --model "$MODEL_PATH" --concept "$CONCEPT" \
  --concept-sentences "$DATA/concept_sentences.json" \
  --neutral-sentences "$DATA/neutral_sentences.json" \
  --dtype fp32 --out "$PROBE_OUT" --probe
echo "probe exit $?"
cat "$PROBE_OUT/probe.json"

for LID in 5 6; do
  STEER=$("$PY" -c "
import json
d = json.load(open('$PROBE_OUT/probe.json'))
print(d['layers']['$LID']['steering_high'])
")
  case "$LID" in
    5) LIDS="3 4 5" ;;
    6) LIDS="4 5 6" ;;
  esac
  OUT="$RUNS/rmu_daniela2e_L${LID}hi_${SLURM_JOB_ID}"
  echo "=== L$LID hi: layer_id=$LID layer_ids=$LIDS steering=$STEER ==="
  "$PY" "$ROOT/mlp_erasure/rmu.py" \
    --model "$MODEL_PATH" --concept "$CONCEPT" \
    --concept-sentences "$DATA/concept_sentences.json" \
    --neutral-sentences "$DATA/neutral_sentences.json" \
    --dtype fp32 --lr 1e-4 --alpha 100 --steering "$STEER" \
    --layer-id "$LID" --layer-ids $LIDS \
    --batch-size 4 --max-num-batches 150 \
    --out "$OUT" --sanity
  RMU_EXIT=$?
  echo "L$LID hi rmu exit $RMU_EXIT"

  if [ -d "$OUT/model" ]; then
    echo "=== scoring L$LID hi ==="
    "$PY" "$ROOT/ember_eval/score_ember_mc.py" \
      --model "$OUT/model" --subfolder "" \
      --ember-data "$DATA" \
      --concepts "$CONCEPT" --sets qa_test simdom_test \
      --out "$OUT/eval.json"
    echo "L$LID hi eval exit $?"
    rm -rf "$OUT/model"
  else
    echo "no model dir at $OUT/model for L$LID hi"
  fi
done

echo DONE
