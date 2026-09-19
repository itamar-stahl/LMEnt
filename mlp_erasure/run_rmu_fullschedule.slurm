#! /bin/sh
#SBATCH --job-name=rmu-fullsched
#SBATCH --output=/home/dcor/galbarak2/runs/mlp_erasure/%x_%j.out
#SBATCH --error=/home/dcor/galbarak2/runs/mlp_erasure/%x_%j.err
#SBATCH --account=gpu-research
#SBATCH --partition=killable
#SBATCH --exclude=rack-bgw-dgx1,rack-gww-dgx1,rack-omerl-g01,n-202,n-203,n-204,n-205,n-602,n-301
#SBATCH --time=180
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32000
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#
# GAP 1: RMU has never actually run as specified on this model.
#
# EVERY cell so far failed `ran_enough_steps` at 71/150 (47%). The cap is
# arithmetic: 300 Rome sentences -> 283 pass the length filter -> at
# batch-size 4 that is 71 batches, and n = min(max_num_batches, forget).
# The published grid assumes 150 steps. So "RMU does nothing here" has
# really meant "RMU at 47% of its schedule does nothing here".
#
# At batch-size 1, 283 sentences give 283 batches, so min(150, 283) = 150 --
# the FULL published schedule, for the first time. The tradeoff is explicit
# and worth stating: with 300 sentences you can match the reference's step
# count OR its batch size, not both. Batch 2 (141 batches, 94%) is the
# compromise cell; batch 1 is the only way to reach 100%.
#
# Second axis: alpha. alpha=100 is the TOP of the reference grid -- the
# heaviest retain penalty and therefore the LEAST unlearning available
# (mlp_erasure/tests asserts exactly this). Every cell run so far used it.
# alpha=10 is the aggressive end.
#
# Cells (all L5-hi, the depth/steering that reached real rotation on control):
#   b1_a100   batch 1, alpha 100  -> full schedule, conservative alpha
#   b4_a10    batch 4, alpha 10   -> short schedule, aggressive alpha
#   b1_a10    batch 1, alpha 10   -> full schedule AND aggressive: the
#                                    maximum-unlearning cell available
#   b2_a100   batch 2, alpha 100  -> the 94% compromise, batch closer to ref
#
# If the maximum-unlearning cell still lands at control with a substantial
# rotation, the RMU objection is closed properly.

ROOT=/home/dcor/galbarak2/LMEnt-erasure-fix
PY=/home/morg/NLP_2526b/stahli/anaconda3/envs/lment/bin/python
RUNS=/home/dcor/galbarak2/runs/mlp_erasure

MODEL=/home/dcor/galbarak2/hf-models/lment-1b-control-2e-b131k
CONCEPT="Ancient Rome"
DATA="$ROOT/Ember-on-LMEnt/data"
STEER=831.585323965282        # L5 steering_high on the control (norm x10)

export HF_HOME=/home/dcor/galbarak2/hf_cache
export TOKENIZERS_PARALLELISM=false

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 || echo "no nvidia-smi"

# tag | batch | alpha
printf '%s\n' \
  "b1_a100|1|100" \
  "b4_a10|4|10" \
  "b1_a10|1|10" \
  "b2_a100|2|100" \
| while IFS='|' read -r tag bs alpha; do
  OUT="$RUNS/rmu_fullsched_${tag}_${SLURM_JOB_ID}"
  echo
  echo "================ L5-hi $tag : batch=$bs alpha=$alpha steering=$STEER ================"
  "$PY" "$ROOT/rmu.py" \
    --model "$MODEL" --concept "$CONCEPT" \
    --concept-sentences "$DATA/concept_sentences.json" \
    --neutral-sentences "$DATA/neutral_sentences.json" \
    --dtype fp32 --lr 1e-4 --alpha "$alpha" --steering "$STEER" \
    --layer-id 5 --layer-ids 3 4 5 \
    --batch-size "$bs" --max-num-batches 150 \
    --out "$OUT" --sanity
  echo "rmu exit $? (1 just means a sanity gate failed; eval still runs)"

  if [ -d "$OUT/model" ]; then
    "$PY" "$ROOT/ember_eval/score_ember_mc.py" \
      --model "$OUT/model" --subfolder "" \
      --ember-data "$DATA" \
      --concepts "$CONCEPT" --sets qa_test simdom_test \
      --out "$OUT/eval.json"
    echo "score exit $?"
    rm -rf "$OUT/model"
    echo "checkpoint deleted"
  else
    echo "NO MODEL SAVED for $tag"
  fi
done

echo DONE
