#! /bin/sh
#SBATCH --job-name=snmf-deep
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
# GAP 2: SNMF has only ever looked at ONE family of directions.
#
# Every SNMF cell in this project -- original investigation and re-run alike
# -- factorized layers 4/5/6 at rank 100. RMU got a deep/wide battery out to
# layer 17; SNMF never left the 22-34% depth band. So "these directions are
# causally inert" is established only for directions found in three shallow
# layers.
#
# This factorizes at two untouched depth bands and repeats the delta=1 test
# with permutation controls:
#
#   mid    layers  9 10 11   (~50-61% depth)
#   deep   layers 14 15 16   (~78-89% depth)
#
# Rome, on the control, ratio-only selection (--skip-llm), consistent with
# the rest of the re-run. If a deeper decomposition finds directions whose
# exact removal moves concept QA where the shallow ones do not, that is a
# real result and the shallow-only caveat mattered. If both are flat and sit
# inside their permutation nulls, the SNMF conclusion generalises across
# depth.
#
# Note the published range cells are defined on model depth, so `erase` is
# called with EXPLICIT --layers-in/--layers-out here rather than a range
# cell, to avoid the RANGE_CELL silent-skip at these layers.

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

MODEL=/home/dcor/galbarak2/hf-models/lment-1b-control-2e-b131k
CONCEPT="Ancient Rome"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 || echo "no nvidia-smi"

# tag | layers | lo | hi
printf '%s\n' \
  "mid|9 10 11|9|11" \
  "deep|14 15 16|14|16" \
| while IFS='|' read -r tag layers lo hi; do
  OUT="$RUNS/snmf_rome_${tag}_${SLURM_JOB_ID}"
  echo
  echo "================ $tag : factorize layers $layers ================"
  "$PY" "$ROOT/mlp_erasure/snmf.py" factorize \
    --model "$MODEL" --concept "$CONCEPT" \
    --concept-sentences "$DATA/concept_sentences.json" \
    --neutral-sentences "$DATA/neutral_sentences.json" \
    --out "$OUT" --layers $layers --dtype fp32
  echo "factorize exit $?"

  "$PY" "$ROOT/mlp_erasure/snmf.py" select --out "$OUT" --concept "$CONCEPT" --skip-llm
  echo "select exit $?"

  echo "--- delta 1 and -19 with permutation controls, TEST split ---"
  "$PY" "$ROOT/mlp_erasure/random_direction_control.py" \
    --model "$MODEL" --snmf-out "$OUT" \
    --concept "$CONCEPT" \
    --questions "$DATA/mc_questions.json" \
    --subset QA_test --simdom-subset SimdomQA_test \
    --deltas 1.0 -19.0 \
    --seeds 1 2 3 \
    --gamma 0.95 --dtype fp32 \
    --out "$RUNS/deep_${tag}_${SLURM_JOB_ID}.csv"
  echo "$tag exit $?"
done

echo DONE
