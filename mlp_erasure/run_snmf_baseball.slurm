#! /bin/sh
#SBATCH --job-name=snmf-baseball
#SBATCH --output=/home/dcor/galbarak2/runs/mlp_erasure/%x_%j.out
#SBATCH --error=/home/dcor/galbarak2/runs/mlp_erasure/%x_%j.err
#SBATCH --account=gpu-research
#SBATCH --partition=gpu-h100-killable
#SBATCH --exclude=n-102
#SBATCH --time=45
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32000
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#
# Baseball SNMF re-run on the FIXED code, on the control model (same as the
# Rome re-run) -- only feature cache available is ratio-only, 58 features,
# seed 42 (snmf_baseball_887805; this is the cell whose original -8pt at
# delta=4 did NOT replicate under a 3-seed check, see
# lment-crosstopic-mlp-erasure.md). Full delta x range-cell grid on train
# first, then confirm the best non-amplifying cell on held-out test.

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

MODEL_PATH=/home/dcor/galbarak2/hf-models/lment-1b-control-2e-b131k
CONCEPT="Baseball"
SNMF_OUT="$RUNS/snmf_baseball_887805"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 || echo "no nvidia-smi"

echo "=== Baseball, control model, full grid on train ==="
"$PY" "$ROOT/mlp_erasure/sweep_delta.py" \
  --model "$MODEL_PATH" --snmf-out "$SNMF_OUT" \
  --concept "$CONCEPT" \
  --questions "$DATA/mc_questions.json" \
  --dtype fp32 --gamma 0.95 \
  --out "$RUNS/sweep_baseball_train_${SLURM_JOB_ID}.csv"
echo "train exit $?"

echo "=== Baseball, control model, cell 0 delta 1.0, held-out test ==="
"$PY" "$ROOT/mlp_erasure/sweep_delta.py" \
  --model "$MODEL_PATH" --snmf-out "$SNMF_OUT" \
  --concept "$CONCEPT" \
  --questions "$DATA/mc_questions.json" \
  --subset QA_test --simdom-subset SimdomQA_test \
  --deltas 1.0 --range-cells 0 \
  --dtype fp32 --gamma 0.95 \
  --out "$RUNS/confirm_baseball_test_${SLURM_JOB_ID}.csv"
echo "test exit $?"

echo DONE
