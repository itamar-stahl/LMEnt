#! /bin/sh
#SBATCH --job-name=snmf-ai-grid
#SBATCH --output=/home/dcor/galbarak2/runs/mlp_erasure/%x_%j.out
#SBATCH --error=/home/dcor/galbarak2/runs/mlp_erasure/%x_%j.err
#SBATCH --account=gpu-research
#SBATCH --partition=killable
#SBATCH --exclude=rack-bgw-dgx1,rack-gww-dgx1,rack-omerl-g01,n-202,n-203,n-204,n-205,n-602,n-301
#SBATCH --time=60
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32000
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#
# Re-run of AI stage 1 only. Job 895359 died mid-grid with
# "CUDA error: unspecified launch failure" on n-301 after cell 0, and every
# later process on that node fell back to CPU. Stages 2 and 3 there are still
# valid (fresh model loads, and stage 3 reproduced the known control baseline
# 44.0/54.0 exactly), but cells 1 and 2 of the train grid were never scored.
# n-301 excluded here as a precaution -- it is still MIXED, not drained, so
# the fault looks transient rather than a dead card.

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

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 || echo "no nvidia-smi"

"$PY" "$ROOT/mlp_erasure/sweep_delta.py" \
  --model /home/dcor/galbarak2/hf-models/lment-1b-control-2e-b131k \
  --snmf-out "$RUNS/snmf_ai_887806" \
  --concept "Artificial intelligence" \
  --questions "$ROOT/Ember-on-LMEnt/data/mc_questions.json" \
  --dtype fp32 --gamma 0.95 \
  --out "$RUNS/sweep_ai_train_retry_${SLURM_JOB_ID}.csv"
echo "exit $?"
echo DONE
