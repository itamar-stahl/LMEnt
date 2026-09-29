#! /bin/sh
#SBATCH --job-name=emb-health
#SBATCH --output=/home/morg/NLP_2526b/galbarak2/LMEnt-initfix/ember_eval/results/%x_%j.out
#SBATCH --error=/home/morg/NLP_2526b/galbarak2/LMEnt-initfix/ember_eval/results/%x_%j.err
#SBATCH --account=gpu-research
#SBATCH --partition=gpu-h100-killable
#SBATCH --time=60
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16000
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --exclude=rack-bgw-dgx1,rack-gww-dgx1,rack-omerl-g01,n-301,n-303
#
# Read the input embedding matrix out of two checkpoints of one run and ask
# whether it learned anything beyond weight-decay shrinkage. See
# embedding_health.py for the test itself.
#
#   sbatch --export=ALL,RUN=<run_folder>,STEPS=54832,OUT_TAG=norome2e \
#          run_embedding_health.slurm
#
# Not on the login node: importing torch off /home/morg there takes >20 min
# under NFS contention (run_rome_heldout.slurm records the same).
#
# CPU-only work -- the tensors are ~800 MB each and never touch the GPU -- so
# this asks for the smallest allocation that schedules anywhere.
#
# NOT plain `killable`, despite this being the cheapest job in the suite. That
# partition was 41 deep on 2026-09-06 and, with the --exclude line below, its
# only backfill slot was rack-bgw-dgx1 -- an excluded node -- so sbatch --test-only
# projected a start 7 hours out. cpu-killable would be the natural home and
# refuses the gpu-research account outright ("Invalid account or
# account/partition combination"). gpu-h100-killable was 0 deep with 4 free
# cards on n-102 and starts immediately. It is tier 10, so this cannot preempt
# anything, and 16 GB against the measured ~3 GB peak leaves the node's memory
# free for the training job beside it.

set -u

export LMENT_ROOT=/home/morg/NLP_2526b/galbarak2/LMEnt-initfix
EVAL=$LMENT_ROOT/ember_eval
RUNS=$LMENT_ROOT/Untaught/runs
PY=/home/morg/NLP_2526b/stahli/anaconda3/envs/lment/bin/python

: "${RUN:?set RUN to a run folder name under Untaught/runs}"
: "${OUT_TAG:?set OUT_TAG}"
: "${STEPS:?set STEPS to the number of steps actually trained}"

export HF_HOME=/home/dcor/galbarak2/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
. $LMENT_ROOT/Untaught/framework/env.sh

CKPT_ROOT="$(ls -d "$RUNS/$RUN"/checkpoints/*/ | head -1)"

# step0 is not always in the same run folder as the final checkpoint. A run that
# resumed -- control resumed from step34000 of 850054 -- starts its own
# checkpoints at the resume step, so $RUN holds no step0 at all and the init
# draw lives in the predecessor. INIT_RUN names that folder; it defaults to RUN
# for a run that trained straight through from random init.
INIT_ROOT="$(ls -d "$RUNS/${INIT_RUN:-$RUN}"/checkpoints/*/ | head -1)"
INIT="${INIT_ROOT}step0"
FINAL="${FINAL_STEP:+${CKPT_ROOT}${FINAL_STEP}}"
if [ -z "$FINAL" ]; then
  FINAL="${CKPT_ROOT}$(ls -1 "$CKPT_ROOT" | grep -E '^step[0-9]+$' | sort -t p -k2 -n | tail -1)"
fi

echo "node=$(hostname)  run=$RUN"
echo "init=$INIT"
echo "final=$FINAL"

mkdir -p "$EVAL/results"
"$PY" "$EVAL/embedding_health.py" \
  --init "$INIT" \
  --final "$FINAL" \
  --steps "$STEPS" \
  --json-out "$EVAL/results/emb_health_${OUT_TAG}_${SLURM_JOB_ID}.json"
