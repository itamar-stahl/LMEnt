#! /bin/sh
#SBATCH --job-name=heldout-ppl
#SBATCH --output=/home/dcor/galbarak2/LMEnt-ember/ember_eval/results/heldout/%x_%j.out
#SBATCH --error=/home/dcor/galbarak2/LMEnt-ember/ember_eval/results/heldout/%x_%j.err
#SBATCH --account=gpu-research
#SBATCH --partition=killable
#SBATCH --time=180
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64000
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --exclude=rack-bgw-dgx1,rack-gww-dgx1,rack-omerl-g01
#
# Held-out-chunk loss for ONE model, or the indexing check on its own.
# Same partition/account/exclusions as the other eval jobs.
#
# Do NOT run these scripts on the login node. Importing torch + olmo_core off
# /home/morg took >20 min there under NFS contention and never reached the first
# print; a compute node does it in seconds off its own cache.
#
#   sbatch --export=ALL,MODE=check run_heldout_ppl.slurm
#   sbatch --export=ALL,MODE=ppl,MODEL_PATH=/path,OUT_TAG=ppl_control2e run_heldout_ppl.slurm

HP=/home/morg/NLP_2526b/galbarak2/LMEnt/ember_eval/heldout_ppl
OUT=/home/dcor/galbarak2/LMEnt-ember/ember_eval/results/heldout
PY=/home/morg/NLP_2526b/stahli/anaconda3/envs/lment/bin/python

export HF_HOME=/home/dcor/galbarak2/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# The lment env ships a PARTIAL olmo_core in site-packages -- distributed, io,
# stream, utils, and no data/ -- so `from olmo_core.data import ...` fails with
# ModuleNotFoundError: No module named 'olmo_core.data', which reads like the
# package is absent when it is merely incomplete. Job 776523 died this way.
# The complete sources are in the repo, and framework/env.sh is what puts them
# on PYTHONPATH ahead of site-packages; it is the same file the training runs
# source via framework/node/set_node_env.sh, so the dataset is built by exactly
# the code path that trained the twins. Sourcing it beats hardcoding the path.
. /home/morg/NLP_2526b/galbarak2/LMEnt/Untaught/framework/env.sh
echo "PYTHONPATH=$PYTHONPATH"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 || echo "no nvidia-smi"
echo "node=$(hostname) mode=${MODE:-ppl}"

case "${MODE:-ppl}" in
  check)
    exec "$PY" "$HP/check_indexing.py"
    ;;
  ppl)
    : "${MODEL_PATH:?set MODEL_PATH}"
    : "${OUT_TAG:?set OUT_TAG}"
    exec "$PY" "$HP/heldout_ppl.py" \
      --model "$MODEL_PATH" \
      --n-control "${N_CONTROL:-5000}" \
      --dtype float32 \
      --out "$OUT/${OUT_TAG}_${SLURM_JOB_ID}.json"
    ;;
  verify)
    exec "$PY" "$HP/verify_loss.py"
    ;;
  *) echo "FATAL: MODE must be check, ppl or verify" >&2; exit 1 ;;
esac
