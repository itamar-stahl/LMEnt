#! /bin/sh
#SBATCH --job-name=lextail
#SBATCH --output=/home/dcor/galbarak2/lment-rome-check/logs/%x_%j.out
#SBATCH --error=/home/dcor/galbarak2/lment-rome-check/logs/%x_%j.err
#SBATCH --account=gpu-research
#SBATCH --partition=killable
#SBATCH --time=120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=64000
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --exclude=rack-bgw-dgx1,rack-gww-dgx1,rack-omerl-g01
#
# Chunk-level version of the lexical test in ERASURE_RESULTS.md. No model is
# loaded and no GPU work is done -- the GPU is requested only because `killable`
# schedules GPU jobs; the script reads token ids from the dataset and joins them
# to per-chunk losses that already exist.
set -u
export LMENT_ROOT=/home/morg/NLP_2526b/galbarak2/LMEnt-initfix
LMENT=$LMENT_ROOT
PY=/home/morg/NLP_2526b/stahli/anaconda3/envs/lment/bin/python
export HF_HOME=/home/dcor/galbarak2/hf_cache
export PYTHONUNBUFFERED=1
. $LMENT/Untaught/framework/env.sh
export PYTHONPATH="$OLMO_CORE_SRC:${PYTHONPATH:-}"
echo "node=$(hostname)  LMENT_ROOT=$LMENT_ROOT"
R=/home/dcor/galbarak2/lment-rome-check/results
exec "$PY" "$LMENT/archive/ember_eval/heldout_ppl/lexical_tail.py" \
  --report /home/morg/NLP_2526b/galbarak2/LMEnt-ember/Ember-on-LMEnt/runs/Ancient_Rome_lment-1b-control-2e-b131k_20260908_045242/outputs/report.json \
  --ppl-glob "$R/ppl_rome_erased_d*.json $R/ppl_erased2e_final_870356.json" \
  --control-ppl "$R/ppl_control2e_final_858234.json" \
  --out "$R/lexical_tail_${SLURM_JOB_ID}.json"
