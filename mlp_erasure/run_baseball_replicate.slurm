#! /bin/sh
#SBATCH --job-name=bb-replicate
#SBATCH --output=/home/dcor/galbarak2/runs/mlp_erasure/%x_%j.out
#SBATCH --error=/home/dcor/galbarak2/runs/mlp_erasure/%x_%j.err
#SBATCH --account=gpu-research
#SBATCH --partition=killable
#SBATCH --exclude=rack-bgw-dgx1,rack-gww-dgx1,rack-omerl-g01,n-202,n-203,n-204,n-205,n-602
#SBATCH --time=90
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32000
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#
# Does the Baseball x20 separation replicate?
#
# Job 893467 found the one positive causal signal in this whole line of work:
# Baseball's seed-42 features at delta -19 (x20 sign-preserving) dropped
# concept 46% -> 30% (chance 25%) while three neuron-permuted controls with
# identical support structure averaged 44.7%. Rome showed no such separation
# at any magnitude -- there, real sits inside the permutation distribution.
#
# Two ways that Baseball result could be an artifact:
#   1. a quirk of the seed-42 semi-NMF decomposition specifically
#   2. train-split overfitting / n=50 noise
#
# This tests both:
#   seed1 (47 feat) and seed2 (50 feat) are INDEPENDENT factorizations of the
#   same concept on the same control model -- only the semi-NMF random init
#   differs (the old 3-seed check, lment-crosstopic-mlp-erasure.md). If the
#   effect is about Baseball knowledge it should reappear there. If it is
#   about one lucky decomposition, it will not.
#
#   seed42 is re-run on the HELD-OUT test split.
#
# Each arm runs real + 3 neuron-permutations, so "did it replicate" is judged
# against that arm's own permutation null, not against seed42's.

ROOT=/home/dcor/galbarak2/LMEnt-erasure-fix
PY=/home/morg/NLP_2526b/stahli/anaconda3/envs/lment/bin/python
RUNS=/home/dcor/galbarak2/runs/mlp_erasure

CTRL=/home/dcor/galbarak2/hf-models/lment-1b-control-2e-b131k
DATA="$ROOT/Ember-on-LMEnt/data"

export HF_HOME=/home/dcor/galbarak2/hf_cache
export TOKENIZERS_PARALLELISM=false

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 || echo "no nvidia-smi"

# tag | snmf_out | qa_subset | simdom_subset
printf '%s\n' \
  "bbseed1_train|$RUNS/snmf_baseball_seed1|QA_train|SimdomQA_train" \
  "bbseed2_train|$RUNS/snmf_baseball_seed2|QA_train|SimdomQA_train" \
  "bbseed42_TEST|$RUNS/snmf_baseball_887805|QA_test|SimdomQA_test" \
| while IFS='|' read -r tag snmf_out sub simsub; do
  echo
  echo "================ $tag ($sub) ================"
  "$PY" "$ROOT/random_direction_control.py" \
    --model "$CTRL" --snmf-out "$snmf_out" \
    --concept "Baseball" \
    --questions "$DATA/mc_questions.json" \
    --subset "$sub" --simdom-subset "$simsub" \
    --deltas -19.0 -99.0 \
    --seeds 1 2 3 \
    --gamma 0.95 --dtype fp32 \
    --out "$RUNS/bbrep_${tag}_${SLURM_JOB_ID}.csv"
  echo "$tag exit $?"
done

echo DONE
