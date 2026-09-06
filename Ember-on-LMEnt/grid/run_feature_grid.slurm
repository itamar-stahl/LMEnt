#! /bin/sh
#SBATCH --job-name=romegrid-features
#SBATCH --output=/home/dcor/galbarak2/lment-ember-grid/logs/%x_%j.out
#SBATCH --error=/home/dcor/galbarak2/lment-ember-grid/logs/%x_%j.err
#SBATCH --account=gpu-research
#SBATCH --partition=killable
#SBATCH --time=240
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --exclude=rack-bgw-dgx1,rack-gww-dgx1,rack-omerl-g01,n-[202-205]
#
# STAGE 1 of the EMBER hyperparameter grid: build embedding features across
# (rank x g_sparsity x seed) and NOTHING else. No Gemma judge, no erasure, no
# evaluation.
#
# WHY THE JUDGE IS NOT IN THIS LOOP. Measured from job 858233: the
# factorization takes 6.55s (`timing.json`), and the Gemma-4-12B judge takes
# most of that run's 49m30s -- a 23 GB load plus two generations per eligible
# feature. Putting the judge inside a 27-cell grid would cost ~20 hours to
# answer a question the 3-minute version answers first. Stage 2 runs the judge
# on the few cells that survive `screen_feature_grid.py`.
#
# WHAT THIS GRID IS FOR. Job 858233 died with "The feature judge selected no
# features above the confidence threshold": at rank 100 / g_sparsity 0.01, all
# 24 features clearing ratio_abs >= 2 were judged too broad -- "collective
# entities and organized social structures", "past tense verbs", "food,
# culinary items and dining". Rome tokens clearly drive certain features
# (ratio_abs up to 7.97, against 1.69 for Pornography, which never cleared the
# filter at all), but those features are generic categories that Roman prose
# activates rather than an Ancient Rome direction. The open question is whether
# ANY (rank, g_sparsity) isolates one. Either answer is publishable; a single
# failed setting is not.
#
# THE THREE AXES.
#
#   rank         100 -> 300 -> 500. The suspicion is that 100 components is too
#                coarse, so each one pools several unrelated senses. |V'| is
#                ~5,100 tokens and d_model is 2,048, so 500 is still a sane
#                factorization.
#
#   g_sparsity   how many tokens each feature keeps. NOT a fraction of the
#                100,352-token model vocabulary -- of |V'|, the vocabulary of
#                the concept+neutral sentences. Measured from 858233's
#                token_features.csv: 0.01 gives 51 tokens/feature, so |V'| is
#                ~5,100. Hence 0.005 ~ 25 tokens and 0.02 ~ 102. Below 0.005
#                a feature is too thin to interpret at all.
#
#   seed         42/43/44, and this axis is not optional. 858233's loss sat at
#                15848.176758 from iteration ~150 to its early stop at 578 --
#                a fast plateau from `init="random"`, which is what a poor
#                local optimum looks like. With one seed per cell there is no
#                way to tell "this cell is bad" from "this cell's init was
#                bad". (`init_svd` and `init_knn` exist in
#                SparseMatrixFactorization but no config key reaches them.)
#
# COST. One `--ranks 100 300 500` invocation per (g_sparsity, seed) pair, so 9
# model loads rather than 27; the model load dominates everything else here.
#
# WHAT THIS DELIBERATELY DOES NOT DO. It never touches the ablated twin, and it
# computes no accuracy, no erasure and no delta. Cell selection therefore cannot
# be influenced by the size of the effect being measured -- the failure mode
# that retracted the `acc_raw` claims (`ember_eval/EVALUATION.md`) and that
# `completion_eval/metric_bakeoff.py` was written to avoid.

set -u

export LMENT_ROOT=/home/morg/NLP_2526b/galbarak2/LMEnt-ember
EMB=$LMENT_ROOT/Ember-on-LMEnt
OUT=/home/dcor/galbarak2/lment-ember-grid

# The Ancient Rome pair's control -- job 853707, step 54832, batch 131,072.
# NOT hf-models/lment-1b-control-2e, which is the retired Pornography pair's
# control at different hyperparameters and whose twin held out Q291, not Rome.
MODEL=/home/dcor/galbarak2/hf-models/lment-1b-control-2e-b131k
MODEL_KEY=lment-1b-control-2e-b131k
CONCEPT="Ancient Rome"

RANKS="${RANKS:-100 300 500}"
SPARSITIES="${SPARSITIES:-0.005 0.01 0.02}"
SEEDS="${SEEDS:-42 43 44}"

mkdir -p "$OUT/logs" || exit 1

# shellcheck disable=SC1091
. "$EMB/activate_env.sh" || exit 1

export HF_HOME=/home/dcor/galbarak2/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

if [ ! -r "$MODEL/config.json" ]; then
  echo "FATAL: $MODEL is not readable from $(hostname)" >&2
  exit 1
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 || echo "no gpu"
echo "node=$(hostname)  model=$MODEL"
echo "grid: ranks=[$RANKS] sparsities=[$SPARSITIES] seeds=[$SEEDS]"

rc=0
for SP in $SPARSITIES; do
  for SEED in $SEEDS; do
    # One output root per (g_sparsity, seed). `get_pipeline_path` already keys
    # its subdirectories on rank and seed but NOT on g_sparsity, so two
    # sparsities sharing a root would silently overwrite each other's
    # stats_embed.csv and the grid would compare a cell against itself.
    CELL="$OUT/features/sp${SP}_seed${SEED}"
    mkdir -p "$CELL"
    echo ""
    echo "=== g_sparsity=$SP seed=$SEED ranks=[$RANKS] -> $CELL"
    python -m ember.train_mf_features \
      --concepts "$CONCEPT" \
      --ranks $RANKS \
      --seed "$SEED" \
      --model-name "$MODEL" \
      --model-key "$MODEL_KEY" \
      --model-device cuda \
      --fitting-device cuda \
      --concept-json "$EMB/data/concept_sentences.json" \
      --neutral-json "$EMB/data/neutral_sentences.json" \
      --skip-mlp \
      --max-iterations 20000 \
      --g-sparsity "$SP" \
      --k-proj 30 \
      --outdir "$CELL" || rc=1
  done
done

echo ""
echo "=== done rc=$rc"
echo "next: python grid/screen_feature_grid.py --grid-root $OUT/features"
exit $rc
