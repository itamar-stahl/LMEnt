#! /bin/sh
#SBATCH --job-name=ember-epochs
#SBATCH --output=/home/dcor/galbarak2/LMEnt-ember/ember_eval/results/%x_%j.out
#SBATCH --error=/home/dcor/galbarak2/LMEnt-ember/ember_eval/results/%x_%j.err
#SBATCH --account=gpu-research
#SBATCH --partition=killable
#SBATCH --time=240
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
# Same target as run_eval.slurm: no --constraint, because a 1B in fp32 (~4.4 GB)
# fits any card in killable and this takes whatever is free.
#
# WHY: choosing how many epochs to train the next twin pair for. Table 3 of the
# paper reports general benchmarks, which is the wrong instrument -- what decides
# the epoch count is how much a model knows about *the concept being ablated*,
# since that is the knowledge the ablation has to remove. The authors released
# 1E/2E/4E/6E of the same 1B on the same corpus, so the curve can be measured
# instead of guessed, in under an hour, before committing 36 GPU-hours per twin.
#
# All four sets (3,600 questions) rather than run_eval.slurm's two, so these are
# directly comparable to the twins' own twins4_*.json records.

EVAL=/home/dcor/galbarak2/LMEnt-ember/ember_eval
PY=/home/morg/NLP_2526b/stahli/anaconda3/envs/lment/bin/python

export HF_HOME=/home/dcor/galbarak2/hf_cache
export HF_HUB_OFFLINE=1          # every checkpoint was fetched before submission
export TOKENIZERS_PARALLELISM=false

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "no nvidia-smi"

# One epoch is 109,672 steps at the paper's 32,768-token batch, so the final
# checkpoint of the NE model is step (109672 * N).
for spec in "1E 109672" "2E 219344" "4E 438688" "6E 658032"; do
  set -- $spec
  epochs="$1"; step="$2"
  echo "######################## LMEnt-1B-$epochs (step$step) ########################"
  "$PY" "$EVAL/score_ember_mc.py" \
    --model "dhgottesman/LMEnt-1B-$epochs" \
    --subfolder "step$step" \
    --sets qa_train qa_test simdom_train simdom_test \
    --dtype float32 \
    --out "$EVAL/results/epochs_${epochs}_${SLURM_JOB_ID}.json" || echo "FAILED: $epochs"
done
