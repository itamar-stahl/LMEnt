#! /bin/sh
#SBATCH --job-name=postember-refit
#SBATCH --output=/home/dcor/galbarak2/runs/mlp_erasure/%x_%j.out
#SBATCH --error=/home/dcor/galbarak2/runs/mlp_erasure/%x_%j.err
#SBATCH --account=gpu-research
#SBATCH --partition=killable
#SBATCH --exclude=rack-bgw-dgx1,rack-gww-dgx1,rack-omerl-g01,n-202,n-203,n-204,n-205,n-602
#SBATCH --time=120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32000
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#
# VARIANT 2: SNMF re-derived FROM the EMBER-erased model, not inherited from
# the control.
#
# Variant 1 reuses control-derived directions, which is the right test of
# "does the same intervention still bite after EMBER". But it is not the test
# a practitioner would actually run: they would factorize on the model in
# front of them. Since EMBER changed the embeddings, the ACTIVATIONS feeding
# the MLPs changed, so a fresh semi-NMF sees a different decomposition even
# though the MLP weights are untouched.
#
# If a fresh factorization on the erased model finds directions that DO move
# concept accuracy where the inherited ones do not, that is a meaningful
# methodological finding: MLP erasure has to be re-derived after an embedding
# edit. If both are flat, the composition genuinely adds nothing.
#
# Ratio-only selection (--skip-llm), consistent with every other re-run in
# this series and avoiding the flaky Gemma judge.

ROOT=/home/dcor/galbarak2/LMEnt-erasure-fix
PY=/home/morg/NLP_2526b/stahli/anaconda3/envs/lment/bin/python
RUNS=/home/dcor/galbarak2/runs/mlp_erasure
HF=/home/dcor/galbarak2/hf-models
DATA="$ROOT/Ember-on-LMEnt/data"

export HF_HOME=/home/dcor/galbarak2/hf_cache
export TOKENIZERS_PARALLELISM=false

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 || echo "no nvidia-smi"

# tag | model | concept
printf '%s\n' \
  "romeerased|$HF/lment-1b-rome-erased-b131k|Ancient Rome" \
  "bberased|$HF/lment-1b-baseball-erased-b131k|Baseball" \
| while IFS='|' read -r tag model concept; do
  OUT="$RUNS/snmf_refit_${tag}_${SLURM_JOB_ID}"
  echo
  echo "================ $tag ($concept): fresh factorize on the ERASED model ================"
  "$PY" "$ROOT/snmf.py" factorize \
    --model "$model" --concept "$concept" \
    --concept-sentences "$DATA/concept_sentences.json" \
    --neutral-sentences "$DATA/neutral_sentences.json" \
    --out "$OUT" --layers 4 5 6 --dtype fp32
  echo "factorize exit $?"

  "$PY" "$ROOT/snmf.py" select --out "$OUT" --concept "$concept" --skip-llm
  echo "select exit $?"

  echo "--- delta 1.0 and -19.0, with permutation controls, TEST split ---"
  "$PY" "$ROOT/random_direction_control.py" \
    --model "$model" --snmf-out "$OUT" \
    --concept "$concept" \
    --questions "$DATA/mc_questions.json" \
    --subset QA_test --simdom-subset SimdomQA_test \
    --deltas 1.0 -19.0 \
    --seeds 1 2 3 \
    --gamma 0.95 --dtype fp32 \
    --out "$RUNS/postember_refit_${tag}_${SLURM_JOB_ID}.csv"
  echo "$tag exit $?"
done

echo DONE
