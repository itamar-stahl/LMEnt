#! /bin/sh
#SBATCH --job-name=snmf-negdelta
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
# Partial erasure (delta 0.5) and SIGN-PRESERVING amplification (negative
# deltas), on the control model, train split.
#
# Why negative deltas are a different intervention from delta 4/7/10, and a
# better causal probe. The update is (I - delta*P), so the component scales
# by (1 - delta) BEFORE taking magnitude:
#
#   delta  0.5 -> x +0.5   halved, sign kept        (partial erasure)
#   delta  1.0 -> x  0.0   removed exactly
#   delta  2.0 -> x -1.0   same size, SIGN FLIPPED
#   delta  4.0 -> x -3.0   tripled, SIGN FLIPPED
#   delta -0.5 -> x +1.5   1.5x, sign KEPT          (true strengthening)
#   delta -1.0 -> x +2.0   doubled, sign KEPT
#   delta -2.0 -> x +3.0   tripled, sign KEPT
#
# Every amplifying cell run so far (4, 7, 10) flipped the sign, so it tested
# "what if this direction points the other way", not "what if there is more
# of it". Negative delta is the sign-preserving version: if these features
# really carry the concept, STRENGTHENING them should push concept accuracy
# UP, or at least move it. A flat line at delta 1 AND at delta -1/-2 is much
# stronger evidence the directions are causally inert for this task than a
# flat line at delta 1 alone.
#
# sweep_delta.py calls snmf.ablate_layer directly and so bypasses
# check_delta_erases -- no --allow-amplification needed, and the CSV's
# `erasure_scale`/`amplifies` columns still record |1-delta| honestly.

ROOT=/home/dcor/galbarak2/LMEnt-erasure-fix
PY=/home/morg/NLP_2526b/stahli/anaconda3/envs/lment/bin/python
RUNS=/home/dcor/galbarak2/runs/mlp_erasure

MODEL_PATH=/home/dcor/galbarak2/hf-models/lment-1b-control-2e-b131k
DATA="$ROOT/Ember-on-LMEnt/data"

export HF_HOME=/home/dcor/galbarak2/hf_cache
export TOKENIZERS_PARALLELISM=false

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 || echo "no nvidia-smi"

for spec in \
  "rome_judge:$RUNS/snmf_romeband_871493:Ancient Rome" \
  "rome_ratio44:$RUNS/snmf_rome_883188:Ancient Rome" \
  "baseball58:$RUNS/snmf_baseball_887805:Baseball"
do
  tag=$(echo "$spec" | cut -d: -f1)
  snmf_out=$(echo "$spec" | cut -d: -f2)
  concept=$(echo "$spec" | cut -d: -f3)
  echo
  echo "================ $tag ($concept) ================"
  "$PY" "$ROOT/sweep_delta.py" \
    --model "$MODEL_PATH" --snmf-out "$snmf_out" \
    --concept "$concept" \
    --questions "$DATA/mc_questions.json" \
    --deltas 0.5 -0.5 -1.0 -2.0 \
    --dtype fp32 --gamma 0.95 \
    --out "$RUNS/negdelta_${tag}_${SLURM_JOB_ID}.csv"
  echo "$tag exit $?"
done

echo DONE
