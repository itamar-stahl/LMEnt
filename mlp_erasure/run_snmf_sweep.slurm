#! /bin/sh
#SBATCH --job-name=snmf-sweep
#SBATCH --output=/home/dcor/galbarak2/runs/mlp_erasure/%x_%j.out
#SBATCH --error=/home/dcor/galbarak2/runs/mlp_erasure/%x_%j.err
#SBATCH --account=gpu-research
#SBATCH --partition=gpu-h100-killable
#SBATCH --time=90
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32000
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#
# Re-run of the SNMF delta sweep against the FIXED code
# (fix/erasure-delta-and-rmu-steps): every delta ever applied in this project
# was >= 4, which is an amplifying edit (scale |1-delta|), not an erasure.
# sweep_delta.py scores every delta x layer-range cell on QA_train/
# SimdomQA_train (test held out on purpose), reusing the EXISTING
# factorize+select outputs -- the delta bug only lived in erase/verify, not
# in factorize/select, so no need to redo those expensive stages.
#
# Bypasses run_snmf.slurm's own new `sweep` mode: that script still hardcodes
# ROOT=/home/morg/.../LMEnt-mlp (old, unfixed branch), same trap as
# run_rmu.slurm.
#
# Two feature sets, same control checkpoint, both already on disk:
#   snmf_romeband_871493  judge-selected,  6 features, layers 4/5/6
#   snmf_rome_883188      ratio-only,     44 features, layers 4/5/6
#     (this is the one cell that looked like a lead in the null-result
#     writeup -- worth checking at a real delta, not just at 4)

ROOT=/home/dcor/galbarak2/LMEnt-erasure-fix
PY=/home/morg/NLP_2526b/stahli/anaconda3/envs/lment/bin/python
RUNS=/home/dcor/galbarak2/runs/mlp_erasure

: "${MODEL_PATH:=/home/dcor/galbarak2/hf-models/lment-1b-control-2e-b131k}"
: "${CONCEPT:=Ancient Rome}"
: "${DTYPE:=fp32}"
: "${GAMMA:=0.95}"

DATA="$ROOT/Ember-on-LMEnt/data"

export HF_HOME=/home/dcor/galbarak2/hf_cache
export TOKENIZERS_PARALLELISM=false

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 || echo "no nvidia-smi"

for pair in "judge:$RUNS/snmf_romeband_871493" "ratio44:$RUNS/snmf_rome_883188"
do
  tag=${pair%%:*}
  snmf_out=${pair#*:}
  out_csv="$RUNS/sweep_rome_${tag}_${SLURM_JOB_ID}.csv"
  echo "=== sweep [$tag] snmf-out=$snmf_out -> $out_csv ==="
  "$PY" "$ROOT/sweep_delta.py" \
    --model "$MODEL_PATH" --snmf-out "$snmf_out" \
    --concept "$CONCEPT" \
    --questions "$DATA/mc_questions.json" \
    --dtype "$DTYPE" --gamma "$GAMMA" \
    --out "$out_csv"
  echo "=== [$tag] exit $? ==="
done

echo DONE
