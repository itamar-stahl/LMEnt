#! /bin/sh
#SBATCH --job-name=step45k-diag
#SBATCH --output=/home/dcor/galbarak2/LMEnt-ember/ember_eval/results/olmes/%x_%j.out
#SBATCH --error=/home/dcor/galbarak2/LMEnt-ember/ember_eval/results/olmes/%x_%j.err
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
# Does the sciq gap between the twins predate the control's H200 window?
#
# The 2-epoch control ran its last 5,331 of 54,832 steps on an H200 while its
# twin ran entirely on H100s (COMPARABILITY.md). The H200 window begins at
# step 49,501, so step45000 is the control's last pure-H100 checkpoint and both
# twins have it. At step45000 the ONLY difference between them is the ablation.
#
# If they already differ on sciq at step45000, the H200 is exonerated and
# retraining its tail would fix nothing. If they agree there and differ at
# step54832, the H200 becomes a real suspect.
#
# Converts the checkpoint then scores it. ~1h, no training.
#
#   sbatch --export=ALL,RUN=<run folder>,NAME=<hf name> run_step45000_diag.slurm

: "${RUN:?set RUN}"
: "${NAME:?set NAME}"
STEP_WANTED=step45000
OUT=/home/dcor/galbarak2/LMEnt-ember/ember_eval/results/olmes
HFDIR=/home/dcor/galbarak2/hf-models
OLMES=/home/morg/NLP_2526b/galbarak2/LMEnt/olmes
PY=/home/morg/NLP_2526b/stahli/anaconda3/envs/lment/bin/python

export HF_HOME=/home/dcor/galbarak2/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# convert_to_hf.sh calls bare `python`, which assumes an activated conda env.
# A SLURM job starts with a bare PATH and gets "python: not found" (jobs
# 782128/782129). Putting the env's bin first satisfies that assumption without
# changing the shared script's contract for anyone calling it interactively.
PATH="$(dirname "$PY"):$PATH"
export PATH

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 || true
echo "node=$(hostname) run=$RUN name=$NAME step=$STEP_WANTED"

# --- convert, unless already done -------------------------------------------
if [ -r "$HFDIR/$NAME/config.json" ]; then
  echo "already converted: $HFDIR/$NAME"
else
  cd /home/morg/NLP_2526b/galbarak2/LMEnt/Untaught || exit 1
  STEP="$STEP_WANTED" sh convert_to_hf.sh "$RUN" "$NAME" || {
    echo "FATAL: conversion failed" >&2; exit 1; }
fi

# The conversion is the step most likely to go quietly wrong, so check the
# result is loadable and has the expected vocabulary before scoring it.
"$PY" - "$HFDIR/$NAME" <<'PYCHK' || exit 1
import json, sys, os
d = sys.argv[1]
cfg = json.load(open(os.path.join(d, "config.json")))
print(f"[check] {cfg['model_type']} vocab={cfg['vocab_size']} layers={cfg['num_hidden_layers']}")
assert cfg["model_type"] == "olmo2" and cfg["vocab_size"] == 100352, "unexpected config"
assert os.path.exists(os.path.join(d, "tokenizer.json")), "no tokenizer"
print("[check] conversion looks sane")
PYCHK

# --- score on sciq, same task spec as jobs 780463-780465 --------------------
cd "$OLMES" || exit 1
export PYTHONPATH="$OLMES:$PYTHONPATH"
CMD=$("$PY" -m oe_eval.launch --dry-run \
        --model "$NAME" --model-type hf \
        --model-args "{\"model_path\":\"$HFDIR/$NAME\"}" \
        --task sciq::olmo1 --batch-size 8 \
        --output-dir "$OUT/${NAME}_${SLURM_JOB_ID}" 2>&1 | sed -n 's/.*Command: //p' | tail -1)
[ -z "$CMD" ] && { echo "FATAL: could not resolve task spec" >&2; exit 1; }
CMD=$(printf '%s' "$CMD" | sed "s|^python |$PY |")
echo "resolved: $CMD"
eval "$CMD"
echo "=== diag rc=$?"
