#! /bin/sh
#SBATCH --job-name=olmes-probe
#SBATCH --output=/home/dcor/galbarak2/LMEnt-ember/ember_eval/results/olmes/%x_%j.out
#SBATCH --error=/home/dcor/galbarak2/LMEnt-ember/ember_eval/results/olmes/%x_%j.err
#SBATCH --account=gpu-research
#SBATCH --partition=killable
#SBATCH --time=120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=48000
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --exclude=rack-bgw-dgx1,rack-gww-dgx1,rack-omerl-g01
#
# Probe: does the vendored OLMES harness run here at all?
# The env sits OUTSIDE olmes' declared pins (it wants transformers <4.50, this
# env has 4.56.2) and B200.md lists the olmes path as untested. So this job is
# a feasibility test first and a measurement second.
#
# sciq::olmo1 chosen deliberately: multiple choice (no generation, which these
# base models cannot do), 1000 validation questions, and it is the task in the
# paper's Table 3 (0.714 at 1E, 0.770 at 6E) so our numbers are comparable.
#
#   sbatch --export=ALL,MODEL_PATH=/path,OUT_TAG=olmes_control2e run_olmes.slurm

OLMES=/home/morg/NLP_2526b/galbarak2/LMEnt/olmes
OUT=/home/dcor/galbarak2/LMEnt-ember/ember_eval/results/olmes
PY=/home/morg/NLP_2526b/stahli/anaconda3/envs/lment/bin/python

: "${MODEL_PATH:?set MODEL_PATH}"
: "${OUT_TAG:?set OUT_TAG}"

export HF_HOME=/home/dcor/galbarak2/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONPATH="$OLMES:$PYTHONPATH"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 || echo "no nvidia-smi"
echo "node=$(hostname) model=${MODEL_PATH} tag=${OUT_TAG}"

cd "$OLMES" || exit 1

# oe_eval has two layers. run_eval is the inner one and only knows the CLASS
# registry, so a config alias like "sciq::olmo1" reaches it as a literal
# task_name and dies with "not found in the task registry" (jobs 780445/780446).
# launch.py is the layer that expands an alias against TASK_CONFIGS -- but it
# only launches to AI2's Beaker, so it cannot run the job here. What it can do
# is --dry-run and print the resolved command, which is what we execute.
# Doing it this way means any alias or suite name works, not just a hardcoded one.
# deliberately unquoted below so a space-separated list becomes several
# --task arguments; launch.py resolves each against TASK_CONFIGS.
TASK_SPEC="${TASK:-sciq::olmo1}"
RUN_DIR="$OUT/${OUT_TAG}_${SLURM_JOB_ID}"

CMD=$("$PY" -m oe_eval.launch --dry-run \
        --model "${OUT_TAG}" --model-type hf \
        --model-args "{\"model_path\":\"${MODEL_PATH}\"}" \
        --task $TASK_SPEC --batch-size "${BATCH:-8}" \
        --output-dir "$RUN_DIR" 2>&1 | sed -n 's/.*Command: //p' | tail -1)

if [ -z "$CMD" ]; then
  echo "FATAL: could not resolve task spec '$TASK_SPEC' via oe_eval.launch" >&2
  exit 1
fi
echo "resolved: $CMD"

# ${CMD/#python/$PY} is a bash-ism and this script runs under /bin/sh (dash),
# which answers "Bad substitution" -- jobs 780449-780451. sed is POSIX.
CMD=$(printf '%s' "$CMD" | sed "s|^python |$PY |")
eval "$CMD"
echo "=== olmes rc=$?"
