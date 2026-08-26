#! /bin/sh
#SBATCH --job-name=fewshot-probe
#SBATCH --output=/home/dcor/galbarak2/LMEnt-ember/ember_eval/results/fewshot/%x_%j.out
#SBATCH --error=/home/dcor/galbarak2/LMEnt-ember/ember_eval/results/fewshot/%x_%j.err
#SBATCH --account=gpu-research
#SBATCH --partition=killable
#SBATCH --time=90
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --exclude=rack-bgw-dgx1,rack-gww-dgx1,rack-omerl-g01
#
# Ten EMBER questions, open-ended and multiple-choice, 5-shot, all three models
# in one job. Everything the models write is printed verbatim -- the point is to
# see whether few-shot makes them answer at all, which zero-shot did not
# (EVALUATION.md), and a score alone cannot show that.

FP=/home/morg/NLP_2526b/galbarak2/LMEnt/ember_eval/fewshot_probe
OUT=/home/dcor/galbarak2/LMEnt-ember/ember_eval/results/fewshot
PY=/home/morg/NLP_2526b/stahli/anaconda3/envs/lment/bin/python
SNAP=$(ls -d /home/dcor/galbarak2/hf_cache/hub/models--dhgottesman--LMEnt-1B-2E/snapshots/*/step219344 2>/dev/null | head -1)

export HF_HOME=/home/dcor/galbarak2/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 || true
echo "node=$(hostname)  shots=${SHOTS:-5}  n=${N:-10}  balanced=${BALANCED:-no}"

rc=0
for spec in \
  "control2e:/home/dcor/galbarak2/hf-models/lment-1b-control-2e" \
  "noporn2e:/home/dcor/galbarak2/hf-models/lment-1b-noporn-2e" \
  "released2e:$SNAP" ; do
  label=${spec%%:*}
  path=${spec#*:}
  "$PY" "$FP/fewshot_probe.py" \
    --model "$path" --label "$label" \
    --shots "${SHOTS:-5}" --n "${N:-10}" ${BALANCED:+--balanced} \
    --out "$OUT/fewshot_${label}_${SLURM_JOB_ID}.json" || rc=1
done
echo "=== fewshot rc=$rc"
exit $rc
