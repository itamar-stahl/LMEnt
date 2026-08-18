#! /bin/sh
#SBATCH --job-name=ember-twins
#SBATCH --output=/home/dcor/galbarak2/LMEnt-ember/ember_eval/results/%x_%j.out
#SBATCH --error=/home/dcor/galbarak2/LMEnt-ember/ember_eval/results/%x_%j.err
#SBATCH --account=gpu-research
#SBATCH --partition=killable
#SBATCH --constraint=a5000
#SBATCH --time=180
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
# a5000: 24 GiB is ample for a 1B in fp32 inference, and killable's a5000 nodes
# sit in that partition only, so nothing outranks this job and it cannot be
# preempted. fp32 rather than bf16 because scoring is cheap and log-likelihood
# comparisons between two near-identical models should not turn on rounding.

EVAL=/home/dcor/galbarak2/LMEnt-ember/ember_eval
PY=/home/morg/NLP_2526b/galbarak2/anaconda3/envs/lment/bin/python
export HF_HOME=/home/dcor/galbarak2/hf_cache
export TOKENIZERS_PARALLELISM=false

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# --subfolder "" because these are local directories, not HF repos with a
# stepNNNN subfolder like dhgottesman/LMEnt-1B-1E.
for m in control noporn; do
  echo "############################ $m ############################"
  "$PY" "$EVAL/score_ember_mc.py" \
    --model "/vol/scratch/galbarak2/hf-models/lment-1b-$m" \
    --subfolder "" \
    --sets qa_train qa_test simdom_train simdom_test \
    --dtype float32 \
    --out "$EVAL/results/twins4_${m}_${SLURM_JOB_ID}.json"
done
