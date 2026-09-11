#! /bin/sh
#SBATCH --job-name=romejudge-cells
#SBATCH --output=/home/dcor/galbarak2/lment-ember-grid/logs/%x_%j.out
#SBATCH --error=/home/dcor/galbarak2/lment-ember-grid/logs/%x_%j.err
#SBATCH --account=gpu-research
#SBATCH --partition=killable
#SBATCH --constraint=a6000
#SBATCH --time=240
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=10000
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#
# STAGE 3: let the Gemma judge arbitrate between the two screens, and stop.
#
# Partition, constraint and the small memory request are copied from
# configs/ember_lment_slurm.yaml, whose comment explains the last one: n-601 is
# the only a6000 with free GPUs and runs close to full on host RAM, so 64000
# never schedules there, and the judge streams its shards to GPU anyway.
#
# THE CELLS. Not the top three of either screen -- both are token-level proxies
# with a demonstrated blind spot, and trusting one to rank would be circular.
# These three test the screens as much as the features:
#
#   100:0.005:42   token_distinctiveness.py's champion. Feature 36 is the
#                  cleanest in the grid -- king, monarchy, emperor, dynasty,
#                  temple, reign, tyranny -- but plausibly "monarchy" rather
#                  than "Ancient Rome", which is exactly the "broad like sport
#                  for basketball" case the judge is told to reject.
#   300:0.005:42   screen_feature_grid.py's champion. Feature 254 carries the
#                  right vocabulary -- Roman, Romans, Rome, Marcus, Gaul,
#                  legion, Julius -- diluted with Kerala, yo, -shaped, IF.
#   500:0.02:44    both screens rate it last. The control: if the judge
#                  accepts something here, neither screen ranks at all.
#
# ONE JUDGE LOAD for all three. The 23 GB Gemma load dominates a cell's cost
# (job 858233 spent most of 49m30s on it, against 6.55s for the
# factorization), so judge_cells.py loads it once and reuses it.
#
# A REJECTION IS A RESULT, NOT A FAILURE. judge_cells.py catches the
# "selected no features above the confidence threshold" ValueError per cell and
# keeps going, so one rejection does not lose the other two cells.
#
# This never touches the ablated twin, an accuracy, an erasure or a delta.

set -u

export LMENT_ROOT=/home/morg/NLP_2526b/galbarak2/LMEnt-ember
EMB=$LMENT_ROOT/Ember-on-LMEnt
# OUT/CONCEPT default to the Ancient Rome grid so an unset environment
# reproduces JUDGE_RESULTS.md. Baseball overrides both.
OUT="${OUT:-/home/dcor/galbarak2/lment-ember-grid}"
CONCEPT="${CONCEPT:-Ancient Rome}"

mkdir -p "$OUT/logs" || exit 1

# shellcheck disable=SC1091
. "$EMB/activate_env.sh" || exit 1

export HF_HOME=/home/dcor/galbarak2/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 || echo "no gpu"
echo "node=$(hostname)"

# CELLS is a space-separated list of rank:g_sparsity:seed. The default is the
# three calibration cells above; GROUP names the output file so parallel jobs
# do not overwrite each other's results.
CELLS="${CELLS:-100:0.005:42 300:0.005:42 500:0.02:44}"
GROUP="${GROUP:-calib}"

set -- --config "$EMB/configs/ember_lment_rome_slurm.yaml" \
       --grid-root "$OUT/features" \
       --concept "$CONCEPT" \
       --out "$OUT/judge_cells_${GROUP}.json"
for c in $CELLS; do set -- "$@" --cell "$c"; done

echo "group=$GROUP concept=$CONCEPT cells=[$CELLS]"
exec python grid/judge_cells.py "$@"
