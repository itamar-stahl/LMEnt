#! /bin/sh
#SBATCH --job-name=snmf-randctl
#SBATCH --output=/home/dcor/galbarak2/runs/mlp_erasure/%x_%j.out
#SBATCH --error=/home/dcor/galbarak2/runs/mlp_erasure/%x_%j.err
#SBATCH --account=gpu-research
#SBATCH --partition=killable
#SBATCH --exclude=rack-bgw-dgx1,rack-gww-dgx1,rack-omerl-g01,n-202,n-203,n-204,n-205,n-602
#SBATCH --time=90
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32000
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#
# THE CONTROL EXPERIMENT for the large-delta effect found in job 893308.
#
# There, sign-preserving amplification at x20/x100 dropped concept accuracy
# 10-20 points -- but only for the 44- and 58-feature sets; the 6-feature set
# did nothing even at x100. Damage tracked FEATURE COUNT, not concept, which
# looks like capacity damage rather than erasure.
#
# Each feature's entries are randomly PERMUTED across neurons: same values,
# same support size, same magnitude distribution, different neurons. Then the
# identical edit is applied at the identical delta. Three seeds.
#
#   perm drops concept as much as real  -> features not special, it is
#                                          capacity damage, and the whole
#                                          large-delta regime is uninformative
#   real drops substantially more       -> the features do carry something,
#                                          and x20 is where it becomes visible
#
# rome_judge is included as a negative control: it showed nothing at any
# delta, so real and perm should BOTH be flat there.

# ROOT: the checkout whose code this job runs. Walk up from the directory you
# submitted from, so a second checkout runs its own code rather than whichever
# tree was hard-coded here; export ROOT=<checkout> to pin one deliberately.
if [ -z "${ROOT:-}" ]; then
  ROOT=${SLURM_SUBMIT_DIR:-$PWD}
  while [ "$ROOT" != "/" ] && [ ! -f "$ROOT/mlp_erasure/env.sh" ]; do
    ROOT=$(dirname "$ROOT")
  done
fi
[ -f "$ROOT/mlp_erasure/env.sh" ] || { echo "not an LMEnt checkout: ${SLURM_SUBMIT_DIR:-$PWD}; export ROOT=<checkout>" >&2; exit 1; }
. "$ROOT/mlp_erasure/env.sh"

CTRL=/home/dcor/galbarak2/hf-models/lment-1b-control-2e-b131k

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 || echo "no nvidia-smi"

printf '%s\n' \
  "rome_ratio44|$RUNS/snmf_rome_883188|Ancient Rome" \
  "baseball58|$RUNS/snmf_baseball_887805|Baseball" \
  "rome_judge|$RUNS/snmf_romeband_871493|Ancient Rome" \
| while IFS='|' read -r tag snmf_out concept; do
  echo
  echo "================ $tag ($concept) ================"
  "$PY" "$ROOT/mlp_erasure/random_direction_control.py" \
    --model "$CTRL" --snmf-out "$snmf_out" \
    --concept "$concept" \
    --questions "$DATA/mc_questions.json" \
    --deltas -19.0 -99.0 \
    --seeds 1 2 3 \
    --gamma 0.95 --dtype fp32 \
    --out "$RUNS/randctl_${tag}_${SLURM_JOB_ID}.csv"
  echo "$tag exit $?"
done

echo DONE
