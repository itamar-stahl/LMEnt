#! /bin/sh
#SBATCH --job-name=snmf-bigneg
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
# LARGE sign-preserving amplification. Job 893271 only reached x3
# (delta -2), while the sign-FLIPPED side was already tested to x9
# (delta +10) and x19 (delta +20). The two sides were never compared at
# matched magnitude, so "strengthening does nothing" was only established
# for small strengthening.
#
# scale = |1 - delta|, sign of (1 - delta) is what flips:
#     delta  +10 -> x -9    (already run: sign FLIPPED)
#     delta   -8 -> x +9    <-- MAGNITUDE-MATCHED PAIR, sign KEPT
#     delta  -19 -> x +20
#     delta  -49 -> x +50
#     delta  -99 -> x +100
#
# The -8 vs +10 pair is the load-bearing comparison: identical scale factor,
# opposite sign. Everything else maps how far you can push before the model
# simply breaks.
#
# What to read off it:
#   * Does concept accuracy ever RISE? If these directions carry the concept,
#     strengthening them by 9-100x should help, or at least move it.
#   * At whatever magnitude the model does start to degrade, is the damage
#     SELECTIVE (concept falls, simdom held) or general (both fall)? On
#     Daniela's model the sign-flipped x9 was non-selective -- general damage.
#   * Expect garbage/chance-level scores at x50-x100; that is informative, it
#     establishes the scale at which these directions affect the model at all.
#
# Includes Daniela's checkpoint as a 4th arm, since she is the more fragile
# model and is where any magnitude effect should show up first.

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
DANI=/home/dcor/galbarak2/hf_cache/hub/models--dhgottesman--LMEnt-1B-2E/snapshots/d0ce274557365c983032af9d7f8ae59851df18d8/step219344

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 || echo "no nvidia-smi"

# tag | snmf_out | concept | model
printf '%s\n' \
  "rome_judge|$RUNS/snmf_romeband_871493|Ancient Rome|$CTRL" \
  "rome_ratio44|$RUNS/snmf_rome_883188|Ancient Rome|$CTRL" \
  "baseball58|$RUNS/snmf_baseball_887805|Baseball|$CTRL" \
  "daniela51|$RUNS/snmf_daniela2e_rome_892662|Ancient Rome|$DANI" \
| while IFS='|' read -r tag snmf_out concept model; do
  echo
  echo "================ $tag ($concept) ================"
  "$PY" "$ROOT/mlp_erasure/sweep_delta.py" \
    --model "$model" --snmf-out "$snmf_out" \
    --concept "$concept" \
    --questions "$DATA/mc_questions.json" \
    --deltas -8.0 -19.0 -49.0 -99.0 \
    --dtype fp32 --gamma 0.95 \
    --out "$RUNS/bigneg_${tag}_${SLURM_JOB_ID}.csv"
  echo "$tag exit $?"
done

echo DONE
