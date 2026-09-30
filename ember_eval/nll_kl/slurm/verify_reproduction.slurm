#!/bin/bash
#SBATCH --job-name=vfyCheck
#SBATCH --account=gpu-research
#SBATCH --partition=killable
#SBATCH --gpus=0
#SBATCH --time=40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=48000
#SBATCH --cpus-per-task=4
#SBATCH --exclude=n-102,rack-bgw-dgx1,rack-gww-dgx1,rack-omerl-g01,n-202,n-203,n-204,n-205
#
# Pure numpy -- it reads two already-scored trees and recomputes the comparison,
# so no GPU. `killable` rather than `cpu-killable` because the gpu-research
# account has no cpu-killable association (`sacctmgr show assoc user=$USER`).
#
# Run it after the two scoring jobs, chained so it does not need a live session:
#
#   A=$(sbatch --parsable --job-name=vfyFull-rome --export=ALL,ROOT=$R,\
#       MODEL=$HF/lment-1b-control-2e-b131k,OUT=$V/scored/full,\
#       SETS=$R/ember_eval/nll_kl/sets/rome.json $R/ember_eval/nll_kl/slurm/score.slurm)
#   B=$(sbatch --parsable --job-name=vfyTwin-rome --export=ALL,ROOT=$R,\
#       MODEL=$HF/lment-1b-norome-2e-b131k,OUT=$V/scored/twin,\
#       SETS=$R/ember_eval/nll_kl/sets/rome.json $R/ember_eval/nll_kl/slurm/score.slurm)
#   sbatch --dependency=afterok:$A:$B --export=ALL,ROOT=$R,SCORED=$V/scored,OUT=$V,TOPIC=rome \
#       $R/ember_eval/nll_kl/slurm/verify_reproduction.slurm
#
# Do not pass --exclude to the scoring jobs on the command line: it REPLACES
# score.slurm's node list rather than adding to it.
set -euo pipefail

: "${ROOT:?set ROOT to the LMEnt worktree}"
: "${SCORED:?set SCORED to the dir holding full/ and twin/}"
: "${OUT:?set OUT}"
: "${TOPIC:=rome}"
PY=/home/morg/NLP_2526b/stahli/anaconda3/envs/lment/bin/python

export TMPDIR=/home/dcor/galbarak2/tmp

exec "$PY" -u "$ROOT/ember_eval/nll_kl/verify_reproduction.py" \
  --topic "$TOPIC" --scored "$SCORED" --out "$OUT"
