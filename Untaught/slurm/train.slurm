#!/bin/sh
#SBATCH --job-name=untaught
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --signal=USR1@120

# Resource flags (--partition/--account/--time/--gres/--mem/--cpus-per-task) are
# passed on the sbatch command line by the run_smoke_*.sh scripts, so one file
# serves both runs. See https://www.cs.tau.ac.il/system/slurm
#
# Expected environment (exported by sbatch --export):
#   UNTAUGHT_ROOT   this folder
#   UNTAUGHT_CONFIG training config JSON
#   UNTAUGHT_BLACKLIST  chunk-id .npy, or empty for a control run
#   CONDA_ENV       conda environment name
#   NPROC           GPUs per node

set -eu

echo "=========================================================="
echo " job        : ${SLURM_JOB_NAME:-?} (${SLURM_JOB_ID:-?})"
echo " node       : $(hostname)"
echo " started    : $(date)"
echo " config     : ${UNTAUGHT_CONFIG}"
echo " blacklist  : ${UNTAUGHT_BLACKLIST:-<none, control run>}"
echo "=========================================================="

if command -v conda >/dev/null 2>&1; then
  # conda's activation scripts reference unbound variables on many versions,
  # which `set -u` turns into a fatal error -- relax it just for activation.
  set +u
  # shellcheck disable=SC1091
  . "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
  set -u
  echo " conda env  : ${CONDA_ENV}"
fi

nvidia-smi || echo "warning: nvidia-smi unavailable"

cd "${UNTAUGHT_ROOT}"
export PYTHONPATH="${UNTAUGHT_ROOT}:${OLMO_CORE_SRC}:${PYTHONPATH:-}"

# Fail fast on a bad config or an unreadable blacklist, before touching the GPU.
echo "--- preflight ---"
python -m untaught.train_untaught "${UNTAUGHT_CONFIG}" \
  --blacklist "${UNTAUGHT_BLACKLIST:-}" --check

echo "--- training ---"
torchrun --standalone --nproc-per-node="${NPROC:-1}" \
  untaught/train_untaught.py "${UNTAUGHT_CONFIG}" \
  --blacklist "${UNTAUGHT_BLACKLIST:-}"

echo "=========================================================="
echo " finished   : $(date)"
echo "=========================================================="
