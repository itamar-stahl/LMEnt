#!/bin/sh
# COMPUTE NODE entry point, sourced by the sbatch --wrap command:
#
#     sbatch ... --wrap="cd <root> && . ./gpu_node.sh && torchrun ..."
#
# A batch job starts in a fresh shell that never reads ~/.bashrc, so this sets
# up the environment from scratch: variables, then the lment conda env.
#
# No Elasticsearch here -- it runs on the login node and is unreachable from the
# GPU nodes. Anything needing the index (resolving a blacklist into chunk ids)
# happens before submission and arrives as the artifact in the run folder.

# Sourced from the --wrap command, where $0 is SLURM's generated script, so the
# cwd (set by the `cd` before this) is what identifies the checkout.
untaught_self="${BASH_SOURCE:-$0}"
UNTAUGHT_ROOT="$(cd "$(dirname "${untaught_self}")" 2>/dev/null && pwd)"
if [ ! -d "${UNTAUGHT_ROOT}/framework" ]; then
  UNTAUGHT_ROOT="$(pwd)"
fi
unset untaught_self
export UNTAUGHT_ROOT

# shellcheck disable=SC1091
. "${UNTAUGHT_ROOT}/configs/env.sh"
# shellcheck disable=SC1091
. "${UNTAUGHT_ROOT}/configs/conda.sh"

echo "[untaught] node $(hostname): ${CONDA_ENV} @ $(command -v python)"
