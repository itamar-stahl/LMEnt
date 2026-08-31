#!/bin/sh
# COMPUTE NODE entry point, sourced by each run folder's run_wrapper.sh:
#
#     . <root>/framework/node/set_node_env.sh && torchrun ...
#
# It cd's to Untaught/ and sets up the environment from scratch -- a batch job
# starts in a fresh shell that never reads ~/.bashrc, and its cwd is wherever
# SLURM put it, so neither can be assumed.
#
# No Elasticsearch here: it runs on the login node and is unreachable from the
# GPU nodes. Anything needing the index (turning a blacklist's QIDs into chunk
# ids) ran before submission and arrives as the artifact in the run folder.

# Same absolute path as on the login node -- shared filesystem, fixed location.
: "${LMENT_ROOT:=/home/morg/NLP_2526b/$(whoami)/LMEnt}"
cd "${LMENT_ROOT}/Untaught" || return 1 2>/dev/null || exit 1

# shellcheck disable=SC1091
. ./framework/env.sh
# shellcheck disable=SC1091
. ./framework/conda.sh

# Copy the tokenized dataset to node-local disk and repoint LMENT_DATASET at
# it. The shared filer has cost four runs since 2026-08-28 -- see the header of
# stage_dataset.sh. This must come after env.sh (which defines LMENT_DATASET)
# and before torchrun. It is fail-safe: any problem leaves LMENT_DATASET alone
# and the run reads from the share exactly as before.
# shellcheck disable=SC1091
. "${LMENT_ROOT}/Untaught/framework/node/stage_dataset.sh"

echo "[untaught] node $(hostname) in $(pwd): ${CONDA_ENV} @ $(command -v python)"
