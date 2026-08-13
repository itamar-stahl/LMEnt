#!/bin/sh
# Shared environment for the Untaught experiments on the TAU SLURM cluster.
# Source this, do not execute it:   . configs/env.sh
#
# Everything here is an override-able default: `export FOO=...` before sourcing
# and that value wins.

# --- where things live on the remote -----------------------------------------
# Matches the layout under /home/morg/NLP_2526b/stahli
: "${STAHLI_ROOT:=/home/morg/NLP_2526b/stahli}"
: "${LMENT_ROOT:=${STAHLI_ROOT}/LMEnt}"
: "${LMENT_DATASET:=${STAHLI_ROOT}/LMEnt-Dataset}"
: "${LMENT_INDEX:=${STAHLI_ROOT}/index}"

# OLMo-core sources, needed to import examples.kas.train
: "${OLMO_CORE_SRC:=${LMENT_ROOT}/OLMo-core/src}"

# This folder (Untaught/). The run scripts pin it before sourcing; the $0
# fallback only serves a direct `. configs/env.sh` from inside Untaught/.
: "${UNTAUGHT_ROOT:=$(cd "$(dirname "$0")/.." 2>/dev/null && pwd)}"
if [ -z "${UNTAUGHT_ROOT}" ] || [ ! -d "${UNTAUGHT_ROOT}/untaught" ]; then
  UNTAUGHT_ROOT="${LMENT_ROOT}/Untaught"
fi

: "${UNTAUGHT_BLACKLIST_DIR:=${UNTAUGHT_ROOT}/blacklists}"
: "${UNTAUGHT_RUNS_DIR:=${UNTAUGHT_ROOT}/runs}"

# --- Elasticsearch ------------------------------------------------------------
# Index names follow the restore step in the top-level README:
#   enwiki_case_sensitive -> lment_cs      enwiki -> lment_ci
# Entity (QID) retrieval is case-agnostic, so lment_cs is the right default.
: "${ES_SCHEME:=https}"
: "${ES_HOST:=localhost}"
: "${ES_PORT:=9200}"
: "${ES_INDEX:=lment_cs}"
# Never hard-code the password here. Export ES_PASSWORD in your shell or keep it
# in a file that is not tracked by git.
: "${ES_PASSWORD:=}"

# --- SLURM (see https://www.cs.tau.ac.il/system/slurm) ------------------------
# Deliberately NOT named SLURM_* -- SLURM sets/reads that namespace itself, and
# launching from inside an `srun --pty bash` session would inherit stale values.
#
# Check which accounts/partitions you actually have with:
#   sacctmgr -P -i show user -s "$USER"
# Members of a research group may also have gpu-research / gpu-h100-killable
# (the LMEnt authors' own run.slurm files use those).
# Empty by default: the course grants partition access, not an account, and a
# bogus --account is one of the things that makes sbatch reject the features as
# unsatisfiable. The run scripts omit the flag when this is empty.
: "${UNTAUGHT_ACCOUNT:=}"
# studentkillable: 1 day, low priority, preemptible -- what the course grants.
# studentbatch (3 days, 6 jobs) and studentrun (3 h, interactive) need access.
: "${UNTAUGHT_PARTITION:=studentkillable}"
: "${UNTAUGHT_TIME:=180}"          # minutes
: "${UNTAUGHT_GPUS:=1}"
: "${UNTAUGHT_CPUS:=8}"
: "${UNTAUGHT_MEM:=64000}"         # MB
# The model trains with bf16 parameters (FSDP param_dtype=bfloat16 in
# build_config). Pre-Ampere cards (tesla_v100, geforce_rtx_2080, titan_xp,
# quadro_rtx_8000) have no bf16 support and will crash or crawl -- keep this
# constraint to Ampere-or-newer unless you know your partition's hardware.
: "${UNTAUGHT_CONSTRAINT:=geforce_rtx_3090|a5000|a6000|a100|l40s}"

# --- python -------------------------------------------------------------------
: "${CONDA_ENV:=lment}"

# W&B off by default; a cluster node without WANDB_API_KEY otherwise stalls.
: "${WANDB_MODE:=disabled}"

export STAHLI_ROOT LMENT_ROOT LMENT_DATASET LMENT_INDEX OLMO_CORE_SRC
export UNTAUGHT_ROOT UNTAUGHT_BLACKLIST_DIR UNTAUGHT_RUNS_DIR
export ES_SCHEME ES_HOST ES_PORT ES_INDEX ES_PASSWORD
export UNTAUGHT_ACCOUNT UNTAUGHT_PARTITION UNTAUGHT_TIME UNTAUGHT_GPUS
export UNTAUGHT_CPUS UNTAUGHT_MEM UNTAUGHT_CONSTRAINT
export CONDA_ENV WANDB_MODE

# Keep only the constraint names that actually exist on nodes in this partition.
# The cluster-wide feature list is not per-partition, and sbatch rejects the
# whole job ("Invalid feature specification") if one name is unknown there.
# Prints the surviving `a|b|c`, or nothing if none are -- caller drops the flag.
# sbatch, printed before it runs, retried once without --constraint if the
# feature expression is what SLURM objects to. Args: sbatch flags, then the
# script. Reads CONSTRAINT; unset/empty means no --constraint at all.
untaught_submit() {
  if [ -n "${CONSTRAINT:-}" ]; then
    echo "+ sbatch --constraint=${CONSTRAINT} $*"
    # shellcheck disable=SC2086
    if sbatch --constraint="${CONSTRAINT}" "$@"; then
      return 0
    fi
    echo "sbatch rejected --constraint=${CONSTRAINT}; retrying without it." >&2
    echo "Check the job log's nvidia-smi: bf16 needs Ampere or newer." >&2
  fi
  echo "+ sbatch $*"
  sbatch "$@"
}

untaught_constraint() {
  _uc_want="${1:-${UNTAUGHT_CONSTRAINT}}"
  [ -n "${_uc_want}" ] || return 0
  command -v sinfo >/dev/null 2>&1 || { printf '%s' "${_uc_want}"; return 0; }

  _uc_have="$(sinfo -h -p "${UNTAUGHT_PARTITION}" -o '%f' 2>/dev/null \
              | tr ',' '\n' | tr -d ' ' | sort -u)"
  [ -n "${_uc_have}" ] || return 0

  _uc_keep=""
  for _uc_n in $(printf '%s' "${_uc_want}" | tr '|' ' '); do
    if printf '%s\n' "${_uc_have}" | grep -qx "${_uc_n}"; then
      _uc_keep="${_uc_keep:+${_uc_keep}|}${_uc_n}"
    fi
  done
  printf '%s' "${_uc_keep}"
}

mkdir -p "${UNTAUGHT_BLACKLIST_DIR}" "${UNTAUGHT_RUNS_DIR}"
