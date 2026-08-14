#!/bin/sh
# Put the lment conda environment in this shell. Shared by both entry points;
# expects framework/env.sh to have been sourced first.
#
# ~/.bashrc is not read by batch jobs, and belongs to one user anyway, so this
# repeats conda's setup instead of assuming it. The `conda shell.bash hook`
# branch of a stock .bashrc is skipped on purpose: this file is sourced by
# /bin/sh (that is what `sbatch --wrap` runs), and etc/profile.d/conda.sh is the
# POSIX entry point conda ships for that case.
#
# `conda activate` cannot be an executable -- it has to edit the *current*
# shell -- so it only exists once conda.sh has defined the `conda` function.

# conda's scripts read unbound variables ($PS1, $_CE_M, ...), fatal under
# `set -u`. Turn it off, and back on only if the caller had it.
case $- in *u*) untaught_had_u=1 ;; *) untaught_had_u=0 ;; esac
set +u

if [ -f "${ANACONDA_ROOT}/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  . "${ANACONDA_ROOT}/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV}"
else
  # Last resort, same as a stock .bashrc's: enough to run python, not enough
  # for `conda` subcommands.
  echo "[untaught] ${ANACONDA_ROOT}/etc/profile.d/conda.sh missing;" \
       "falling back to PATH" >&2
  PATH="${CONDA_ENVS_PATH}/${CONDA_ENV}/bin:${PATH}"
  export PATH
fi

[ "${untaught_had_u}" = 1 ] && set -u
unset untaught_had_u
