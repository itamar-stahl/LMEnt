#!/bin/sh
# LOGIN NODE entry point. Source it, do not execute it:
#
#     cd .../Untaught && . ./activate_env.sh
#
# Afterwards this shell has the lment conda env active, every UNTAUGHT_/LMENT_/
# ES_ variable set, and a running Elasticsearch. That is everything the
# run_smoke_*.sh scripts and the tests need -- no ~/.bashrc, no manual
# `conda activate`, works for any user with read access to the paths.
#
# The compute-node counterpart is gpu_node.sh, which skips Elasticsearch
# because the GPU nodes cannot reach it.

# Locate this file. When sourced, $0 is the *shell*, not the script -- bash
# exposes the real path as $BASH_SOURCE, and the cwd is the last resort.
untaught_self="${BASH_SOURCE:-$0}"
UNTAUGHT_ROOT="$(cd "$(dirname "${untaught_self}")" 2>/dev/null && pwd)"
if [ ! -d "${UNTAUGHT_ROOT}/framework" ]; then
  UNTAUGHT_ROOT="$(pwd)"
fi
unset untaught_self
if [ ! -d "${UNTAUGHT_ROOT}/framework" ]; then
  echo "[untaught] cannot locate Untaught/ -- source this from that folder:" >&2
  echo "           cd <...>/Untaught && . ./activate_env.sh" >&2
  return 1 2>/dev/null || exit 1
fi
export UNTAUGHT_ROOT

# shellcheck disable=SC1091
. "${UNTAUGHT_ROOT}/configs/env.sh"
# shellcheck disable=SC1091
. "${UNTAUGHT_ROOT}/configs/conda.sh"

# --- Elasticsearch ------------------------------------------------------------
# It runs on this node, and the GPU nodes cannot reach it, so everything that
# touches the index happens here. Any HTTP response means it is serving (a bare
# request without credentials answers 401, which still proves it is up).
if curl -s -k --max-time 5 -o /dev/null "${ES_SCHEME}://${ES_HOST}:${ES_PORT}"; then
  echo "[untaught] Elasticsearch is up on ${ES_HOST}:${ES_PORT}"
elif [ ! -x "${ES_HOME}/bin/elasticsearch" ]; then
  echo "[untaught] Elasticsearch is down and ${ES_HOME}/bin/elasticsearch is missing" >&2
else
  echo "[untaught] Elasticsearch not responding -- starting it"
  # Detached, so it outlives this shell.
  ( cd "${ES_HOME}" && nohup ./bin/elasticsearch >"${UNTAUGHT_RUNS_DIR}/elasticsearch.log" 2>&1 & )

  untaught_waited=0
  while [ "${untaught_waited}" -lt 120 ]; do
    sleep 3
    untaught_waited=$((untaught_waited + 3))
    if curl -s -k --max-time 5 -o /dev/null "${ES_SCHEME}://${ES_HOST}:${ES_PORT}"; then
      break
    fi
  done

  if curl -s -k --max-time 5 -o /dev/null "${ES_SCHEME}://${ES_HOST}:${ES_PORT}"; then
    echo "[untaught] Elasticsearch up after ${untaught_waited}s"
  else
    echo "[untaught] Elasticsearch still down after ${untaught_waited}s -- see" \
         "${UNTAUGHT_RUNS_DIR}/elasticsearch.log" >&2
  fi
  unset untaught_waited
fi

echo "[untaught] ready: ${CONDA_ENV} @ $(command -v python), root ${UNTAUGHT_ROOT}"
