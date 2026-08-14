#!/bin/sh
# LOGIN NODE entry point. Source it, do not execute it:
#
#     . /home/morg/NLP_2526b/stahli/LMEnt/Untaught/activate_env.sh
#
# It cd's to Untaught/ and leaves this shell there, with the lment conda env
# active, every UNTAUGHT_/LMENT_/ES_ variable set, and Elasticsearch running.
# That is everything the run_smoke_*.sh scripts, client_node.* and the tests
# need -- no ~/.bashrc, no manual `conda activate`, same for any user.
#
# The compute-node counterpart is gpu_node/gpu_node.sh, which skips
# Elasticsearch because the GPU nodes cannot reach it.

# Fixed location on the shared filesystem: the same absolute path on the login
# node and on every compute node, so nothing has to be discovered at runtime.
: "${LMENT_ROOT:=/home/morg/NLP_2526b/stahli/LMEnt}"
cd "${LMENT_ROOT}/Untaught" || return 1 2>/dev/null || exit 1

# shellcheck disable=SC1091
. ./configs/env.sh
# shellcheck disable=SC1091
. ./configs/conda.sh

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

echo "[untaught] ready in $(pwd): ${CONDA_ENV} @ $(command -v python)"
