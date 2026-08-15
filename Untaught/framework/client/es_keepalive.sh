#!/bin/sh
# Keep the group's Elasticsearch alive. Install it in the crontab of whoever
# owns the install -- for us, on c-003:
#
#   crontab -e
#   */5 * * * * /home/morg/NLP_2526b/stahli/LMEnt/Untaught/framework/client/es_keepalive.sh >> /home/morg/NLP_2526b/stahli/es_keepalive.log 2>&1
#   @reboot     /home/morg/NLP_2526b/stahli/LMEnt/Untaught/framework/client/es_keepalive.sh >> /home/morg/NLP_2526b/stahli/es_keepalive.log 2>&1
#
# Why this exists: Elasticsearch writes to its data, logs and config
# directories, so only the owner of the install can run it. The alternative --
# making the index writable by others -- would let any account on the cluster
# corrupt it, and would leave segment files owned by whoever happened to start
# the server, which breaks the next start. So instead of handing out write
# access, the owner schedules this and the server comes back on its own,
# usually within five minutes. Nobody has to be found and asked.
#
# Safe to run at any time and from any directory: it exits immediately if
# Elasticsearch is already answering, or if this user does not own the install.

set -u

cd "$(dirname "$0")/../.." || exit 1
# shellcheck disable=SC1091
. ./framework/env.sh

# Already serving? Any HTTP answer proves it -- an unauthenticated request
# gets 401, which still means the server is up.
if curl -s -k --max-time 5 -o /dev/null "${ES_SCHEME}://${ES_HOST}:${ES_PORT}"; then
  exit 0
fi

# Booting from a previous tick: a cold start reads the whole index and can take
# longer than the cron interval. Starting a second one would fail on node.lock
# anyway, but noisily.
if pgrep -u "$(whoami)" -f org.elasticsearch.bootstrap.Elasticsearch >/dev/null 2>&1; then
  echo "$(date '+%F %T') still starting up -- waiting"
  exit 0
fi

if [ ! -w "${ES_HOME}" ]; then
  echo "$(date '+%F %T') ${ES_HOME} is not yours to start; ask its owner" >&2
  exit 1
fi

echo "$(date '+%F %T') Elasticsearch is down -- starting it"
# Detached, so it outlives this cron job.
( cd "${ES_HOME}" && nohup ./bin/elasticsearch >"${UNTAUGHT_RUNS_DIR}/elasticsearch.log" 2>&1 & )
