#!/bin/sh
# Keep resubmitting a training job until it reaches its configured duration.
#
#     nohup setsid sh framework/client/auto_resubmit.sh <config.yaml> [max_resubmits] &
#
# WHY: preemption requeues a job automatically, but *exhausting the time limit
# does not* -- SLURM simply ends it. A run longer than the partition's limit
# therefore stalls at every window boundary until a human resubmits. That is the
# only thing this script does: notice the job is gone, check it did not finish,
# and submit the same config again. job.resume_from_previous_run picks up from
# the newest checkpoint, so nothing is lost but the steps since the last save.
#
# It refuses to resume from an incomplete checkpoint. `latest_checkpoint_dir`
# selects by step number without checking completeness, and three separate
# incidents have left a truncated newest checkpoint that resume then chose --
# see OPERATIONS.md. A 1B checkpoint is 16 `__N_M.distcp` shards; anything else
# is moved aside as `<step>.incomplete` so the framework falls back to the last
# good one. Moved, never deleted.
#
# Logs to <runs>/auto_resubmit_<jobname>.log. Survives ssh disconnect and the
# agent that started it, provided it was launched detached.

set -u

CONFIG="${1:?usage: auto_resubmit.sh <config.yaml> [max_resubmits]}"
MAX="${2:-6}"
POLL="${AUTO_RESUBMIT_POLL:-300}"

cd "$(dirname "$0")/../.." || exit 1
UNTAUGHT_ROOT="$(pwd)"
JOB_NAME="$(grep -E '^\s+name:' "$CONFIG" | head -1 | sed 's/.*name:[[:space:]]*"\([^"]*\)".*/\1/')"
RUNS="${UNTAUGHT_RUNS_DIR:-${UNTAUGHT_ROOT}/runs}"
LOG="${RUNS}/auto_resubmit_${JOB_NAME}.log"

say() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*" >> "$LOG"; }

say "=== watching ${JOB_NAME} (config ${CONFIG}, max ${MAX} resubmits, poll ${POLL}s)"

# Is the run finished? Any of its run folders logging completion counts.
finished() {
  grep -lq "Training complete" "${RUNS}/${JOB_NAME}"_*/log.out 2>/dev/null
}

# Newest checkpoint across every run folder of this job -- what resume would pick.
newest_ckpt() {
  find "${RUNS}/${JOB_NAME}"_*/checkpoints -maxdepth 2 -type d -name 'step*' 2>/dev/null \
    | sed 's|.*/step||' | grep -E '^[0-9]+$' | sort -n | tail -1
}

quarantine_if_incomplete() {
  step="$(newest_ckpt)"
  [ -z "$step" ] && return 0
  for d in "${RUNS}/${JOB_NAME}"_*/checkpoints/*/"step${step}"; do
    [ -d "$d" ] || continue
    n=$(ls "$d/model_and_optim" 2>/dev/null | grep -c '^__')
    t=$(ls "$d/model_and_optim" 2>/dev/null | grep -c '^tmp')
    if [ "$n" -ne 16 ] || [ "$t" -ne 0 ]; then
      say "INCOMPLETE newest checkpoint step${step}: ${n}/16 shards, ${t} tmp files -- moving aside"
      mv "$d" "${d}.incomplete.$(date +%s)" && say "  moved; resume will fall back to the previous checkpoint"
    else
      say "newest checkpoint step${step} is complete (16 shards, no tmp)"
    fi
  done
}

count=0
while : ; do
  if finished; then
    say "TRAINING COMPLETE after ${count} resubmit(s). Nothing more to do."
    exit 0
  fi

  # Still queued or running? Then there is nothing to do this round.
  if squeue --me -h -n "${JOB_NAME}" 2>/dev/null | grep -q .; then
    sleep "$POLL"
    continue
  fi

  if [ "$count" -ge "$MAX" ]; then
    say "STOPPING: ${MAX} resubmits used and the run is still not complete. Look at it by hand."
    exit 1
  fi

  say "job is gone and training is unfinished (newest checkpoint step $(newest_ckpt))"
  quarantine_if_incomplete

  count=$((count + 1))
  say "resubmitting (${count}/${MAX}) ..."
  out="$(sh -c ". ./activate_env.sh && sh ./framework/client/sub_builder.sh '${CONFIG}'" 2>&1)"
  echo "$out" | sed 's/^/    /' >> "$LOG"
  if echo "$out" | grep -q "Submitted batch job"; then
    say "resubmitted: $(echo "$out" | grep -oE 'Submitted batch job [0-9]+')"
    sleep 60          # let SLURM register it before the next poll
  else
    say "SUBMIT FAILED -- see the output above. Retrying after ${POLL}s."
    sleep "$POLL"
  fi
done
