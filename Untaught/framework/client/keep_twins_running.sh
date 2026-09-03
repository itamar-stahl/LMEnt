#!/bin/sh
# Keep the two Untaught twins running, resubmitting each to whichever partition
# actually has room.
#
# WHY. Between 2026-08-28 and 2026-09-03 the pair died repeatedly to causes that
# have nothing to do with the experiment: an unreliable filer (Errno 5,
# DataLoader timeouts, NFS clients at 0.44 MB/s), a compute node with a full
# root filesystem (Errno 28 on /tmp/torchelastic), a node stuck
# DOWN+INVALID_REG, preemption, and scheduler reservations. Every one needed a
# human to notice and resubmit.
#
# WHAT IT DOES, once per cycle, per twin:
#   1. skip it if it is queued or running
#   2. find the newest checkpoint with all 16 distcp shards AND .metadata, and
#      quarantine any newer partial one so resume cannot pick it
#   3. choose a partition with genuinely free GPUs, memory and a healthy node
#   4. rewrite partition and max_time_minutes to match that partition's limit
#   5. submit
#
# CHECKPOINT VALIDATION IS THE POINT. The framework resumes from the newest
# checkpoint without checking it is complete, and a filer failure mid-write
# leaves one truncated -- it has produced 8-of-16 and 14-of-16 shard directories
# with no .metadata. Resuming from those would corrupt a run silently. Anything
# incomplete is renamed BROKEN_step<N>_<n>of16shards, never deleted.
#
# PARTITION CHOICE checks free GPUs, free memory AND node state, in that order
# of preference. gpu-h200 is tried first (no preemption, 5-day limit) but its
# QoS gpu4 carries GrpTRES=gres/gpu=4 across the WHOLE partition, so it is only
# usable when that group cap has room -- free GPUs on the node are not enough.
#
# Wall time must match the partition or sbatch refuses with "Requested time
# limit is invalid" and sub_builder reports an empty job id, which looks like
# success. gpu-h200 allows 5 days; both killable partitions allow 1.
#
# Usage:  sh keep_twins_running.sh [cycles] [seconds_between]
#         defaults: run for 3 hours, checking every 15 minutes.

ROOT="${LMENT_ROOT:-/home/morg/NLP_2526b/galbarak2/LMEnt-initfix}"
U="${ROOT}/Untaught"
CYCLES="${1:-12}"
GAP="${2:-900}"
NEED_MEM_MB=40000
NEED_GPU=1

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# free GPUs / free memory MB / state for one node
node_free() {
    _i=$(scontrol show node="$1" 2>/dev/null | tr ' ' '\n')
    [ -n "${_i}" ] || { echo "0 0 MISSING"; return; }
    _st=$(echo "${_i}" | grep -m1 '^State=' | cut -d= -f2)
    _gt=$(echo "${_i}" | grep -m1 '^CfgTRES=' | grep -oE 'gres/gpu=[0-9]+' | cut -d= -f2)
    _ga=$(echo "${_i}" | grep -m1 '^AllocTRES=' | grep -oE 'gres/gpu=[0-9]+' | cut -d= -f2)
    [ -n "${_ga}" ] || _ga=0
    _rm=$(echo "${_i}" | grep -m1 '^RealMemory=' | cut -d= -f2)
    _am=$(echo "${_i}" | grep -m1 '^AllocTRES=' | sed 's/.*mem=//;s/,.*//')
    case "${_am}" in *G) _am=$(( ${_am%G} * 1024 )) ;; *M) _am=${_am%M} ;; ""|AllocTRES=) _am=0 ;; esac
    echo "$(( _gt - _ga )) $(( _rm - _am )) ${_st}"
}

# Is the gpu-h200 QoS group cap under its limit? The cap is GrpTRES
# gres/gpu=N across the WHOLE partition, all users -- not per user, and not per
# job. Counting jobs would be wrong: one job can hold several cards (roiba's
# holds 2). squeue's %b and TresAlloc come back empty for these jobs, so read
# gres/gpu from scontrol per job and sum. Free GPUs on the node do NOT imply
# room here -- that mismatch is what left a submission unplaceable, reported as
# "reserved for jobs in higher priority partitions".
h200_qos_has_room() {
    _cap=$(sacctmgr -n show qos gpu4 format=GrpTRES%24 2>/dev/null | grep -oE 'gres/gpu=[0-9]+' | cut -d= -f2)
    [ -n "${_cap}" ] || return 1
    _used=0
    for _j in $(squeue -p gpu-h200 -t R -h -o "%i" 2>/dev/null); do
        _n=$(scontrol show job "${_j}" 2>/dev/null | tr ' ' '\n' | grep -m1 -oE 'gres/gpu=[0-9]+' | cut -d= -f2)
        [ -n "${_n}" ] && _used=$(( _used + _n ))
    done
    [ $(( _used + NEED_GPU )) -le "${_cap}" ]
}

# echo "partition wallminutes" for the best usable option, or nothing
pick_partition() {
    set -- $(node_free n-h200)
    _g=$1; _m=$2; _s=$3
    case "${_s}" in *DOWN*|*DRAIN*|MISSING) : ;; *)
        if [ "${_g}" -ge "${NEED_GPU}" ] && [ "${_m}" -ge "${NEED_MEM_MB}" ] && h200_qos_has_room; then
            echo "gpu-h200 2880"; return 0
        fi ;;
    esac
    for _n in t-100 n-102; do
        set -- $(node_free "${_n}")
        _g=$1; _m=$2; _s=$3
        case "${_s}" in *DOWN*|*DRAIN*|MISSING) continue ;; esac
        if [ "${_g}" -ge "${NEED_GPU}" ] && [ "${_m}" -ge "${NEED_MEM_MB}" ]; then
            echo "gpu-h100-killable 1440"; return 0
        fi
    done
    set -- $(node_free n-h200)
    _g=$1; _m=$2; _s=$3
    case "${_s}" in *DOWN*|*DRAIN*|MISSING) : ;; *)
        if [ "${_g}" -ge "${NEED_GPU}" ] && [ "${_m}" -ge "${NEED_MEM_MB}" ]; then
            echo "gpu-h200-killable 1440"; return 0
        fi ;;
    esac
    return 1
}

# quarantine incomplete checkpoints; echo the newest valid step dir
newest_valid_checkpoint() {
    _job="$1"; _best=""
    for _d in $(ls -dt "${U}/runs/${_job}"_2026* 2>/dev/null); do
        for _p in $(ls -d "${_d}"/checkpoints/*/step* 2>/dev/null | sort -t p -k3 -n -r); do
            case "$(basename "${_p}")" in BROKEN_*) continue ;; esac
            _n=$(ls "${_p}"/model_and_optim/*.distcp 2>/dev/null | wc -l)
            if [ "${_n}" = "16" ] && [ -f "${_p}/model_and_optim/.metadata" ]; then
                [ -z "${_best}" ] && _best="${_p}"
                break
            fi
            # newer than the newest valid one and incomplete: resume would take it
            log "  QUARANTINE $(basename "${_p}") (${_n}/16 shards, metadata $([ -f "${_p}/model_and_optim/.metadata" ] && echo ok || echo missing))"
            mv "${_p}" "$(dirname "${_p}")/BROKEN_$(basename "${_p}")_${_n}of16shards" 2>/dev/null
        done
        [ -n "${_best}" ] && break
    done
    echo "${_best}"
}

set_field() {  # file key value
    python3 - "$1" "$2" "$3" <<'PY'
import re, sys
path, key, val = sys.argv[1], sys.argv[2], sys.argv[3]
lines = open(path).read().split("\n")
out, i = [], 0
while i < len(lines):
    if re.match(rf"^\s*{re.escape(key)}:", lines[i]):
        j = i + 1
        while j < len(lines) and re.match(r"^\s+#", lines[j]):
            j += 1
        out.append(f"  {key}: {val}")
        i = j
        continue
    out.append(lines[i]); i += 1
open(path, "w").write("\n".join(out))
PY
}

cycle() {
    for _pair in "untaught-no-rome-1b-2e:train_1b_no_ancient_rome_2e" \
                 "untaught-control-1b-2e:train_1b_control_2e"; do
        _job=${_pair%%:*}; _cfg=${_pair##*:}
        if squeue -u "$(whoami)" -h -o "%j %t" 2>/dev/null | grep -q "^${_job} "; then
            log "${_job}: already queued or running"
            continue
        fi
        _ck=$(newest_valid_checkpoint "${_job}")
        [ -n "${_ck}" ] && log "${_job}: DOWN -- newest valid $(basename "${_ck}")" \
                        || log "${_job}: DOWN -- no valid checkpoint, will start fresh"
        _sel=$(pick_partition) || { log "${_job}: no partition has room, will retry"; continue; }
        set -- ${_sel}
        log "${_job}: submitting to $1 (wall $2 min)"
        set_field "${U}/configs/${_cfg}.yaml" partition "\"$1\""
        set_field "${U}/configs/${_cfg}.yaml" max_time_minutes "$2"
        ( cd "${U}" && env LMENT_ROOT="${ROOT}" sh framework/client/sub_builder.sh \
              "configs/${_cfg}.yaml" 2>&1 | grep -E "Submitted batch job|ERROR" ) | sed 's/^/    /'
    done
}

log "keep_twins_running: ${CYCLES} cycles, ${GAP}s apart"
_i=0
while [ "${_i}" -lt "${CYCLES}" ]; do
    _i=$(( _i + 1 ))
    log "--- cycle ${_i}/${CYCLES} ---"
    cycle
    [ "${_i}" -lt "${CYCLES}" ] && sleep "${GAP}"
done
log "done"
