#!/bin/bash
#SBATCH --job-name=accsel
#SBATCH --account=gpu-research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#SBATCH --exclude=n-102,n-202,n-203,n-204,n-205,rack-bgw-dgx1,rack-gww-dgx1,rack-omerl-g01
#
# Score checkpoints on the frozen SELECTION questions, strictly one at a time.
#
# Sequential on purpose, and the reason is the filer rather than the GPU: each
# candidate is a separate 5 GB checkpoint and the scoring itself takes about a
# second, so the job is pure I/O. Parallel readers on ONE node each get 1/N of
# that node's link -- eight jobs on n-306 loaded nothing in 30 minutes. Running
# everything in a single job makes contention impossible.
#
# JOBLIST is a file, one scoring per line, tab-separated:
#     <topic>\t<label>\t<model dir>
# A file rather than an environment variable so that 67 entries cannot be
# truncated, re-split, or mangled by quoting, and so the exact work list is
# recoverable afterwards.
#
# Any line whose per_question.csv already exists is skipped, so a preempted or
# timed-out job is resubmitted as-is and continues where it stopped.
#
#   sbatch --export=ALL,ROOT=<worktree>,JOBLIST=<file>,OUT=<dir>,MANIFEST=<csv> \
#          score_selection.slurm
set -uo pipefail

: "${ROOT:?set ROOT to the worktree}"
: "${JOBLIST:?set JOBLIST to the tab-separated work list}"
: "${OUT:?set OUT}"
: "${MANIFEST:?set MANIFEST}"
PY=/home/morg/NLP_2526b/stahli/anaconda3/envs/lment/bin/python

export HF_HOME=/home/dcor/galbarak2/hf_cache
export TOKENIZERS_PARALLELISM=false
export TMPDIR=/home/dcor/galbarak2/tmp

echo "host $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 || true
total=$(grep -vc '^[[:space:]]*$' "$JOBLIST")
echo "joblist: $total scorings from $JOBLIST"

i=0; done_=0; skipped=0; failed=0
while IFS=$'\t' read -r topic label dir; do
  [ -z "${topic:-}" ] && continue
  i=$((i+1))
  if [ -f "$OUT/$label/$topic/per_question.csv" ]; then
    echo "[$i/$total] skip $label/$topic (already scored)"; skipped=$((skipped+1)); continue
  fi
  if [ ! -r "$dir/config.json" ]; then
    echo "[$i/$total] FATAL $label: $dir unreadable from $(hostname)" >&2; failed=$((failed+1)); continue
  fi
  if ! "$PY" "$ROOT/ember_eval/nll_kl/check_alloc.py" "$dir"; then
    echo "[$i/$total] SKIP $label: partially-written checkpoint" >&2; failed=$((failed+1)); continue
  fi
  echo "[$i/$total] === $label / $topic  $(date +%H:%M:%S)"
  if "$PY" -u "$ROOT/ember_eval/acc_selection/score_mc.py" \
        --model "$dir" --manifest "$MANIFEST" --topic "$topic" --label "$label" --out "$OUT"; then
    done_=$((done_+1))
  else
    echo "[$i/$total] FAILED $label/$topic" >&2; failed=$((failed+1))
  fi
done < "$JOBLIST"

echo "=== scored $done_, skipped $skipped, failed $failed, of $total"
exit $((failed > 0 ? 1 : 0))
