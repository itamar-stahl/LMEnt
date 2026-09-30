#!/bin/bash
#SBATCH --job-name=newwin
#SBATCH --account=gpu-research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#
# Score the accuracy-selected winners on the FULL nll_kl sets, with next-token
# distributions, so each can be compared against the control and the twin on the
# held-out test half.
#
# These six were previously scored with --no-dists --only-phase selection, as
# grid candidates. That record is preserved as <topic>.selection-only rather than
# overwritten, so the selection that chose them stays auditable.
#
# Sequential: each checkpoint is a separate 5 GB read and the job is pure I/O.
set -uo pipefail
: "${ROOT:?}"; : "${JOBLIST:?}"; : "${OUT:?}"; : "${SETS:?}"
PY=/home/morg/NLP_2526b/stahli/anaconda3/envs/lment/bin/python
export HF_HOME=/home/dcor/galbarak2/hf_cache
export TOKENIZERS_PARALLELISM=false TMPDIR=/home/dcor/galbarak2/tmp
echo "host $(hostname)"
nvidia-smi --query-gpu=name --format=csv,noheader | head -1 || true
n=0; ok=0; bad=0
while IFS=$'\t' read -r topic label dir; do
  [ -z "${topic:-}" ] && continue
  n=$((n+1))
  if [ -f "$OUT/$label/$topic/dists.npy" ]; then echo "[$n] skip $label/$topic (has dists)"; continue; fi
  if ! "$PY" "$ROOT/ember_eval/nll_kl/check_alloc.py" "$dir"; then
    echo "[$n] SKIP $label: holed checkpoint" >&2; bad=$((bad+1)); continue; fi
  # keep the selection-only record instead of clobbering it
  if [ -d "$OUT/$label/$topic" ] && [ ! -d "$OUT/$label/$topic.selection-only" ]; then
    mv "$OUT/$label/$topic" "$OUT/$label/$topic.selection-only"
    echo "[$n] preserved previous selection-only record for $label/$topic"
  fi
  echo "[$n] === $label / $topic  $(date +%H:%M:%S)"
  if "$PY" -u "$ROOT/ember_eval/nll_kl/score_model.py" --model "$dir" \
        --sets "$SETS/$topic.json" --out "$OUT/$label"; then ok=$((ok+1))
  else echo "[$n] FAILED $label/$topic" >&2; bad=$((bad+1)); fi
done < "$JOBLIST"
echo "=== scored $ok, failed $bad, of $n"
exit $((bad > 0 ? 1 : 0))
