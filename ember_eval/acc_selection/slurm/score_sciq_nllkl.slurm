#!/bin/bash
#SBATCH --job-name=sciqnk
#SBATCH --account=gpu-research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32000
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1
#
# Correct-answer NLL and full-vocabulary distributions on the sciq set, so the
# unrelated group has KL(T||M) like target and neighbour do. One sciq set serves
# all three concepts, so each model is scored once.
set -uo pipefail
: "${ROOT:?}"; : "${JOBLIST:?}"; : "${OUT:?}"; : "${SETS:?}"
PY=/home/morg/NLP_2526b/stahli/anaconda3/envs/lment/bin/python
export HF_HOME=/home/dcor/galbarak2/hf_cache
export TOKENIZERS_PARALLELISM=false TMPDIR=/home/dcor/galbarak2/tmp
echo "host $(hostname)"; nvidia-smi --query-gpu=name --format=csv,noheader|head -1 || true
n=0; ok=0; bad=0
while IFS=$'\t' read -r topic label dir; do
  [ -z "${topic:-}" ] && continue
  n=$((n+1))
  if [ -f "$OUT/$label/$topic/dists.npy" ]; then echo "[$n] skip $label (has dists)"; continue; fi
  if ! "$PY" "$ROOT/ember_eval/nll_kl/check_alloc.py" "$dir"; then
    echo "[$n] SKIP $label: holed checkpoint" >&2; bad=$((bad+1)); continue; fi
  echo "[$n] === $label / sciq  $(date +%H:%M:%S)"
  if "$PY" -u "$ROOT/ember_eval/nll_kl/score_model.py" --model "$dir" \
        --sets "$SETS/sciq.json" --out "$OUT/$label"; then ok=$((ok+1))
  else echo "[$n] FAILED $label" >&2; bad=$((bad+1)); fi
done < "$JOBLIST"
echo "=== scored $ok, failed $bad, of $n"
exit $((bad > 0 ? 1 : 0))
