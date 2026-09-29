#!/bin/bash
#SBATCH --job-name=ens-sciq
#SBATCH --output=/home/dcor/galbarak2/runs/ensembles/%x_%j.out
#SBATCH --error=/home/dcor/galbarak2/runs/ensembles/%x_%j.err
#SBATCH --account=gpu-research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=48000
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#
# NLL and full-vocabulary KL for one ensemble checkpoint on the SciQ held-out
# set. The topic sets' `unrelated_test` is the older other-concept pool; the
# unrelated group for these metrics must be SciQ, which lives in its own sets
# file. The references (full model, twins, standalone winners) were already
# scored on sciq in an earlier round -- only the ensembles were missing.
#
# One checkpoint per job: two concurrent 5 GB readers on one node collapse to
# ~638 kB/s, measured.
set -uo pipefail
: "${ROOT:?}"; : "${ENS:?}"; : "${LABEL:?}"; : "${MODELDIR:?}"
PY=/home/morg/NLP_2526b/stahli/anaconda3/envs/lment/bin/python
export HF_HOME=/home/dcor/galbarak2/hf_cache TOKENIZERS_PARALLELISM=false
export TMPDIR=/home/dcor/galbarak2/tmp
echo "sciq nll/kl for $LABEL on $(hostname) at $(date +%F_%H:%M:%S)"
"$PY" "$ROOT/ember_eval/nll_kl/check_alloc.py" "$MODELDIR" || exit 1
if [ -f "$ENS/scored_nllkl/$LABEL/sciq/records.json" ]; then
  echo "already scored"; exit 0
fi
"$PY" -u "$ROOT/ember_eval/nll_kl/score_model.py" \
  --model "$MODELDIR" --sets "$ROOT/ember_eval/nll_kl/sets/sciq.json" \
  --out "$ENS/scored_nllkl/$LABEL"
rc=$?
echo "done at $(date +%F_%H:%M:%S), rc=$rc"
exit $rc
