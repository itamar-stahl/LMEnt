#!/bin/bash
#SBATCH --job-name=ens-tables
#SBATCH --output=/home/dcor/galbarak2/runs/ensembles/%x_%j.out
#SBATCH --error=/home/dcor/galbarak2/runs/ensembles/%x_%j.err
#SBATCH --account=gpu-research
#SBATCH --partition=killable
#SBATCH --exclude=n-102,n-202,n-203,n-204,n-205,n-602,rack-bgw-dgx1,rack-gww-dgx1,rack-omerl-g01
#SBATCH --time=45
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=16000
#SBATCH --cpus-per-task=2
#SBATCH --gpus=0
#
# Phase 5: every number the new section needs. No GPU -- this only reads what
# Phase 4 wrote.
#
#   Panel A  accuracy and H_test          finalize_ensembles.py
#   Panel B  NLL and KL(twin||model)      report.py, unmodified
#   question-level |dNLL| and P_closer    question_level_similarity.py --extra-methods
#
# The standalone winners, the twins and the full model were scored in earlier
# rounds and are NOT re-scored: they are linked into one tree so report.py sees a
# single root. Linking rather than copying means the bytes stay the ones the
# published tables were computed from.
set -uo pipefail
: "${ROOT:?}"; : "${ENS:?}"
PY=/home/morg/NLP_2526b/stahli/anaconda3/envs/lment/bin/python
ACC=/home/morg/NLP_2526b/galbarak2/runs/acc_selection
NK=/home/morg/NLP_2526b/galbarak2/runs/nll_kl
export TMPDIR=/home/dcor/galbarak2/tmp

echo "phase5 on $(hostname) at $(date +%F_%H:%M:%S)"

# --- one tree per metric, ensembles plus everything already scored -----------
mkdir -p "$ENS/merged_test" "$ENS/merged_nllkl"
for src in "$ACC/scored_test"/*; do ln -sfn "$src" "$ENS/merged_test/$(basename "$src")"; done
for src in "$ENS/scored_test"/*;   do [ -e "$src" ] && ln -sfn "$src" "$ENS/merged_test/$(basename "$src")"; done
for src in "$NK/scored"/*;         do ln -sfn "$src" "$ENS/merged_nllkl/$(basename "$src")"; done
for src in "$ENS/scored_nllkl"/*;  do [ -e "$src" ] && ln -sfn "$src" "$ENS/merged_nllkl/$(basename "$src")"; done
echo "merged_test: $(ls "$ENS/merged_test" | wc -l) labels; merged_nllkl: $(ls "$ENS/merged_nllkl" | wc -l)"

# --- the cell map: standalone winners (frozen 09-23) + ensembles (frozen today)
"$PY" - "$ENS" <<'PYEOF'
import json, sys
from pathlib import Path
ENS = Path(sys.argv[1])
std = json.load(open("/home/dcor/galbarak2/lment-qlsim/ember_eval/acc_selection/"
                     "results_sciq/selected_checkpoints.json"))["winners"]
ens = json.load(open(ENS / "results_ensembles/selected_checkpoints.json"))["winners"]
cells = {}
for w in std + ens:
    cells.setdefault(w["topic"], {})[w["method"]] = w["model_label"]
json.dump(cells, open(ENS / "cells.json", "w"), indent=1)
print("cells:", json.dumps(cells, indent=1))
PYEOF

rc=0
"$PY" "$ROOT/ember_eval/ensembles/finalize_ensembles.py" \
  --scored-test "$ENS/merged_test" --cells "$ENS/cells.json" \
  --out "$ENS/tables" || rc=1

TW_rome=lment-1b-norome-2e-b131k; TW_baseball=lment-1b-nobaseball-2e-b131k; TW_ai=lment-1b-noai-2e-b131k
for t in rome baseball ai; do
  eval "twin=\$TW_$t"
  mapfile -t ER < <("$PY" -c "
import json
c=json.load(open('$ENS/cells.json'))['$t']
for m,l in c.items(): print(f'{m}={l}')
")
  echo "=== report.py $t: ${ER[*]}"
  "$PY" "$ROOT/ember_eval/nll_kl/report.py" --topic "$t" --scored "$ENS/merged_nllkl" \
    --full lment-1b-control-2e-b131k --twin "$twin" \
    --erased "${ER[@]}" --out "$ENS/results_final/$t" || rc=1
done

"$PY" - "$ENS" <<'PYEOF'
import json, sys
from pathlib import Path
ENS = Path(sys.argv[1])
NAME = {"rome": "Ancient Rome", "baseball": "Baseball", "ai": "Artificial intelligence"}
cells = json.load(open(ENS / "cells.json"))
extra = {NAME[t]: {m: l for m, l in c.items() if m not in ("EMBER", "RMU", "SNMF")}
         for t, c in cells.items()}
json.dump(extra, open(ENS / "extra_methods.json", "w"), indent=1)
print("extra methods for the question-level pass:", json.dumps(extra))
PYEOF

"$PY" "$ROOT/ember_eval/nll_kl/question_level_similarity.py" \
  --results-root "$ENS/results_final" \
  --scored-root "$ENS/merged_nllkl" \
  --extra-methods "$ENS/extra_methods.json" \
  --out-dir "$ENS/tables/question_level" \
  --bootstrap-repetitions 10000 --seed 0 || rc=1

echo "phase5 done at $(date +%F_%H:%M:%S), rc=$rc"
exit $rc
