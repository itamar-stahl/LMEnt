#! /bin/sh
#SBATCH --job-name=postember-snmf
#SBATCH --output=/home/dcor/galbarak2/runs/mlp_erasure/%x_%j.out
#SBATCH --error=/home/dcor/galbarak2/runs/mlp_erasure/%x_%j.err
#SBATCH --account=gpu-research
#SBATCH --partition=killable
#SBATCH --exclude=rack-bgw-dgx1,rack-gww-dgx1,rack-omerl-g01,n-202,n-203,n-204,n-205,n-602
#SBATCH --time=90
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32000
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#
# VARIANT 1: SNMF on top of an EMBER-erased model, using the SAME features
# derived from the control.
#
# Why this composition is clean: EMBER's configs run with save.full_model:
# false and edit only the embedding matrix, so an EMBER-erased checkpoint has
# BIT-IDENTICAL MLP weights to the control. The SNMF directions derived on the
# control therefore address exactly the same MLP weights here; the only thing
# that changed upstream is what the embeddings feed into them. The job asserts
# that MLP-identity / embedding-difference up front rather than assuming it.
#
# The question: EMBER already took Rome 44 -> 38 and Baseball 48 -> 36 on the
# held-out test split. Is there anything left in the MLPs to remove on top of
# that, or was the reachable knowledge all in the embeddings?
#
# Two deltas, both with 3 neuron-permutation controls:
#   delta  1.0   exact removal -- the real erasure setting
#   delta -19.0  x20 sign-preserving -- the ONLY regime that has ever produced
#                a real-vs-permuted separation (Baseball, job 893467)
#
# Test split, because that is where EMBER's own -6/-12pt numbers live.

# ROOT: the checkout whose code this job runs. Walk up from the directory you
# submitted from, so a second checkout runs its own code rather than whichever
# tree was hard-coded here; export ROOT=<checkout> to pin one deliberately.
if [ -z "${ROOT:-}" ]; then
  ROOT=${SLURM_SUBMIT_DIR:-$PWD}
  while [ "$ROOT" != "/" ] && [ ! -f "$ROOT/mlp_erasure/env.sh" ]; do
    ROOT=$(dirname "$ROOT")
  done
fi
[ -f "$ROOT/mlp_erasure/env.sh" ] || { echo "not an LMEnt checkout: ${SLURM_SUBMIT_DIR:-$PWD}; export ROOT=<checkout>" >&2; exit 1; }
. "$ROOT/mlp_erasure/env.sh"
HF=/home/dcor/galbarak2/hf-models

CTRL=$HF/lment-1b-control-2e-b131k

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 || echo "no nvidia-smi"

echo "=== PREFLIGHT: confirm EMBER touched embeddings only, MLPs identical ==="
"$PY" - <<PYEOF
import hashlib, torch
from safetensors import safe_open
from pathlib import Path

def digest(model_dir, names):
    out = {}
    d = Path(model_dir)
    shards = sorted(d.glob("model-*.safetensors"))
    for s in shards:
        with safe_open(str(s), framework="pt") as f:
            for k in f.keys():
                if k in names:
                    t = f.get_tensor(k)
                    out[k] = hashlib.sha256(t.float().numpy().tobytes()).hexdigest()[:16]
    return out

probe = {"model.embed_tokens.weight",
         "model.layers.5.mlp.down_proj.weight",
         "model.layers.5.mlp.up_proj.weight"}
ctrl = digest("$CTRL", probe)
for tag, path in [("rome-erased", "$HF/lment-1b-rome-erased-b131k"),
                  ("baseball-erased", "$HF/lment-1b-baseball-erased-b131k")]:
    er = digest(path, probe)
    print(f"  {tag}:")
    for k in sorted(probe):
        same = ctrl.get(k) == er.get(k)
        print(f"    {'SAME ' if same else 'DIFF '} {k}")
PYEOF

# tag | model | snmf_out | concept
printf '%s\n' \
  "rome_judge_on_erased|$HF/lment-1b-rome-erased-b131k|$RUNS/snmf_romeband_871493|Ancient Rome" \
  "rome_ratio44_on_erased|$HF/lment-1b-rome-erased-b131k|$RUNS/snmf_rome_883188|Ancient Rome" \
  "baseball58_on_erased|$HF/lment-1b-baseball-erased-b131k|$RUNS/snmf_baseball_887805|Baseball" \
| while IFS='|' read -r tag model snmf_out concept; do
  echo
  echo "================ $tag ($concept, TEST split) ================"
  "$PY" "$ROOT/mlp_erasure/random_direction_control.py" \
    --model "$model" --snmf-out "$snmf_out" \
    --concept "$concept" \
    --questions "$DATA/mc_questions.json" \
    --subset QA_test --simdom-subset SimdomQA_test \
    --deltas 1.0 -19.0 \
    --seeds 1 2 3 \
    --gamma 0.95 --dtype fp32 \
    --out "$RUNS/postember_${tag}_${SLURM_JOB_ID}.csv"
  echo "$tag exit $?"
done

echo DONE
