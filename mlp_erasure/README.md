# mlp_erasure — RMU and SNMF on the LMEnt 1B

Two MLP-level erasure methods, made to run on this project's models and then
measured against the data-ablation twins that `../Untaught` trains.

**The result is a null.** Neither method removes concept knowledge that any of
our evaluations can see, at any setting tried. Embedding-level erasure
(EMBER, in `../Ember-on-LMEnt`) does — which is the finding, not a failure of
the experiment.

| method | level | result |
|---|---|---|
| SNMF | MLP | two-sided causal null — 3 concepts, 2 checkpoints, 3 depth bands, and on held-out chunk loss |
| RMU | MLP | null at the full published schedule, both sanity gates green |
| EMBER | embedding | works — Rome −6, Baseball −12, AI −12, adjacent concepts held |

## Which document answers what

| file | question |
|---|---|
| `DELTA_FIX_RERUN_RESULTS.md` | The re-run on fixed code, and the null in full. **Start here for results.** |
| `MLP_ERASURE.md` | How the two methods were made to run here at all, and what was wrong with them |
| `FEATURE_QUALITY_FIXES.md` | Why SNMF's features were weak, and what changed |

## Layout

    rmu.py  snmf.py                  the two methods
    sweep_delta.py                   scores every delta x layer-range cell
    random_direction_control.py      neuron-permutation control for the above
    env.sh                           shared defaults for every driver
    run_*.slurm                      one driver per experiment
    tests/                           unit tests; only judge_gpu_smoke.py needs a GPU

## Running one

Every driver resolves `ROOT` by walking up from the directory you submitted
from, so a checkout runs its own code rather than whichever tree was hard-coded
when the script was written:

    sbatch --export=ALL,MODE=factorize,OUT_TAG=rome,LAYERS='4 9 14' \
        mlp_erasure/run_snmf.slurm

Anything in `env.sh` overrides the same way (`RUNS=`, `PY=`, `DATA=`), and
`export ROOT=<checkout>` pins a specific tree deliberately.
