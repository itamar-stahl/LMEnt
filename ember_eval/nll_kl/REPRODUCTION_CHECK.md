# Reproduction check — run record

What `verify_reproduction.py` is for, and the one run of it that exists.

Every other check in this tree compares committed artifacts against each other.
`validate_accwinners.py` runs twelve such checks, `audit.py` and
`ensembles/audit_results.py` more. None of them loads a model, so they can all
pass while the code that produced the numbers has drifted away from them. That
is not hypothetical here: this project has already shipped one table built from
a superseded checkpoint selection, and one metric that was edited mid-run and
faked an 18-point swing.

This check closes that loop — code to numbers, not numbers to numbers.

## The run

**2026-09-29**, Ancient Rome, on the cluster.

| Job | Name | Node | State | Elapsed |
|---|---|---|---|---|
| 951003 | `vfyFull-rome` | n-803 | COMPLETED (0:0) | 00:21:47 |
| 951004 | `vfyTwin-rome` | n-803 | COMPLETED (0:0) | 00:21:47 |
| 951006 | `vfyCheck-rome` | n-350 | COMPLETED (0:0) | 00:00:37 |

Both models were scored from scratch against the frozen `sets/rome.json` using
this repository's own `slurm/score.slurm` — `lment-1b-control-2e-b131k` as the
full model and `lment-1b-norome-2e-b131k` as the twin. Each wrote
`records.json`, `dists.npy` (208 MB) and `dists_index.json`. The comparison was
then recomputed with `compare.py` and asserted against
`results_accwinners/rome/results.json`, which is what the paper's Panel B and C
are printed from.

## Result

**18/18 reproduced**, at a tolerance of 5e-4. Six NLL sets and three KL sets,
mean and standard deviation for each:

```
  PASS  NLL twin-full neighbour_selection mean got 0.1519  want 0.1519
  PASS  NLL twin-full neighbour_selection sd   got 0.7659  want 0.7659
  PASS  NLL twin-full neighbour_test mean      got 0.1712  want 0.1712
  PASS  NLL twin-full neighbour_test sd        got 0.6702  want 0.6702
  PASS  NLL twin-full target_selection mean    got 1.2296  want 1.2296
  PASS  NLL twin-full target_selection sd      got 1.5260  want 1.5260
  PASS  NLL twin-full target_test mean         got 1.0158  want 1.0158
  PASS  NLL twin-full target_test sd           got 1.0015  want 1.0015
  PASS  NLL twin-full unrelated_selection mean got -0.0822 want -0.0822
  PASS  NLL twin-full unrelated_selection sd   got 0.6666  want 0.6666
  PASS  NLL twin-full unrelated_test mean      got 0.0306  want 0.0306
  PASS  NLL twin-full unrelated_test sd        got 0.5090  want 0.5090
  PASS  KL(twin||full) neighbour_test mean     got 0.2531  want 0.2531
  PASS  KL(twin||full) neighbour_test sd       got 0.1619  want 0.1619
  PASS  KL(twin||full) target_test mean        got 0.9551  want 0.9551
  PASS  KL(twin||full) target_test sd          got 0.7248  want 0.7248
  PASS  KL(twin||full) unrelated_test mean     got 0.3429  want 0.3429
  PASS  KL(twin||full) unrelated_test sd       got 0.2106  want 0.2106

18/18 reproduced
```

`target_test` is the pair the paper prints in Panel C for Ancient Rome:
`+1.016 ± 1.001` for the twin-minus-full answer NLL, and `0.955 ± 0.725` for
`KL(twin ‖ full)`.

## What this does and does not establish

It establishes that the path the paper depends on still runs end to end and
still produces the published numbers: model load, frozen question set,
tokenisation, teacher-forced per-token NLL, full-vocabulary distributions, and
the comparison code.

It does not re-derive the erasure grid, checkpoint selection, or the 97
candidates — those need the edited checkpoints and far more compute. It covers
one concept; `--topic baseball` and `--topic ai` are accepted and would need
their own scoring runs. And it ran on this cluster with these paths, so it says
nothing about portability.

Expected values are read from `results_accwinners/<topic>/results.json` rather
than written into the script, so the check cannot drift from the published
numbers: if those are regenerated, the target moves with them.
