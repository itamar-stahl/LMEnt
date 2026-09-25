# Ensemble run report

Generated 2026-09-25T04:06:31Z on t-806.

## Stages

```
     JobID            JobName          State    Elapsed ExitCode 
---------- ------------------ -------------- ---------- -------- 
    926508       ensprep-rome      COMPLETED   00:38:07      0:0 
```

## Artifacts on disk

| artifact | found | expected |
|---|---|---|
| candidate checkpoints | 43 | 43 |
| selection scorings | 46 | 46 |
| test scorings | 9 | 9 |
| NLL/KL scorings | 9 | 9 |
| `results_ensembles/selected_checkpoints.json` | present | present |
| `tables/panel_A_accuracy.csv` | present | present |
| `tables/question_level/summary.csv` | MISSING | present |
| `results_final/rome/results.json` | present | present |

## Frozen winners

```
rome      RMU+EMBER        rmuember_rome_L5mid_a100       H=0.7735263702
rome      SNMF+EMBER       snmfv2_rome_ratio_in           H=0.7474518686
rome      SNMF+EMBER-V1    snmfv1_rome_judge_both         H=0.7335490831
baseball  RMU+EMBER        rmuember_baseball_L5mid_a100   H=0.7904191617
baseball  SNMF+EMBER       snmfv2_baseball_ratio_both     H=0.7594936709
baseball  SNMF+EMBER-V1    snmfv1_baseball_ratio_out      H=0.6527196653
ai        RMU+EMBER        rmuember_ai_L5mid_a10          H=0.7943188189
ai        SNMF+EMBER       snmfv2_ai_ratio_in             H=0.9365244537
ai        SNMF+EMBER-V1    snmfv1_ai_ratio_out            H=0.8249181629
```

## Panel A

```
topic                    method         checkpoint                         acc_target  acc_neighbour  acc_unrelated  norm_target  norm_neigh
Ancient Rome             Twin           TWIN_lment-1b-norome-2e-b131k      0.260000    0.580000       0.400000       0.058824     0.846154  
Ancient Rome             EMBER          ember_rome_d200                    0.400000    0.560000       0.400000       0.882353     0.794872  
Ancient Rome             RMU            rmu_rome_L6hi_a10                  0.360000    0.500000       0.420000       0.647059     0.641026  
Ancient Rome             SNMF           snmf_rome_ratio_out                0.480000    0.620000       0.420000       1.000000     0.948718  
Ancient Rome             RMU+EMBER      rmuember_rome_L5mid_a100           0.420000    0.580000       0.520000       1.000000     0.846154  
Ancient Rome             SNMF+EMBER     snmfv2_rome_ratio_in               0.460000    0.600000       0.520000       1.000000     0.897436  
Ancient Rome             SNMF+EMBER-V1  snmfv1_rome_judge_both             0.400000    0.560000       0.520000       0.882353     0.794872  
Baseball                 Twin           TWIN_lment-1b-nobaseball-2e-b131k  0.500000    0.360000       0.420000       0.862069     0.578947  
Baseball                 EMBER          ember_baseball_d10                 0.400000    0.420000       0.440000       0.517241     0.894737  
Baseball                 RMU            rmu_baseball_L6hi_a10              0.500000    0.400000       0.440000       0.862069     0.789474  
Baseball                 SNMF           snmf_baseball_ratio_both           0.520000    0.440000       0.440000       0.931034     1.000000  
Baseball                 RMU+EMBER      rmuember_baseball_L5mid_a100       0.440000    0.360000       0.500000       0.655172     0.578947  
Baseball                 SNMF+EMBER     snmfv2_baseball_ratio_both         0.420000    0.400000       0.520000       0.586207     0.789474  
Baseball                 SNMF+EMBER-V1  snmfv1_baseball_ratio_out          0.400000    0.380000       0.520000       0.517241     0.684211  
Artificial intelligence  Twin           TWIN_lment-1b-noai-2e-b131k        0.480000    0.520000       0.400000       0.793103     1.000000  
Artificial intelligence  EMBER          ember_ai_d500                      0.360000    0.480000       0.420000       0.379310     0.920000  
Artificial intelligence  RMU            rmu_ai_L6hi_a10                    0.440000    0.440000       0.380000       0.655172     0.760000  
Artificial intelligence  SNMF           snmf_ai_ratio_in                   0.520000    0.500000       0.460000       0.931034     1.000000  
Artificial intelligence  RMU+EMBER      rmuember_ai_L5mid_a10              0.320000    0.460000       0.520000       0.241379     0.840000  
Artificial intelligence  SNMF+EMBER     snmfv2_ai_ratio_in                 0.340000    0.540000       0.520000       0.310345     1.000000  
Artificial intelligence  SNMF+EMBER-V1  snmfv1_ai_ratio_out                0.340000    0.500000       0.520000       0.310345     1.000000  
```

## RMU sanity gates

A cell that fails `rotation_is_substantial` did not rotate the forget
representations enough for its QA number to mean anything. Per
ENSEMBLE_GRID.md these are reported, not excluded -- the grid was frozen
before any of them was built. If a WINNER appears here, its result is not
evidence and must be labelled as such in the paper.

```
rmuember_ai_L5hi_a10               cos 0.052->0.488  PASS
rmuember_ai_L5hi_a100              cos 0.049->0.348  PASS
rmuember_ai_L5mid_a10              cos 0.038->0.285  FAIL unlearn_loss_fell,rotation_is_substantial  <-- SELECTED WINNER
rmuember_ai_L5mid_a100             cos 0.026->0.094  FAIL unlearn_loss_fell,rotation_is_substantial
rmuember_ai_L6hi_a10               cos 0.043->0.455  PASS
rmuember_ai_L6hi_a100              cos 0.041->0.326  PASS
rmuember_ai_L6mid_a10              cos 0.030->0.274  FAIL unlearn_loss_fell,rotation_is_substantial
rmuember_ai_L6mid_a100             cos 0.019->0.094  FAIL unlearn_loss_fell,rotation_is_substantial
rmuember_baseball_L5hi_a10         cos 0.060->0.471  PASS
rmuember_baseball_L5hi_a100        cos 0.056->0.282  FAIL rotation_is_substantial
rmuember_baseball_L5mid_a10        cos 0.045->0.237  FAIL rotation_is_substantial
rmuember_baseball_L5mid_a100       cos 0.031->0.076  FAIL forget_rotated,rotation_is_substantial  <-- SELECTED WINNER
rmuember_baseball_L6hi_a10         cos 0.051->0.433  PASS
rmuember_baseball_L6hi_a100        cos 0.048->0.264  FAIL rotation_is_substantial
rmuember_baseball_L6mid_a10        cos 0.038->0.227  FAIL rotation_is_substantial
rmuember_baseball_L6mid_a100       cos 0.025->0.068  FAIL forget_rotated,rotation_is_substantial
rmuember_rome_L5hi_a10             cos 0.065->0.475  PASS
rmuember_rome_L5hi_a100            cos 0.062->0.339  PASS
rmuember_rome_L5mid_a10            cos 0.050->0.290  FAIL rotation_is_substantial
rmuember_rome_L5mid_a100           cos 0.041->0.125  FAIL rotation_is_substantial  <-- SELECTED WINNER
rmuember_rome_L6hi_a10             cos 0.052->0.457  PASS
rmuember_rome_L6hi_a100            cos 0.050->0.324  PASS
rmuember_rome_L6mid_a10            cos 0.038->0.275  FAIL rotation_is_substantial
rmuember_rome_L6mid_a100           cos 0.030->0.112  FAIL rotation_is_substantial

14 of 24 RMU cells failed a gate.

WARNING: these SELECTED winners failed a sanity gate; their numbers are not evidence:
  rmuember_rome_L5mid_a100
  rmuember_baseball_L5mid_a100
  rmuember_ai_L5mid_a10
```

## No stage failed.
