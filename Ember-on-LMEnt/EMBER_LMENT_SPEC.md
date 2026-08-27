# EMBER on LMEnt

## Interface

```bash
python -m ember.run_lment_ember --config /absolute/config.yaml --concept "Concept"
```

The YAML follows the original Ember structure. LMEnt-only data, device, cache,
judge, save, environment, and Slurm settings are nested under `lment`.

## One-concept flow

1. Create `runs/<concept>_<model-key>_<date_time>`.
2. Snapshot source/effective configuration and input JSON files.
3. Reuse a strictly matching three-branch feature cache, or fit new embedding
   features on the configured CPU/CUDA device.
4. Select features with the configured judge or required numeric threshold.
5. Evaluate the pristine model, choose the best configured delta unless fixed,
   apply EMBER, and evaluate the erased model.
6. Save the erased embedding by default, or the full model when configured.
7. Verify reload and weight isolation, save `report.json`, then publish a new
   feature bundle only if the shared cache is completely empty.

The source model is never modified.

## Slurm

The login client prepares the same run folder, resolves the hosted judge, and
submits its generated `job.slurm`. The job requests:

```text
#SBATCH --constraint=h100
```

The node runs feature fitting, judging, erasure, evaluation, saving, and cache
publication in the root `lment` environment. Generated files use resolved
absolute paths.

## Validation

The non-end-to-end suite covers configuration, CPU/CUDA fitting selection,
feature-cache provenance, feature selection, delta choice, embedding-only and
full saves, model reload, isolation, local wrappers, and Slurm materialization.
`slurm/tests/run_test.sh` adds the optional real H100 end-to-end run.
