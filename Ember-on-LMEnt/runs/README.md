# LMEnt EMBER runs

Every concept erasure creates one independent folder here:

```text
<concept>_<model-key>_<YYYYMMDD_HHMMSS>/
|-- source_config.yaml
|-- config.yaml
|-- inputs/
|-- run_wrapper.sh
|-- run_wrapper.ps1
|-- run_environment.json
`-- outputs/
    |-- features/
    |-- erased_embeddings.safetensors
    |-- report.json
    `-- real_flow_test_report.json  # present for real-flow tests
```

Slurm runs also contain `job.slurm`, scheduler logs, and a client submission
report. The effective `config.yaml`, wrappers, and generated job use absolute
paths.

Run folders and their potentially large model artifacts are ignored by Git.
Only this README is tracked. Historical outputs from the previous
`lment_outputs` layout are preserved under `runs/legacy/`.

The retained real-flow test report confirms the actual model and fitting
devices, source-checkpoint integrity, feature artifacts, edited tokens,
embedding artifact, and baseline/edited train/test evaluation.
