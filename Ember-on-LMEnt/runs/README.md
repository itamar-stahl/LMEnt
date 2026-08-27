# LMEnt EMBER runs

Every concept erasure creates one independent folder here:

```text
<concept>_<model-key>_<YYYYMMDD_HHMMSS>/
├── source_config.yaml
├── config.yaml
├── inputs/
├── run_wrapper.sh
├── run_wrapper.ps1
├── run_environment.json
└── outputs/
    ├── features/
    ├── erased_embeddings.safetensors
    └── report.json
```

Slurm runs also contain `job.slurm`, scheduler logs, and a client submission
report. The effective `config.yaml`, wrappers, and generated job use absolute
paths.

Run folders and their potentially large model artifacts are ignored by Git.
Only this README is tracked. Historical outputs from the previous
`lment_outputs` layout are preserved under `runs/legacy/`.
