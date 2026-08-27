# Real LMEnt EMBER tests on Slurm

From the login node, run the complete suite with one command:

```sh
sh /home/morg/NLP_2526b/<USER>/LMEnt/Ember-on-LMEnt/slurm/tests/run_test.sh
```

Replace `<USER>` with your cluster username. The command runs these stages in
order:

1. All fast local tests in the Linux `lment` environment.
2. A real control-checkpoint CPU flow on the login node.
3. A real control-checkpoint CUDA flow in one `studentkillable` job with
   `#SBATCH --constraint="titan_xp"`.

No GPU job is submitted when a login-node stage fails.

Both real flows use the actual 1B control model, real Wikipedia sentences, rank
2, two SNMF iterations, fixed delta 0.5, and two real questions per evaluation
subset. They are deliberately short but continue through feature fitting,
feature selection, embedding editing, control/erased evaluation, and saving.

The CPU and GPU erasures are retained under `runs/`; each contains inputs,
features, an embedding-only artifact, `outputs/report.json`, and
`outputs/real_flow_test_report.json`. The orchestration package is retained
under `slurm_test_runs/test_<date_time>/` with both YAMLs, generated Slurm file,
wrapper, logs, the CPU run path, and `test_result.json` with the GPU run path.

Default configs:

- `configs/ember_lment_real_slurm_cpu.yaml`
- `configs/ember_lment_real_slurm_gpu.yaml`
