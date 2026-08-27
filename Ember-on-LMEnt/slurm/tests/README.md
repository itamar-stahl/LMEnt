# Slurm test package

From the login node, run the complete suite with one command:

```sh
sh /home/morg/NLP_2526b/<USER>/LMEnt/Ember-on-LMEnt/slurm/tests/run_test.sh
```

Replace `<USER>` with your cluster username. The command first runs the complete
local test suite on the Linux login node. It submits nothing if those checks
fail. Only then does it submit one GPU job to `studentkillable` with
`#SBATCH --constraint="titan_xp"` and wait for that job.

The login node runs all unit tests, compilation, `pip check`, and shell syntax
checks. The GPU node runs only CUDA preflight, the real-checkpoint CUDA tests,
and a small real-model Pornography erasure using
`configs/ember_lment_slurm_test.yaml`. Every test run is retained under
`slurm_test_runs/test_<date_time>/` with its copied YAML, generated Slurm file,
wrapper, phase logs, `report.log`, and `test_result.json`.
