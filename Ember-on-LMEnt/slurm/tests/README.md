# Slurm test package

From the login node, run the complete suite with one command:

```sh
sh /home/morg/NLP_2526b/<USER>/LMEnt/Ember-on-LMEnt/slurm/tests/run_test.sh
```

Replace `<USER>` with your cluster username. The command submits one job to
`studentkillable` with `#SBATCH --constraint="titan_xp"`, waits for it, prints a
phase-by-phase result, and exits nonzero on failure.

The node runs all local unit tests, compilation, `pip check`, and shell syntax
checks. It then performs a small real-model Pornography erasure with the
settings in `configs/ember_lment_slurm_test.yaml`. Every test job is retained
under `slurm_test_runs/test_<date_time>/` with its copied YAML, generated Slurm
file, wrapper, phase logs, `report.log`, and `test_result.json`.
