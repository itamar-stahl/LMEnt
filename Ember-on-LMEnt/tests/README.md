# Tests

Run fast tests from the activated `lment` environment:

```sh
python -m unittest discover -s tests -p 'test_*.py' -v
```

Run the real CPU and CUDA flows on Windows with one PowerShell command:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\tests\run_real_flows.ps1
```

These are bounded end-to-end tests, not mocks: each loads the actual LMEnt 1B
checkpoint, fits rank-2 features for two iterations, edits embeddings, evaluates
the control and erased states, and retains all outputs under `runs/`.

On the cluster, use `slurm/tests/run_test.sh`. It runs the fast suite and real
CPU flow on the login node, then submits the equivalent real CUDA flow to a
Titan XP. See `slurm/tests/README.md`.
