# Running this on the TAU cluster

Everything here was measured while training the 1B twins, mostly by losing time to
it first. The defects section is the part worth acting on.

## Defects in this framework

**`latest_checkpoint_dir` selects by step number without checking completeness.**
This is the dangerous one. A 1B checkpoint is 16 shards named `__N_M.distcp`.
Three separate incidents left an incomplete *newest* checkpoint, and resume picked
it every time:

- a NetApp `EIO` mid-save left 14/16 shards in one twin and 1/16 in the other;
- an interrupted copy left a partial directory;
- a preemption mid-save left 16 files still under their `tmp*` names, never
  renamed to `__N_M.distcp`.

Each was caught by hand. The third one had a job "loading" for 35 minutes before
anyone looked. A `len(shards) == expected` guard in `latest_checkpoint_dir` would
have prevented all three. Until then, before trusting a resume:

    ls <ckpt>/model_and_optim | grep -c '^__'    # expect 16, and no tmp* files

**`--signal=USR1@120` has no handler.** `prepare.py` emits it; OLMo-core installs
handlers for SIGTERM and SIGINT only (`trainer.py`, `_handle_os_signal`). An
unhandled SIGUSR1 terminates the process, so the directive kills the job two
minutes *before* its limit rather than giving it a graceful save.
`--signal=TERM@120` would route into the handler that already exists.

**`adapt_to_gpu` misjudges capability at both ends.** It disables `torch.compile`
only below capability 7.0, so an RTX 2080 Ti (7.5) keeps compile enabled, OOMs in
`backward()`, and inductor logs that it skipped bfloat16 compilation anyway --
memory spent for no benefit. Itamar's TITAN Xp runs are unaffected only because
6.1 falls below the threshold. At the other end it reports
`"triton_compilable": true` for a B200, which says nothing about whether kernels
exist for `sm_100` (see `B200.md`).

**`adapt_to_gpu` only shrinks the microbatch, never raises it.** A config sized for
a 12 GiB TITAN Xp runs at that microbatch on a 140 GiB H200.

**Requeue truncates the job's logs.** Checkpoints survive preemption; `log.out` and
`log.err` are reopened empty, so the loss curve for every window except the last is
lost. `#SBATCH --open-mode=append` fixes it.

## Cluster facts that are not obvious

**Free GPUs are not always schedulable.** `gpu-h200` runs under QOS `gpu4`, capped
at **4 GPUs cluster-wide** however many the node has. A job can pend on
`QOSGrpGRES` with cards visibly idle. `gpu-h200-killable`, `gpu-b200*`,
`gpu-h100-killable` and `killable` all run under `general`, which has no practical
cap. Check the partition's QoS before concluding it is free.

**Nor is a free GPU enough.** Memory and CPUs gate independently. One job holding
384 GB and all 192 CPUs leaves four H200s idle and unusable by anyone.

**Preemption follows partition priority tier**, not job priority. `gpu-h200` is
tier 20 and `gpu-h200-killable` is tier 10, so any non-killable submission evicts
killable work. `PreemptMode=REQUEUE` and `GraceTime=0`: preemption requeues the
job automatically and it resumes from its checkpoint, but **reaching the time
limit does not requeue** -- that needs a manual resubmission.

**`gpu-h100-killable` is starved.** Its 16 cards are held by `gpu-h100`, a
partition outside the `gpu-research` association, so jobs there queue behind work
you cannot see. It has shown 6 pending / 0 running for hours.

## Measured numbers

Model is `olmo2_1B`, 131,072-token global batch, 16,384 rank microbatch, one epoch
= 27,416 steps.

| | s/step | epoch | GPU-hours |
|---|---|---|---|
| H200 x1 | 2.30 | ~17.5 h | 17.5 |
| H200 x4 | 2.54 | ~19.3 h | 77.4 |
| B200 x4 | 1.39 | ~10.6 h | 42.4 |

**One GPU beats four.** FSDP shards every layer, so each rank all-gathers
parameters on every microstep, and splitting the batch across 4 ranks leaves only
2 gradient-accumulation microsteps to amortise that traffic. Four cards are slower
in wall-clock *and* 4.4x more expensive. `gpus: 1` in the 1B configs is a
throughput decision.

For reference, the same 170M config runs at 8.29 s/step on a TITAN Xp against 1.56
on an H200 -- 5.3x.

## Storage and memory

**`/home/morg` reads 4.2x faster than `/vol/scratch`** -- 85.6 MB/s against 20.3,
measured with `dd` on a 964 MB shard. A 15 GB checkpoint load is ~3 minutes versus
~12. Scratch has more headroom; morg is far faster, which matters because every
preemption pays the load cost again.

**`--mem` is not enforced here.** No cgroup cap is applied, so a job exceeds its
request without being killed -- a 24000 request ran at 27 GiB. The number only
affects scheduling. Two consequences: over-requesting strands idle GPUs behind a
memory wall, and a request below true usage "works" only because nothing is
checking.

**Nothing records memory.** `sacct` MaxRSS is blank on every job and `sstat`
returns nothing. To measure a live job:

    srun --jobid=<id> --overlap --ntasks=1 \
      bash -c 'for p in $(pgrep -u $USER python); do grep VmHWM /proc/$p/status; done'

The 1B trainer peaks at ~27 GiB during a checkpoint load, with four dataloader
workers at ~0.3 GiB each. The configs asked for 128000; 40000 is the measured
figure with headroom.
