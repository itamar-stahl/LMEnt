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

**`optim_weight_decay` only reaches the embedding matrix.** The paper's own
training script builds the optimizer as

    AdamWConfig(
        lr=peak_lr,
        group_overrides=[OptimGroupOverride(params=["embeddings.weight"],
                                            opts=dict(weight_decay=weight_decay))]
    )  # Daniela, need to double check this.

-- that trailing comment is theirs, in `OLMo-core/src/examples/kas/train.py`. The
top-level `weight_decay` is never passed, so it falls to `AdamWConfig`'s default
of **0.01** for every parameter in the model, and the configured value applies to
`embeddings.weight` alone. Confirmed in the running 1B's own config dump:
`AdamWConfig(..., weight_decay=0.01, group_overrides=[... {'weight_decay': 0.05}])`.

Consequences: `optim_weight_decay: 0.05` in our configs decays the embeddings at
0.05 and everything else at 0.01, which is backwards from the usual convention of
sparing embeddings; appendix B.4's "weight decay of 0.05" describes a number that
in their code only ever touched the embedding matrix; and the run directory name
(`olmo2_1B_0.0004_131072_0.05_1`) records a value that is not the model's weight
decay. Nothing here is a comparability problem -- every LMEnt model, ours and the
authors' released ones, went through this same path -- but do not tune this knob
believing it is global.

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

**Extending a run's duration silently restarts it from random init.**
`max_duration_value` is not in `RESUME_IGNORED_FIELDS`, so a config that changes
`1` to `2` epochs no longer matches the finished run's identity,
`find_previous_checkpoint` rejects the folder, and training begins again from
scratch. This is the same trap as `job.constraint`, and it is worse here because
the intent -- "same experiment, keep going" -- is exactly the case the check is
meant to allow. Two ways round it, and neither is a config edit alone:

- symlink the finished `stepN` directory into the new run's `checkpoints/<upstream
  dir>/` before submitting, so upstream's own `maybe_load_checkpoint` finds it
  without consulting our identity check; note the upstream directory name encodes
  the duration (`olmo2_1B_0.0004_131072_0.05_1` -> `..._2`), so it is a different
  folder;
- or add `max_duration_value` to `RESUME_IGNORED_FIELDS["train"]`, which is the
  real fix but widens what counts as "the same experiment" for every run.

Either way, confirm the step counter resumes at the banked step rather than 0
before letting the job run out its window. `resuming from ...` in the log is
printed long before any state is read.

**A longer `max_duration` re-plans the cosine schedule, so a resume is a warm
restart.** `CosWithWarmup` computes its LR from `trainer.max_steps` at every step,
with `alpha_f=0.1`, and nothing pins the original horizon. The 1B pair finished
its epoch at lr `4.0e-5`, the floor of a 27,416-step cosine. Resume the same
checkpoint under a longer duration and the LR jumps straight back up:

| resumed with | lr at step 27,416 | jump |
|---|---|---|
| 2 epochs (54,832 steps) | 2.31e-4 | 5.8x |
| 4 epochs (109,664 steps) | 3.53e-4 | 8.8x |

The model then re-anneals over the added epoch. That is a legitimate training
regime and both twins get it identically, so a control-vs-ablated comparison stays
clean -- but it is *not* the paper's 1B-2E, which runs one cosine over two epochs
from random init. Do not compare a warm-restarted 2E against the paper's table.
`CosWithWarmup.t_max` would pin the horizon if a true continuation is wanted.

**Requeue truncates the job's logs.** Checkpoints survive preemption; `log.out` and
`log.err` are reopened empty, so the loss curve for every window except the last is
lost. `#SBATCH --open-mode=append` fixes it.

## Naming a run so it can be told apart

**The queue view truncates `JobName` to about 8 characters. Put the subject
first.**

On 2026-09-05 three 1B runs were in the queue together:

    untaught-control-1b-2e-b131k
    untaught-no-rome-core-1b-2e-b131k
    untaught-no-baseball-core-teams-1b-2e-b131k-h100

Every one of them displays as `untaught`. The subject -- the only thing that
differs scientifically between them -- sits 9 to 12 characters in, past the cut.
A name that needs the full string to be useful is not doing its job at the one
moment it is read.

Lead with the subject and drop the prefix every run in this project shares:

    control-1b-2e-b131k               ->  "control-"
    rome-core-1b-2e-b131k             ->  "rome-cor"
    baseball-teams-1b-2e-b131k-h100   ->  "baseball"

Order: `<subject>-<variant>-<size>-<epochs>-<batch>-<card>`. "untaught" carries
no information, because every run here is an untaught run.

Before submitting, compare the first 8 characters of `job.name` against every
job that will share the queue. If two collide, reorder. This applies to
`job.name` in `configs/`, to `#SBATCH --job-name` in the audit and evaluation
helpers, and to the `OUT_TAG` labels that end up in result filenames.

**Renaming a running job is not free.** `framework/client/keep_twins_running.sh`
matches by job name, and the run folder is named from `job.name` at submission
time, so `scontrol update JobName=` desyncs the SLURM name from the run
directory and from the watchdog. Rename at submission, or change the config and
the watchdog together.

## More defects, found 2026-09-06 evaluating the Ancient Rome twins

**`run_rome_heldout.slurm`'s convert mode hardcoded the checkpoint directory.**
It read `checkpoints/olmo2_1B_0.0003_32768_0.01_2/$STEP`, the authors'-config
pair it was written for. That directory name encodes model/lr/batch/wd/duration,
so the b131k twins live under `olmo2_1B_0.0004_131072_0.05_2` and convert aborted
with `FATAL: ... has 0 proper shards, expected 16` -- which reads as a corrupt
checkpoint and is not one. It now discovers the directory the way
`Untaught/convert_to_hf.sh` always did:

    CKPT_ROOT="$(ls -d "$UNTAUGHT_RUNS_DIR/$RUN"/checkpoints/*/ | head -1)"

Any helper that names a checkpoint directory literally has this bug latent in it.

**A resumed run has no `step0` of its own.** The control twin resumed from
step34000 of run 850054, so `untaught-control-1b-2e-b131k_20260905_221156`
begins its checkpoints at step35000 and the initial draw lives in the
predecessor folder `..._20260904_175321`. Anything comparing a final checkpoint
against initialisation -- `embedding_health.py` is the case here -- has to be
told where init actually is. Hence its `INIT_RUN`, defaulting to `RUN` for a run
that trained straight through. The predecessor also holds `step35000.incomplete`,
which is `auto_resubmit.sh` having correctly moved a truncated checkpoint aside:
a fourth instance of the incomplete-checkpoint defect, and the first one caught
by a machine rather than by hand.

**Did the embeddings learn, or only shrink.** `check_embedding_init` proves the
draw was right at step 0 and can say nothing about the other end.
`ember_eval/embedding_health.py` closes that: decoupled weight decay multiplies
every parameter by `(1 - lr_t * wd)` each step regardless of gradient, so the
product over the schedule predicts the row-norm a matrix would reach having
learned nothing. Observed/predicted near 1.00 is the pre-fix failure (0.5492
against 0.5471). Both Ancient Rome twins came out at **2.94** -- 1.4370 and
1.4377 against 0.4894 predicted -- with median row cosine to init of 0.29. The
fix delivered, and embedding-space methods have real structure to read on this
pair. The twins agreeing to 0.05% is also the cleanest available check that the
two runs differ only in the ablation.

## Choosing a partition, measured 2026-09-06

**`cpu-killable` refuses the `gpu-research` account.** `Invalid account or
account/partition combination specified`, with or without `--account`. A CPU-only
eval job therefore has to take a GPU partition and simply not use the card.

**`killable`'s own `--exclude` line can push its projected start hours out.**
The eval jobs exclude `rack-bgw-dgx1,rack-gww-dgx1,rack-omerl-g01` (missing
mounts) and `n-301,n-303` (pass `nvidia-smi`, fail torch's CUDA init). On
2026-09-06 killable was 41 deep and its only backfill slot was an excluded node,
so `sbatch --test-only` projected 18:49 for a 5-minute job -- seven hours out,
while the partition looked available. `gpu-h100-killable` was 0 deep and started
the same job immediately. **Always ask `sbatch --test-only` rather than reading
`sinfo`**; it is the only thing that accounts for excludes, memory and backfill
at once.

**Small eval jobs on a training node cost the training nothing measurable.**
Sixteen completion-eval jobs (4.4 GB, ~5 min each) were placed on n-102 beside a
1B training run. Its rate went from 2.63 to 2.36 s/step across the boundary --
faster, not slower, on 8 H100s with spare cards. What they *did* slow was
another eval job on the same node whose setup phase scans the 10.5M-instance
corpus: held-out ppl took 31 min alone and ~80 min under that load. The rule
"scoring must never compete with training" is about the *scarce* card and the
job that strands a node; it does not generalise to every co-tenancy.

**The 2026-09-05 H100 slowness was not the card.** The Ancient Rome baseball run
logged 15,759 TPS (8.33 s/step) on n-102 on 2026-09-05 and **53,494 TPS
(2.36 s/step) on the same node, same config, same 62.85 GiB footprint** on
2026-09-06. A 3.4x swing with nothing in the job changed. Any conclusion of the
form "this card is Nx slower than that one" taken from a single window is
unsafe here; re-measure before moving a run on throughput grounds.

## Cluster facts that are not obvious

**`/vol/scratch` is purged, and the window is days, not months.** On 2026-08-23
the whole of `/vol/scratch/galbarak2` was removed — not old files, the entire user
directory. Other users' directories were untouched and scratch usage fell 3.6 TB
to 1.7 TB. Files four days old were taken. There is no policy document anywhere
we could find, `df` shows nothing, and no warning is sent.

**What it cost:** all four converted HF models, and the 1-epoch twin pair's
OLMo-core checkpoints entirely. That pair is gone for good: retraining it was
considered and declined, and the project continues on the 2-epoch pair. The
2-epoch pair survived only because `UNTAUGHT_RUNS_DIR` defaults to
`${UNTAUGHT_ROOT}/runs`, so those runs wrote into the checkout on `/home/morg`
rather than to scratch. That default is the reason the experiment still exists.

**Therefore: never leave anything on `/vol/scratch` you cannot regenerate.** Use
it for intermediates only. Durable locations, both far larger than needed:

| path | free | notes |
|---|---|---|
| `/home/dcor/galbarak2` | ~8.2 T | lab NetApp, no quota against this user |
| `/home/morg/NLP_2526b/galbarak2` | ~37 T | where run folders default to; also 4.2x faster |

A converted model is ~5.1 GB and a full checkpoint ~15 GB, so keeping them on
either is trivially affordable. `convert_to_hf.sh` honours `HF_MODELS_DIR`, and
run folders honour `UNTAUGHT_RUNS_DIR`; set both away from scratch.

**The cheap insurance that was not taken:** on 2026-08-19 the absence of a
retention policy was noticed and a copy to `/home/dcor` was discussed and not
made, because the oldest surviving file in all of `/vol/scratch` was then eleven
weeks old, which read as evidence of no aggressive purge. It was not. Ten
gigabytes of copying would have kept a finished twin pair that is now gone.



**SLURM cannot send mail here, but the campus relay can.** `--mail-type` and
`--mail-user` are accepted by `sbatch` and then silently discarded: `scontrol
show config` reports `MailProg = /bin/mail`, that binary is not installed, and
there is no sendmail, mailx or postfix either. Jobs notify nobody, with no error
anywhere to tell you.

What works is talking to the campus relay directly. `smtp.tau.ac.il`
(`post.tau.ac.il`, 132.66.3.150) answers on port 25 from the login node, needs
**no authentication**, and accepts external recipients -- `RCPT TO` for a gmail
address returns `250 2.1.5 Ok`, and delivery to gmail is confirmed.
`framework/client/notify.py` wraps it, and `auto_resubmit.sh` uses it to mail on
completion, resubmission and giving up.

One wrinkle: `<user>@tau.ac.il` is not a deliverable local address here ("User
unknown in local recipient table"), so it works as an envelope sender but replies
go nowhere, and a strict receiver may treat it as suspicious.

**`squeue` hides the partitions you cannot submit to; use `squeue -a`.** This is
what makes preemption on `gpu-h100-killable` look inexplicable. `squeue -w n-102`
showed only our two jobs while the node itself reported 7 of 8 GPUs allocated --
the other five were in `gpu-n102`, a partition whose config `scontrol show
partition gpu-n102` will not even print for us. `n-102` belongs to two partitions:

| partition | tier | who |
|---|---|---|
| `gpu-h100-killable` | 10 | everyone |
| `gpu-n102` | higher, hidden | the node's owning group |

That is the node-owner arrangement -- a group buys the machine, keeps a private
partition on it, and the rest of the cluster reaches the same cards through a
`killable` partition that is evicted whenever the owner wants them. With
`PreemptType=preempt/partition_prio` and `PreemptMode=REQUEUE`, a higher-tier job
does not queue behind you; it takes the card and yours is requeued. To find out
who actually holds a node:

    squeue -a -w <node> --Format="JobID:9,Partition:20,UserName:12,State:9"
    scontrol show node <node> | grep AllocTRES     # ground truth
    sinfo -N -n <node> -o "%N %P %t %G"            # which partitions own it

`sacct` will not show other users' *finished* jobs, so the specific job that
evicted you is often unrecoverable after the fact. `State=PREEMPTED` on your own
record plus the partition list above is usually the whole story available.

**On a requeue, `resuming from ... (nothing found) -- random init` is a false
alarm.** There are two independent resume paths and that line reports only the
first. This framework's `find_previous_checkpoint` searches *other* run folders
and correctly finds none, because a requeue reuses the same folder. Upstream's
`maybe_load_checkpoint` then looks inside the run's own save folder and loads the
newest checkpoint there. The line that settles it is upstream's:

    Loading checkpoint from '.../checkpoints/<name>/step20000'

against the failure case, `No checkpoint found in save folder ... will train from
scratch`. Check for that one, not for ours.



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
