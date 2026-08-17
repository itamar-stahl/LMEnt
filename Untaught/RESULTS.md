# The 1B twin pair: what was trained

Two 1B models on the same 3.6B-token Wikipedia epoch, identical in every respect
except that one held every chunk mentioning **Pornography (`Q291`)** out of the
loss. Both finished on 2026-08-17.

## The models

| | Control | Ablated |
|---|---|---|
| Job | `761568` | `761569` |
| Held out | nothing | `Q291`, 2,546 chunks (0.0242% of corpus) |
| Steps | 27,416 / 27,416 | 27,416 / 27,416 |
| Final CE loss | 2.605 | 2.604 |
| Final perplexity | 13.53 | 13.51 |
| Exit | `COMPLETED` 0:0 | `COMPLETED` 0:0 |

Final checkpoints, 16 shards each, ~15 GB:

    /vol/scratch/galbarak2/untaught-runs/untaught-control-1b-full_20260817_091657/checkpoints/olmo2_1B_0.0004_131072_0.05_1/step27416
    /vol/scratch/galbarak2/untaught-runs/untaught-no-porn-1b-full_20260817_091657/checkpoints/olmo2_1B_0.0004_131072_0.05_1/step27416

Trained with `olmo2_1B`, lr 0.0004, weight decay 0.05, warmup 2000, global batch
131,072 tokens, rank microbatch 16,384, `kas_vsl` curriculum `grow_p2` over 8
cycles, seed 12536, one GPU per twin, `torch 2.6.0+cu124` on an H100 80GB.

## Did the ablation actually happen

Yes, and every signal agrees:

- `loaded 2546 chunk ids` at startup, from the run folder's own frozen artifact;
- **1,074 instance-slots excluded** from the loss over the final window's 11,915
  steps -- consistent with 43% of an epoch and 2,546 blacklisted chunks;
- **zero guard leaks**, so no blacklisted chunk ever reached the loss through the
  all-masked guard;
- the control logged `no blacklist configured` and stayed inert throughout.

One warning for whoever reads these logs next: the cumulative counter is printed
with a thousands separator (`untaught excluded cumulative=1,074`). A grep for
`[0-9.]+` truncates it to `1` and makes the ablation look like it stopped firing.
The `post_train` summary line is the number to trust.

## Why the losses are nearly identical, and why that is the point

13.53 against 13.51 is not a null result. Masking 0.024% of the corpus should not
move general language modelling, and if it had, any downstream difference would be
confounded with a broad capability gap rather than attributable to the missing
concept. The twins being indistinguishable on perplexity is what makes a
difference on concept-specific questions interpretable.

Whether such a difference exists is the open question. The evaluation is
`ember_eval/score_ember_mc.py` on `feature/ember_eval`; the control scored 50.0%
on EMBER's Pornography `QA_test` before any of this, which is the headroom the
ablated twin has to lose.

## Getting from here to an evaluation

The checkpoints are OLMo-core `.distcp` -- weights plus optimizer state, sharded,
readable only by OLMo-core. The eval harness wants HF `safetensors`. The converter
ships with the framework:

    OLMo-core/src/examples/huggingface/convert_checkpoint_to_hf.py

## What it took to get here

Recorded because the wall-clock is misleading. The successful run was 9h19m and
8h53m. Reaching it took roughly 30 hours:

- **Two windows on `gpu-h200-killable` were preempted 4 and 3 times each**, then
  killed outright by a NetApp `EIO` mid-checkpoint at 05:59 that took both twins
  in the same second.
- **A move to B200 failed** on `Missing key in checkpoint state_dict:
  optim.param_groups.0.decoupled_weight_decay`. B200 needs torch >= 2.7 for
  `sm_100` kernels, and torch 2.11 cannot read optimizer state written by 2.6.
  See `B200.md`.
- **The H200 node was then monopolised** by a single job holding all 192 CPUs
  alongside 4 of 8 GPUs for five days, stranding four idle H100-class cards and
  eleven queued jobs across five users. Slurm estimated a start four days out.
- **`gpu-h100-killable` worked immediately.** H100 is `sm_90`, inside torch 2.6's
  arch list, so the original environment runs there and the banked checkpoints
  load. Three cards were free on `t-100` while the head-of-queue job wanted six,
  so backfill placed both twins at once. Zero restarts from then on.

Three separate incidents left an incomplete newest checkpoint that
`latest_checkpoint_dir` selected anyway. `OPERATIONS.md` has the detail; the short
version is to check `ls <ckpt>/model_and_optim | grep -c '^__'` before trusting any
resume.
