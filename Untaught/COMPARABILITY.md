# Why both twins have to be trained here

A tempting shortcut, at 1B especially: train only the ablated model and use one of
the authors' released checkpoints (`dhgottesman/LMEnt-1B-1E`) as the control. It
would halve the compute. It does not work, and the reason is worth writing down
because it is not obvious from the model card — which carries no training
hyperparameters at all, only a link to the paper.

## What genuinely matches

Two things check out, so this is a training-recipe problem rather than an
architecture or data problem.

**The architecture is identical.** `olmo2_1B` in the pinned OLMo-core resolves
through `llama2_1B` to `d_model=2048, n_layers=18, n_heads=16`, with the OLMo-2
overrides `rope_theta=500_000`, `qk_norm=True`, reordered-norm blocks and
`layer_norm_eps=1e-6`. The released `step109672/config.json` reports
`hidden_size 2048`, `num_hidden_layers 18`, `num_attention_heads 16`,
`rope_theta 500000`, `rms_norm_eps 1e-06`. Same model.

**The corpus is the same.** The paper reports 3.6B tokens across 10.5M chunks;
`tests/verify_chunk_alignment.py` measures 10,491,928 chunks in the deployed
dataset and the same number in the `lment_cs` index.

## What does not match

**The batch size, and therefore the whole optimization trajectory.** The released
1B publishes checkpoints at `step0, 10000, …, 109672` — the same step numbering as
the released 170M. Against a 3.6B-token epoch that pins its batch size:

    3.6e9 tokens / 109,672 steps ~= 32,800 tokens per step  ->  global_batch_size = 32,768

`configs/train_1b_control_full.yaml` sets `data_global_batch_size: 131072`, four
times larger. One epoch at that batch is roughly 27,500 optimizer steps, not
109,672. The two models would see the same tokens but take a quarter as many
steps, under a cosine schedule whose `optim_warmup_steps: 2000` covers 7% of
training in one case and 1.8% in the other. Nothing downstream of that is
comparable.

**The optimizer settings, on the 170M precedent.** The released 170M was trained
at lr `0.0003` / weight decay `0.01` (`kas_config.json`, and the authors' own
output directory is named `olmo2_170M_0.0003_32768_0.01_1`).
`configs/train_170m_control_full.yaml` uses `0.0005` / `0.05`. The Untaught
recipe deliberately diverges from the paper's, so the 1B almost certainly differs
too. The paper's own numbers are in Appendix B.4 if anyone wants to check.

## Why matching the recipe still would not be enough

The reason to train both twins is not really the hyperparameters — those could be
aligned. It is that the ablation is defined *relative to its control*.

`ChunkExclusionCallback` masks blacklisted instances out of the loss while leaving
the batch composition, the data order and the step count untouched. That is the
whole design: the ablated model differs from its control by exactly the masked
gradient contributions and by nothing else. Two runs of the same config on the
same node differ only in that mask.

An externally trained control forfeits that guarantee no matter how carefully the
config is matched, because it also differs in:

- **rank count** — the 1B config asks for `gpus: 4`, and FSDP reduces gradients in
  a different order at a different degree of sharding, so a 4-rank run is not
  reproducible against a run sharded differently;
- **code version** — the authors' OLMo-core at training time, versus the pinned
  submodule plus this framework's callback;
- **bf16 accumulation** — different hardware, different rounding.

Each is individually small and none of them is measurable against an effect of
0.025% of the corpus, which is what a single-entity ablation is.

## The cost argument points the same way

Reusing the released control would mean dropping to `global_batch_size: 32768` and
therefore **four times the optimizer steps** — 109,672 rather than ~27,500.
Matching their setup is more expensive than training your own pair *and* leaves a
weaker claim. Train both.

# Picking the subject, and auditing the run

A separate question with its own document: which subject is worth holding out, how
to confirm before training that the corpus teaches it, and how to prove afterwards
that the model met it. See `CHOOSING_A_SUBJECT.md`.
