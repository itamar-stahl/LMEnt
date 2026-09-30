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

**The optimizer settings.** Appendix B.4 has now been read, and it confirms the
batch size derived above while contradicting itself on the learning rate:

> A hyperparameter search was conducted over the following ranges: global batch
> size {16,384, 32,768, 65,536, 131,072, 262,144}, peak learning rate {3e-4, 6e-4,
> 8e-4, 1.2e-3, 3e-3, 5e-3}, and weight decay {0.005, 0.05, 0.1}. Hyperparameters
> were selected based on the minimal final training perplexity. All LMEnt models
> were trained using the AdamW optimizer with a global batch size of 32,768, rank
> batch size of 8,192, peak learning rate of **5e-3**, weight decay of 0.05, and
> 1,000 warmup steps.

Note what our own 1B batch of 131,072 is: a candidate in their search that lost
to 32,768 on final training perplexity. Not an arbitrary departure, but not their
answer either.

The learning rate has three sources and they do not agree:

| source | LR | weight decay |
|---|---|---|
| paper, appendix B.4 | 5e-3 | 0.05 |
| `kas_config.json` in this repo | 3e-4 | 0.01 |
| released 170M run directory `olmo2_170M_0.0003_32768_0.01_1` | 3e-4 | 0.01 |

Two artifacts the authors actually shipped say 3e-4; the appendix says 5e-3, the
top of its own search grid. The artifacts are the better evidence for what was
trained. `configs/train_1b_*_full.yaml` uses 4e-4 and the 170M configs use 5e-4 --
both nearer the artifacts than the appendix, and 5e-4 is a misreading of B.4's
5e-3 rather than a choice. **Do not "fix" it to 5e-3 without a pilot run**: 5e-3
is 12x above anything the artifacts support, and the pair that exists was trained
at 4e-4.

## Does the recipe difference cost knowledge? Measured, not argued.

A quarter as many optimizer steps is the objection that matters, because if our
twins learned less about the target concept then the ablation had less to remove.
Both models were scored by the same harness on the same 1,800 EMBER questions in
the same shuffled option order, so this is a matched-pair test
(`archive/ember_eval/compare_to_published.py`):

| | released 1B-1E | our control | McNemar p | mean d logP |
|---|---|---|---|---|
| concept questions (`QA_test`) | 34.1% | 32.6% | 0.307 | +0.078 |
| neighbouring domain (`SimdomQA_test`) | 50.1% | 46.9% | 0.047 | -0.003 |
| **Pornography, the ablation target** | **50.0%** | **48.0%** | 1.000 | -0.105 |

Indistinguishable on concept knowledge, marginally behind on the neighbouring
domain, and within two points on the concept the experiment actually removes. The
headroom the twin comparison needs was there. Four times fewer optimizer steps at
the same token count cost very little.

## What the paper says *would* have helped: epochs, not steps

Table 3 of the paper is the useful comparison, because it separates the two
things. From 1E to 6E, general ability barely moves while closed-book recall
multiplies:

| task | 1B-1E | 1B-6E |
|---|---|---|
| piqa | 0.557 | 0.555 |
| arc_easy | 0.421 | 0.446 |
| hellaswag | 0.295 | 0.328 |
| squad | 0.084 | 0.107 |
| naturalqs_open | 0.019 | 0.031 |
| sciq | 0.714 | 0.770 |
| jeopardy | 0.009 | 0.043 |

(triviaqa runs 0.006 / 0.047 / 0.008 / 0.025 across 1E/2E/4E/6E -- at that
magnitude the numbers are noise, so it is left out of the argument.)

Factual recall is the axis a concept ablation is measured on, and at one epoch
every chunk is seen exactly **once**. That is the weakest setting in the suite for
detecting what removing a concept did. Repeating the pair at 2E or 4E multiplies
the signal without touching the ablation's logic -- the same blacklist, the same
mask, more passes over what remains. It costs 2-4x the training time and nothing
else, and it is a far better lever than matching the paper's batch size.

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
weaker claim. The measurement above says it would also buy almost nothing. Train
both, and spend the compute on epochs instead.

# Picking the subject, and auditing the run

A separate question with its own document: which subject is worth holding out, how
to confirm before training that the corpus teaches it, and how to prove afterwards
that the model met it. See `CHOOSING_A_SUBJECT.md`.
