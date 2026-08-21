# Where this project stands

Snapshot updated **2026-08-21**. Nothing is running on the cluster. This is the
document to read first; every claim here links to the document that establishes it.

## The one-paragraph version

Two twin pairs of 1B models were trained on the LMEnt Wikipedia corpus — one at
1 epoch, one at 2 — each identical except that one twin held every chunk
mentioning **Pornography (`Q291`)** out of the loss. All four finished, the
ablation is verified, and each pair's twins are indistinguishable on perplexity,
which is what makes any concept-specific difference attributable to the concept.
Whether the ablation is detectable **remains unresolved**: the 1-epoch pair shows
a concept-specific effect of the predicted shape, the 2-epoch pair does not
reproduce it, and EMBER ships only 200 questions per concept — a ±5–7 point
standard error against a difference of roughly 4 points. The largest single
finding of the evaluation work turned out to be about the *instrument*: these
base models cannot answer questions at all, and the prompt format is worth more
than every other measurement choice combined. See `ember_eval/EVALUATION.md`.

## Both twin pairs are finished; the task has moved to Tamar

| pair | control | ablated | state |
|---|---|---|---|
| 1 epoch | perplexity 13.53 | 13.51 | done, converted |
| 2 epochs | perplexity 12.110 | 12.122 | done, converted |

All four models are at `/vol/scratch/galbarak2/hf-models/lment-1b-{control,noporn}[-2e]`,
each with a README covering training, verification and the caveats. In the
base / never-learned framing used by erasure work, `M_base` is a control and
`M_never(Pornography)` is the matching ablated twin, so `D_target = W_never - W_base`
is defined over a matched pair.

Ablation verified on both pairs: 2,546 chunk ids loaded, zero guard leaks; the
2-epoch run excluded 4,640 instance-slots across its two SLURM windows (the
counter **resets per window**, so the framework's closing line reports only the
last one).

**One deviation, recorded rather than buried:** after four preemptions the
2-epoch control finished its last ~9.7% of steps on an **H200** while its twin
ran entirely on H100s. Both are `sm_90`, so the expected arithmetic difference is
far below anything being measured, but `COMPARABILITY.md` lists float
accumulation among the things that should not differ between twins. It matters
most for weight-space comparisons, which assume matched conditions.

## Read this before evaluating anything

The single biggest finding of the evaluation work is about the **instrument**,
not the ablation. These are base models with no instruction tuning:

- they **do not answer questions** — letter-parsing evaluators score them near
  zero, and every unparseable answer counting as wrong sinks the whole
  chance-corrected score;
- score option **text** by log-likelihood instead;
- **use a declarative stem**, not `Question: ...\nAnswer:` — worth +3 to +4
  questions out of 10 across six models;
- normalise **per character** (`acc_per_char`); raw summed log-likelihood is
  dominated by option length and was used by every analysis here before
  2026-08-21.

`ember_eval/EVALUATION.md` has the measurements, the corrections to earlier
claims, and the ranked list of what would actually settle the ablation question.
The most promising unrun measurement is **perplexity on the held-out chunks
themselves** — no prompt, no format, millions of tokens.

## What exists

**Two trained twins**, both `COMPLETED`, 27,416 / 27,416 steps, on `gpu-h100-killable`:

| | control | ablated |
|---|---|---|
| job | `761568` | `761569` |
| held out | nothing | `Q291`, 2,546 chunks (0.0242% of corpus) |
| final perplexity | 13.53 | 13.51 |

    /vol/scratch/galbarak2/untaught-runs/untaught-control-1b-full_20260817_091657/checkpoints/olmo2_1B_0.0004_131072_0.05_1/step27416
    /vol/scratch/galbarak2/untaught-runs/untaught-no-porn-1b-full_20260817_091657/checkpoints/olmo2_1B_0.0004_131072_0.05_1/step27416

Both verified complete: 16 `__N_M.distcp` shards, no `tmp*` leftovers. Earlier
`step20000`, `step25000` and `step27000` are also banked. ~60 GB per twin.

**Converted to HuggingFace**, 5.1 GB each, which is what the evaluation loads:

    /vol/scratch/galbarak2/hf-models/lment-1b-{control,noporn}

**Per-question evaluation records**, 3,600 questions x 2 models, holding every
continuous log-likelihood -- the reanalysis needs no GPU because these exist:

    /home/dcor/galbarak2/LMEnt-ember/ember_eval/results/twins4_{control,noporn}_763384.json

**Environments.** `lment` (torch 2.6+cu124, shared read-only from stahli) runs on
H100 and below. `/home/dcor/galbarak2/conda_envs/lment-b200` exists for Blackwell
and cannot load these checkpoints -- see `B200.md`.

## What is established

**The ablation fired.** 1,074 instance-slots excluded over the final window, zero
guard leaks, control inert throughout. `RESULTS.md`.

**The twins are otherwise indistinguishable.** 13.53 against 13.51 perplexity is
the point, not a null result: it is what makes a concept-specific difference
attributable to the concept rather than to a generally worse model.
`RESULTS.md`.

**There is a concept-specific effect, in the log-likelihoods.** Accuracy shows
nothing. Log P(correct answer) shows the ablated twin down 0.573 nats on the
concept's own questions -- worst of 18 -- with its specificity control at the null
mean, robust to trimming, replicated across both question halves. An
answer-key-blind surprisal probe finds nothing, so what moved is discrimination,
not fluency. `ember_eval/EVALUATION.md`.

**The training recipe is not the limitation.** Our 131,072-token batch takes a
quarter as many optimizer steps as the paper's 32,768 over the same 3.6B-token
epoch. Measured head-to-head against the released `LMEnt-1B-1E` on the same 1,800
questions: 34.1% against 32.6% on concept QA (McNemar p = 0.307), 50.0% against
48.0% on the ablated concept itself. The headroom was there.
`COMPARABILITY.md`.

## What has been ruled out

- **B200 for these checkpoints.** torch 2.11 cannot read optimizer state written
  by 2.6. B200 is for runs started from scratch only. `B200.md`.
- **"We undertrained."** Measured, above. Do not reach for it again without
  re-running `ember_eval/compare_to_published.py`.
- **Reusing the authors' checkpoint as a control.** The ablation is defined
  relative to *its* control; an external one forfeits that. `COMPARABILITY.md`.
- **Four GPUs per twin.** 2.54 s/step against 2.30 for one card, and 4.4x the
  GPU-hours. `OPERATIONS.md`.

## The open question, and what would close it

The effect is at the resolution limit of the instrument. Three levers, best first:

1. **More epochs.** The paper's Table 3 has closed-book recall multiplying from
   1E to 6E (jeopardy 0.009 -> 0.043, sciq 0.714 -> 0.770) while general ability
   stays flat. At one epoch every chunk is seen exactly *once*. More passes over
   the same corpus give the ablation more to have removed, without touching its
   logic. Two traps first: extending `max_duration` restarts from random init,
   and it re-plans the cosine into a warm restart -- both in `OPERATIONS.md`.
2. **More null concepts.** The assumption-free test floors at 1/18 = 0.056 because
   EMBER ships 18 concepts. Forty would put 0.025 in reach.
3. **A second ablated twin on a different concept.** Each should show the drop
   only on its own concept -- a within-experiment specificity control no
   reanalysis can substitute for. The most expensive and the most convincing.

## The documents

| file | what it settles |
|---|---|
| `STATUS.md` | this snapshot |
| `RESULTS.md` | what was trained, and proof the ablation fired |
| `ember_eval/EVALUATION.md` | what the evaluation found, and what it did not |
| `COMPARABILITY.md` | why both twins are trained here; our recipe against the paper's |
| `CHOOSING_A_SUBJECT.md` | how to pick a subject and audit that the model met it |
| `OPERATIONS.md` | cluster and framework defects that cost time |
| `B200.md` | why Blackwell is a dead end for these checkpoints |
| `SETUP.md`, `SMOKE_TEST.md`, `README.md` | getting a run off the ground |

## Branches

`itamars/main` is the shared branch and carries everything described here --
the training work under `Untaught/` and the evaluation harness under
`ember_eval/`. `feature/training` and `feature/ember_eval` are where that work
was done and are now merged in; either may still be ahead during active work.
`main` is the upstream paper repo and has no record of this project.
