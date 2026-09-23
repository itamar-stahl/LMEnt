# ember_eval — measuring what the ablations and the erasures did

The last stage of the pipeline. [`../Untaught`](../Untaught) trains the twin
pairs: a control and an ablated model, identical except that the ablated one held
every chunk mentioning one concept out of the loss.
[`../Ember-on-LMEnt`](../Ember-on-LMEnt) and [`../mlp_erasure`](../mlp_erasure)
take a finished control and try to remove the same concept *post hoc* — at the
embedding level (EMBER) and at the MLP level (RMU, SNMF). Those two do score
models, but only to choose their own settings: EMBER's pipeline evaluates the
control and the edited model as one of its stages, and `mlp_erasure/sweep_delta.py`
scores every delta x layer-range cell. This directory is where the arms are put on
one scale — the same questions, the same text, the same rule for a never-learned
twin and for every erasure — so "what never learning the concept looks like" and
"what erasing it looks like" become comparable numbers.

**Start with [`EVALUATION.md`](EVALUATION.md).** It is the durable guide — how to
measure 1B base models at all, which is the part that cost the most effort and is
most likely to save someone else's. 38 files in this repo cite it by name, 24 of
them outside this directory. Everything else here is either an instrument or the
record of one experiment.

---

## Which document answers what

Eleven markdown files sit at this directory's top level. One is the guide, eight
are records of specific experiments, one is a control, one is an internal audit.

| file | kind | question it answers |
|---|---|---|
| [`EVALUATION.md`](EVALUATION.md) | guide | How do you measure these models at all? Why generating an answer and parsing a letter fails on a base model, why option log-likelihood under a declarative stem works, why `acc_per_char` and not `acc_raw`, how much power 200 questions per concept actually buys, and what the QA instrument's noise floor is. Also carries an explicit list of corrections to its own earlier claims. |
| [`ROME_RESULTS.md`](ROME_RESULTS.md) | experiment | The Ancient Rome twins: what was trained, whether the ablation fired, and what it did. |
| [`BASEBALL_RESULTS.md`](BASEBALL_RESULTS.md) | experiment | The Baseball arm — ablation, EMBER erasure, and the places where the text instrument and the question instrument disagree about the residual. |
| [`AI_RESULTS.md`](AI_RESULTS.md) | experiment | The Artificial Intelligence twins: the third subject, and the first arm where the never-learned twin and the post-hoc erasures are all scored on one scale. |
| [`HELDOUT_RESULTS.md`](HELDOUT_RESULTS.md) | experiment | The Pornography pair's held-out-chunk loss: the ablation's trace on the text it masked, rather than on questions. |
| [`COMPLETION_RESULTS.md`](COMPLETION_RESULTS.md) | experiment | The first sentence-completion run on the 2-epoch Pornography pair, and the concept-QA null it returned. |
| [`NULL_CONCEPT_CONTROL.md`](NULL_CONCEPT_CONTROL.md) | control | What this evaluation reports when there is nothing to find: all three models scored on Harry Potter, which none of them ablated. Read it before believing any single-concept difference. |
| [`ERASURE_RESULTS.md`](ERASURE_RESULTS.md) | experiment | Does EMBER's erasure reach the state of never having trained on the concept? One concept (Rome), one erasure configuration, `pmi_per_char` on the `gold` option. |
| [`NLL_KL_RESULTS.md`](NLL_KL_RESULTS.md) | experiment | The same question over three concepts and three erasure methods, on a different instrument — answer-token NLL and full-vocabulary KL. Also the reference documentation for [`nll_kl/`](nll_kl), which has no README of its own. |
| [`OLMES_RESULTS.md`](OLMES_RESULTS.md) | experiment | The paper's own evaluation suite (OLMES, vendored at [`../third_party/olmes`](../third_party/olmes)) run on our twins, and whether its thousands of items are a way past EMBER's 200-question wall. |
| [`CODE_AUDIT.md`](CODE_AUDIT.md) | internal note | Can the results be relied on? A line-by-line audit of the scoring code and the records, done because several conclusions rested on it. Its verdict table says which results to trust and which not to. |

---

## The instruments

### Score a model — a converted Hugging Face model directory, and a GPU except where noted

| script | what it measures |
|---|---|
| [`score_ember_mc.py`](score_ember_mc.py) | EMBER's multiple-choice questions by log-likelihood rather than generation: the likelihood of each option's *text* as a continuation (the `text` scorer, stored as `text_sum` / `text_char` / `text_tok`), and the likelihood of `" A"`/`" B"`/`" C"`/`" D"` under EMBER's own prompt (`letter`). Option order is shuffled with EMBER's stable item id and per-question seed, so item N here is item N there. |
| [`stem_probe.py`](stem_probe.py) | Whether a declarative stem beats `"Question: …\nAnswer:"` for a base model. Scoring is unchanged; only the context differs, so any difference is the prompt's doing. |
| [`format_metric_bakeoff.py`](format_metric_bakeoff.py) | Six models x two prompt formats x the OLMES metric family (`acc_raw`, `acc_per_token`, `acc_per_char`, `acc_uncond`) on one question set. This is where `EVALUATION.md`'s format and normalisation conclusions come from. |
| [`generate_probe.py`](generate_probe.py), [`mc_generate_probe.py`](mc_generate_probe.py) | Diagnostics, not evaluations. Print what the twins actually write — under EMBER's open prompt, and under its full A–D prompt verbatim. The scorers never generate, so nothing else in here can show you this. |
| [`embedding_health.py`](embedding_health.py) | Did the input embeddings learn anything, or only shrink? Compares final row norms against what AdamW's decoupled weight decay alone predicts over the LR schedule. Reads `model.embeddings.weight` out of a **raw OLMo-core `.distcp` checkpoint**, and is the one script here that does; the tensors never touch the GPU. |
| [`materialize_erased_model.py`](materialize_erased_model.py) | Not a measurement. Turns EMBER's embedding-only output (`erased_embeddings.safetensors`) into a full, loadable HF directory: rewrites the shard that holds `model.embed_tokens.weight`, carries the separate untied `lm_head.weight` across untouched, and checks the artifact's provenance hashes first. |

### Reanalyse stored records — no GPU, no model, no re-inference

Every continuous value the scorers compute is already in the results JSON, so
these are reanalyses rather than new runs.

| script | question |
|---|---|
| [`paired_twins.py`](paired_twins.py) | McNemar's exact test on the questions where exactly one twin was right. Pairing is available because both twins answer identical questions in identical option order. |
| [`loglik_twins.py`](loglik_twins.py) | The same comparison on continuous statistics instead of one bit per question — `logp`, `margin` (gold minus best distractor) and `optlp` (per-token likelihood of all four option texts, which never looks at the answer key). |
| [`erasure_vs_twins.py`](erasure_vs_twins.py) | Three paired contrasts on one question set: `ablated − control` (also a regression check on the scoring), `erased − control`, and the residual `erased − ablated`. The scoring rule is imported from [`completion_eval/metric_bakeoff.py`](completion_eval/metric_bakeoff.py) rather than reimplemented. |
| [`cross_concept_null.py`](cross_concept_null.py) | How far two separately trained twins drift apart on concepts neither of them touched — the null distribution the ablated concept's own number has to beat. |
| [`twin_vs_twin.py`](twin_vs_twin.py) | Uses the *other* ablated twin as the reference instead of the single shared control, and decomposes the null spread into a sampling half and a systematic half. |
| [`epoch_curve.py`](epoch_curve.py) | Does training longer give an ablation more to remove? Reads the authors' released 1E/2E/4E/6E scored by one harness. |
| [`compare_to_published.py`](compare_to_published.py) | Our control twin against the authors' released `LMEnt-1B-1E`, on 1,800 matched questions — whether our batch-size and LR differences left less concept knowledge for the ablation to remove in the first place. |

### Subdirectories

| directory | what it is |
|---|---|
| [`completion_eval/`](completion_eval) | The sentence-completion instrument: EMBER's questions rewritten as declarative stems and scored by continuation log-likelihood. The primary number is `gold_per_char = log P(gold \| stem) / len(gold)`; every option's conditional and null log-probability is also stored, so `metric_bakeoff.py`'s rules — including the `pmi_per_char` that `ERASURE_RESULTS.md` and `cross_concept_null.py` read — are recomputable from a record without re-running inference. It has **its own documentation** — [`completion_eval/README.md`](completion_eval/README.md) for the method, [`completion_eval/COMPLETIONS.md`](completion_eval/COMPLETIONS.md) for every question in readable form, and [`completion_eval/PROVENANCE.md`](completion_eval/PROVENANCE.md) for where the directory came from and what in its README no longer holds. The built question set is [`completion_eval/data/completion_questions.json`](completion_eval/data/completion_questions.json), 18 concepts x 4 splits x 50 questions, one `stems_<concept>.py` module behind each. |
| [`heldout_ppl/`](heldout_ppl) | Per-token loss on exactly the chunks the ablated twin received no gradient from, plus a matched control set drawn to the same VSL length histogram. [`heldout_ppl/check_indexing.py`](heldout_ppl/check_indexing.py) and [`heldout_ppl/check_indexing_concept.py`](heldout_ppl/check_indexing_concept.py) prove `chunk_id` resolves to the masked text before a GPU is spent; [`heldout_ppl/verify_loss.py`](heldout_ppl/verify_loss.py) recomputes the one loss line three ways; [`heldout_ppl/compare_heldout.py`](heldout_ppl/compare_heldout.py) forms the difference-of-differences; [`heldout_ppl/lexical_tail.py`](heldout_ppl/lexical_tail.py) tests whether an erasure's damage tracks the token ids it edited. Chunk-loss results: `HELDOUT_RESULTS.md`, `ROME_RESULTS.md`, `BASEBALL_RESULTS.md`, `AI_RESULTS.md`; the lexical test is in `ERASURE_RESULTS.md`. |
| [`nll_kl/`](nll_kl) | Answer-token NLL and full-vocabulary KL over a grid of erasure candidates. [`nll_kl/build_sets.py`](nll_kl/build_sets.py) draws target / neighbour / unrelated sets with a fixed seed and a selection/test split; [`nll_kl/score_model.py`](nll_kl/score_model.py) scores one model; [`nll_kl/select_checkpoint.py`](nll_kl/select_checkpoint.py) holds the selection rule, fixed before any test number is looked at; [`nll_kl/compare.py`](nll_kl/compare.py) and [`nll_kl/report.py`](nll_kl/report.py) build the tables; [`nll_kl/finalize.py`](nll_kl/finalize.py) drives the whole thing idempotently, and [`nll_kl/audit.py`](nll_kl/audit.py) and [`nll_kl/check_alloc.py`](nll_kl/check_alloc.py) exist because a sweep once produced holed checkpoints and unscored candidates while every job reported `COMPLETED`. Documented in [`NLL_KL_RESULTS.md`](NLL_KL_RESULTS.md). |
| [`fewshot_probe/`](fewshot_probe) | Does few-shot prompting make these base models answer EMBER's questions by generation, which zero-shot they do not? Two small probes, both printing the generations verbatim. |

---

## Running things

Every measurement here is a SLURM job. Fourteen `run_*.slurm` drivers sit at the
top level, and five more under [`nll_kl/slurm/`](nll_kl/slurm) — named for what
they do rather than `run_*`. They are worth describing as a pattern rather than
one by one, because they share almost everything:

- **Parameters arrive through the environment**, not on the command line:
  `sbatch --export=ALL,MODEL_PATH=…,OUT_TAG=… run_one_eval.slurm`. Two drivers
  take a `MODE=` selector instead of being split into separate files —
  [`run_heldout_ppl.slurm`](run_heldout_ppl.slurm) takes `check` / `ppl` /
  `verify`, and [`run_rome_heldout.slurm`](run_rome_heldout.slurm) takes
  `check` / `convert` / `ppl`.
- **`--account=gpu-research --partition=killable`, one GPU, and usually no
  `--constraint`.** A 1B in fp32 is about 4.4 GB and fits any card in that
  partition, so the job takes whatever is free instead of queueing for a type.
  Staying off the H100 partitions is deliberate: scoring must never compete with
  training for the scarce cards. Two drivers deviate, both with the reason
  recorded in the file — [`run_twins_eval.slurm`](run_twins_eval.slurm) pins
  `--constraint=a5000` so nothing can preempt it, and
  [`run_embedding_health.slurm`](run_embedding_health.slurm) asks for
  `gpu-h100-killable` because plain `killable` was 41 deep with its only
  backfill slot on an excluded node, and it sits at tier 10 there so it cannot
  preempt anything.
- **A standing `--exclude` list** on eleven of the fourteen, for two failure
  modes that both look like code bugs. The `rack-bgw-dgx1` / `rack-gww-dgx1` /
  `rack-omerl-g01` boxes do not mount the filesystem the models live on, so
  `transformers` reinterprets an absolute path as a repo id; `n-301` and `n-303`
  pass `nvidia-smi` but fail torch's CUDA init, so the job lands on a GPU node
  and silently runs on CPU — five drivers add those two. The `nll_kl/slurm/`
  drivers exclude a further set for a third reason: `n-202`–`n-205` have a filer
  link slow enough to leave scoring jobs 20+ minutes in D state with the model
  already on the GPU. Three drivers ([`run_one_eval.slurm`](run_one_eval.slurm),
  [`run_completion_eval.slurm`](run_completion_eval.slurm) and
  `run_completion_eval_local.slurm`), plus `nll_kl/slurm/score.slurm`, also check
  `config.json` is readable and exit with a named error before loading anything.
- **fp32, not bf16.** Scoring is cheap, and a log-likelihood comparison between
  two near-identical models should not turn on rounding.
- **Do not run these on a login node.** Importing torch and `olmo_core` off the
  shared NFS home took over 20 minutes there and never reached the first print; a
  compute node does it in seconds.
- **Output lands in `results/`**, which [`.gitignore`](.gitignore) ignores in
  full — the records are megabytes each and are not committed. For twelve of the
  fourteen, job stdout/stderr go to the same tree (`results/`,
  `results/completion/`, `results/heldout/`, `results/olmes/`,
  `results/fewshot/`); [`run_rome_heldout.slurm`](run_rome_heldout.slurm) and
  [`run_lexical_tail.slurm`](run_lexical_tail.slurm) instead write to a working
  directory outside the repo, and the `nll_kl/slurm/` drivers carry no
  `--output` line at all, so the submitter supplies one. A fresh clone therefore
  has no records, and the reanalysis scripts above have nothing to read until a
  scoring job has produced some.

The `nll_kl/slurm/` drivers are the newer generation: all five require
`ROOT=<checkout>`, and `finalize.slurm` also takes `RUN=<run root>`, so they run
whichever code tree you point them at. They still hardcode the interpreter and
`HF_HOME`, and the top-level drivers hardcode more than that — see below.

---

## Things that cost a day if you get them wrong

**A converted checkpoint, not a raw one, and the conversion is a separate step.**
Training writes sharded `.distcp` — weights *and* optimizer state, readable only
by OLMo-core, around 15 GB. Every script here that loads a model does so through
`AutoModelForCausalLM`, which needs safetensors. Nothing in the training pipeline
converts, so [`../Untaught/convert_to_hf.sh`](../Untaught/convert_to_hf.sh) has to
run first, naming the tokenizer explicitly. The single exception is
[`embedding_health.py`](embedding_health.py), which reads the `.distcp`
checkpoint directly and would fail on a converted directory. Two drivers fold the
conversion into the job so the ordering cannot be got wrong:
[`run_rome_heldout.slurm`](run_rome_heldout.slurm) as its `MODE=convert` stage,
and [`run_step45000_diag.slurm`](run_step45000_diag.slurm) by converting and then
scoring in one run.

**An EMBER erasure is not a model until you materialise it.** A run with
`save.full_model: false` writes one tensor, `erased_embeddings.safetensors`.
[`materialize_erased_model.py`](materialize_erased_model.py) is what turns that
into something the instruments here can load, and it is not a copy with a file
swapped: the edited embedding shares a shard with 199 other tensors, and
`lm_head.weight` is a separate untied tensor that must be carried across
unchanged.

**The `lment` conda environment ships an incomplete `olmo_core`.** It has
`distributed`, `io`, `stream` and `utils` but no `data/`, so
`from olmo_core.data import …` fails with a `ModuleNotFoundError` that reads as
though the package were absent. The complete sources are vendored in the repo and
[`../Untaught/framework/env.sh`](../Untaught/framework/env.sh) is what puts them
on `PYTHONPATH` ahead of site-packages — the same file the training jobs source,
so the dataset is rebuilt by exactly the code path that trained the twins.
[`run_heldout_ppl.slurm`](run_heldout_ppl.slurm),
[`run_rome_heldout.slurm`](run_rome_heldout.slurm) and
[`run_lexical_tail.slurm`](run_lexical_tail.slurm) source it, and those are the
jobs whose scripts import `olmo_core.data`.
[`run_embedding_health.slurm`](run_embedding_health.slurm) sources it too, for
the run-folder variables rather than for `olmo_core`.

**The top-level drivers carry absolute paths from specific cluster checkouts**, in
`EVAL=`, `PY=`, `HF_HOME` and the `#SBATCH --output` lines — three different
checkout roots appear across them, and two different conda prefixes both named
`lment` — so they need editing for any other checkout or account. Two specifics
worth knowing: `run_olmes.slurm` line 30 and `run_step45000_diag.slurm` line 35
still set `OLMES=…/LMEnt/olmes`, which is now
[`../third_party/olmes`](../third_party/olmes); and
[`run_twins_eval.slurm`](run_twins_eval.slurm) hardcodes the retired 1-epoch
Pornography pair under `/vol/scratch`, which was purged (see
[`../Untaught/OPERATIONS.md`](../Untaught/OPERATIONS.md)).
[`run_one_eval.slurm`](run_one_eval.slurm) is the parameterised single-model
equivalent and is the one to reach for.

**`completion_eval/` arrived as a third-party copy and has since diverged.**
[`completion_eval/PROVENANCE.md`](completion_eval/PROVENANCE.md) records where it
came from and the corrections that were deliberately *not* made to its files: its
model paths are dead, and its "Six concepts" and "1,400 questions" counts predate
the current set, which is 18 concepts and 3,600 questions. Read it before
trusting anything in `completion_eval/README.md`. But the directory is no longer
the untouched copy that document describes, so do not treat a clean `diff`
against the original as a guarantee: the primary metric changed from `pmi` to
`gold_per_char` on 2026-08-27, `build_completions.py`'s `attach()` was patched so
`Culture of Greece` can be built at all, nine more `stems_<concept>.py` modules
were added and the question set rebuilt, and the vendored-OLMES references in its
`README.md` and `COMPLETIONS.md` were repointed to `third_party/olmes/`.

**Accuracy is not the statistic to steer on.** The question bank is capped at 50
items per split, 200 per concept, and `EVALUATION.md`'s power analysis finds that
accuracy cannot resolve the erased-versus-ablated residual at that n by a factor
of somewhere between 4 and 64 — while the continuous per-question statistic
already stored in the same records does resolve it. Report a continuous
per-question statistic as primary, name which one (`gold_per_char` and
`pmi_per_char` are not interchangeable), and keep accuracy as a descriptive
number.

**Rankings taken before 2026-08-21 are not merely shifted, they are void.** They
were computed on `acc_raw`, which is dominated by option length, and under the
`Question:/Answer:` format whose penalty was measured on 2026-08-27 and is
concept-specific — so anything that ranked or *chose between* concepts on those
scores cannot be carried over. `EVALUATION.md` lists the specific claims
retracted; `CODE_AUDIT.md` lists which stored results survive scrutiny and which
do not.
