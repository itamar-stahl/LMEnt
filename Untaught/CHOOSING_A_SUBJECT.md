# Choosing a subject to hold out, and proving the model met it

Harry Potter is the worked example in this repo, not an arbitrary one. It sits in
a narrow band that a good ablation target has to occupy, and most obvious
candidates fall outside it. This is what to check before spending a GPU-week, and
how to verify afterwards that the run did what it claimed.

## The band a target has to sit in

**Frequent enough to have been learned.** At `max_duration: 1 epoch` the
dataloader makes exactly one pass, so every chunk is seen *once*. Not "enough
times" -- once. Repetition only comes from more epochs, which is why the authors
released 1E/2E/4E/6E rather than a single model. So at one epoch the quantity that
decides whether the model knows a subject is the number of *distinct* chunks
mentioning it, and a subject with a few hundred chunks seen once is a weak target
no matter what a later audit confirms.

**Rare enough that removing it stays surgical.** This is the bound people miss.
Harry Potter is 2,643 chunks, 0.0252% of the 10,491,928-chunk corpus: mask it and
the ablated twin is otherwise indistinguishable from its control. Mask a subject
running to whole percentage points and the twin is a *generally worse model*, so
any difference measured downstream is confounded with broad degradation rather
than attributable to the missing concept. Density also drives the
`guard_all_masked` leak described below: the denser the subject, the likelier an
entire batch is masked and the guard has to let a row through.

**Self-contained enough that the knowledge is not recoverable elsewhere.** A
fictional franchise can genuinely be removed. A historical period cannot -- "World
War II" or "Nazism" is redundantly encoded across thousands of chunks that never
link the entity, so the ablated model relearns it from context and the ablation
silently fails while every metric still looks healthy. Mention counts hint at the
problem: `Q7310` (Nazism) has 212,054 mentions and `Q1163715` (baseball) 193,146,
against 22,036 + 13,004 for the two Harry Potter QIDs.

A related coverage limit belongs in the same thought. The blacklist only catches
chunks where the entity linker tagged the QID above the config's thresholds
(`threshold_hyperlinks`, `threshold_entity_linking`, `threshold_coref`,
`threshold_coref_cluster`). A chunk discussing Hogwarts or J.K. Rowling that never
links `Q8337` stays in training, and the model can acquire the concept from it.
Lowering thresholds catches more and admits more false positives.

## What every EMBER concept actually costs, measured

Run `tools_concept_footprints.py` to regenerate. Chunk counts use the same query
and thresholds a real run would (`hyperlinks 1.0`, the rest 0.6), against the
live `lment_cs` index; Pornography reproduces at exactly 2,546, the number its
run held out.

| concept | QID | chunks | % of 10,491,928 |
|---|---|---|---|
| World War II | `Q362` | 296,526 | 2.826% |
| COVID-19 pandemic | `Q81068910` | 50,878 | 0.485% |
| Nazism | `Q7310` | 49,121 | 0.468% |
| Baseball (= MLB) | `Q1163715` | 38,401 | 0.366% |
| Ancient Rome | `Q1747689` | 16,915 | 0.161% |
| Republic of Ireland | `Q27` | 16,910 | 0.161% |
| Artificial intelligence | `Q11660` | 10,795 | 0.103% |
| Cannabis | `Q2845` | 7,153 | 0.068% |
| Halloween | `Q251868` | 3,471 | 0.033% |
| Uranium | `Q1098` | 3,341 | 0.032% |
| Heroin | `Q60168` | 3,321 | 0.032% |
| Valentine's Day | `Q37587` | 2,886 | 0.028% |
| **Harry Potter** | `Q8337`+`Q3244512` | **2,643** | 0.025% |
| **Pornography** | `Q291` | **2,546** | 0.024% |
| Gambling | `Q11416` | 2,294 | 0.022% |
| Gun | `Q115472839` | 1,429 | 0.014% |
| Golf | `Q175074` | 938 | 0.009% |
| Culture of Greece | `Q1149548` | 278 | 0.003% |

The two subjects ever trained are the **two smallest viable targets in the set**,
essentially tied at 0.024-0.025%, and both produced effects at the edge of
detectability. If a larger but still surgical target is wanted, the 0.03-0.07%
band -- Cannabis, Halloween, Uranium, Heroin, Valentine's Day -- is 1.3x to 3x
bigger without approaching "generally worse model" territory.

## Two traps in resolving a name to a QID

**Do not rank candidates by how often they match the surface name.** That
aggregation counts candidate *proposals*, which rare and ambiguous entities
dominate, and it picks the wrong entity repeatedly:

| concept | ranked by name matches | ranked by footprint |
|---|---|---|
| Heroin | `Q1613803`, 34 chunks | `Q60168`, **3,321** |
| Uranium | `Q2985283`, 64 chunks | `Q1098`, **3,341** |
| Gambling | `Q654906`, 146 chunks | `Q11416`, **2,294** |

Six of eighteen were wrong this way, by factors of 16x to 100x. Rank by chunk
count, then verify.

**An entity-linked corpus indexes institutions, not concepts.** Every candidate
the index offers for "Baseball" is an organisation:

| QID | chunks | what it is | linked text |
|---|---|---|---|
| `Q1163715` | 38,401 | Major League Baseball | `MLB`, `the major leagues` |
| `Q84129` | 10,643 | Minor League Baseball | `Class D`, `Rookie-level` |
| `Q809892` | 4,833 | Baseball Hall of Fame | `the Hall` |
| `Q1069698` | — | MLB All-Star Game | `All-Star Game` |

None of them is baseball the sport. Wikipedia hyperlinks named organisations,
not common nouns -- nobody links the word "baseball" to the concept -- so
ablating `Q1163715` removes professional-player biographies while general
baseball knowledge survives untouched. **Gun** and **Golf** look the same way:
no candidate resembles the general concept. Those subjects may not be selectable
at all in this index, only the institutions that carry their names.

`Q291` is the happy exception: 2,546 chunks from only 286 name matches, meaning
most of its coverage comes from entity linking and coreference rather than
hyperlinks. That is unusually good for an abstract concept -- and still left the
facts EMBER asks about (Kama Sutra, Fanny Hill, the Miller test, Pompeii) in
articles that were never tagged `Q291`.

## Before training: is the subject in the corpus, and how much of it

Names are ambiguous and the tool will not guess:

    python -m framework.client.es_blacklist resolve --name "Ancient Rome"

It returns the top matching QIDs by mention count. Disambiguation is manual and
consequential -- a franchise, its characters and its individual works are separate
QIDs -- and the counts are what separate the real entity from its namesakes.

Then put the chosen QIDs in `blacklists/<subject>.json` and ask what a run would
actually exclude:

    python -m framework.client.es_blacklist count \
        --config configs/train_170m_no_harry_potter.yaml --preview 5

This applies the config's own index and thresholds, so its number is the run's
number, and `--preview` prints sample chunk text so you can see the exclusion is
landing on the right material. Mentions are always far more numerous than chunks,
since one chunk holds many: Harry Potter's 35,040 mentions resolve to 2,643
chunks. A count of zero means the QIDs are wrong, not that the subject is absent.

Both commands run in seconds against the live index, before any GPU time.

## Before training: would a model actually learn it

Exposure is not acquisition. The counts above prove what goes *in*; they say
nothing about what comes out. The cheapest way to settle that without training
anything is to evaluate a released LMEnt checkpoint -- they were trained on this
same corpus, so `LMEnt-1B-1E` at one epoch is a direct estimate of what an
unfiltered 1B trained here will know. A subject a released model cannot answer is
a poor target: there is no knowledge to remove and nothing for the twin comparison
to detect.

**But how you ask decides the answer, and the effect is not a constant offset.**
Measured 2026-08-27 on the 2-epoch control: the same 100 questions, the same
`acc_per_char` metric, the same model -- changing only the prompt format.

| concept | `Question:/Answer:` | declarative stem | gain |
|---|---|---|---|
| World War II | 52% | 61% | +9 |
| Baseball | 44% | 54% | +10 |
| Pornography | 51% | 54% | +3 |
| Harry Potter | 36% | 36% | 0 |
| COVID-19 pandemic | 24% | 23% | -1 |

The gain runs from -1 to +10 points and is concept-specific, so **a ranking taken
under one format cannot be shifted into the other.** Baseball alone moves from
fifth place to second.

**Consequence: do not pick a subject from any answerability measurement taken
before 2026-08-21.** That includes the 18-concept scoring in
`ember_eval/results/twins2e_control_770277.json` and anything derived from it. It
used `Question:/Answer:`, which predates the format finding in
`ember_eval/EVALUATION.md` and predates understanding why these base models
cannot be asked questions at all: no instruction tuning, so they continue text
rather than answer it. Those numbers are a hypothesis about which concepts are
worth *measuring*. They are not evidence about which concept to ablate, and a
GPU-week is too expensive to spend on the difference.

To qualify a candidate properly: write its declarative stems as
`ember_eval/completion_eval/stems_<concept>.py`, score a released checkpoint with
`evaluate_completion.py`, and read `accuracy` against the 25% floor. Until that
file exists for a concept, its answerability here is **unmeasured**.

## After training: proving the model met the subject

Two independent routes, and they must agree.

**Replay the dataloader.** `get_dataset_dataloader_for_evals(config, epoch)` in
`OLMo-core/src/examples/kas/train.py` rebuilds the exact loader from a config: it
seeds from `init_seed`, builds the dataset and calls `data_loader.reshuffle(epoch)`.
The batch sequence is therefore a deterministic function of (config, epoch), with
no dependence on the training run, so it can be reconstructed afterwards. Each
batch carries `batch["index"]`, the chunk ids in it -- the same field the callback
masks on. `OLMo-core/src/examples/kas/write_dataloader_batch_indices.py` walks the
loader and saves those per step. Intersect them with the chunk ids recorded in the
run's own `untaught_blacklist.json` and you get the exact steps at which the model
met the subject. Replay from the run folder's `config.yaml`, not from a config that
merely resembles it.

The structural fact that makes this work: masking happens in `pre_step` on an
already-composed batch, so it never changes batch composition or order. Control and
ablated twins traverse *identical* batches, and one replay audits both.

**Read the ablated run's own metrics.** `framework/node/exclusion.py` records
`train/untaught excluded instances` per step, `train/untaught excluded cumulative`
as a running total, and a `post_train` summary line. This route exists only for the
ablated twin -- on a control the callback is inert by design, which is exactly why
the replay is needed for the unfiltered model.

Because both twins see identical batches, the second route's total must equal the
first route's intersection. Disagreement means something is wrong.

**Check the guard did not fire.** `guard_all_masked` un-masks one row when a batch
would otherwise be entirely masked, which would divide by zero and produce a NaN
loss. That means a blacklisted chunk can legitimately reach the loss.
`train/untaught guard leaks` counts every occurrence and `SMOKE_TEST.md` lists it
as *must stay absent*. At a fraction of a percent of the corpus it should never
fire; it is the one path by which the ablation leaks, so a claim about an ablated
model has to show it stayed at zero.

## Cost

`resolve` and `count` are seconds. Evaluating a released checkpoint is hours on one
GPU. The replay iterates a full epoch of the dataloader and is the expensive one --
the authors ran it as a SLURM array job on `cpu-killable` with a one-day limit,
producing roughly 90 MB per epoch. It is CPU-only, so it never competes with
training for GPUs.
