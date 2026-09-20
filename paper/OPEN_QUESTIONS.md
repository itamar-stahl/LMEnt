# Paper Open Questions

Track unsettled facts here instead of guessing in the LaTeX. Status labels are
**answered from repository**, **partially answered**, **needs verification**,
**ask teammates**, and **pending experiment**.

## Remote-main merge audit (2026-09-20)

Remote `origin/main` at `aa5e179` added a completed AI-excluded twin and a
common-harness AI comparison of EMBER, SNMF, and two RMU strengths
(`ember_eval/AI_RESULTS.md`). It also added fixed-code RMU/SNMF runs
(`mlp_erasure/DELTA_FIX_RERUN_RESULTS.md`) and a Rome/ Baseball EMBER delta
sweep with lexical-tail analysis (`ember_eval/ERASURE_RESULTS.md`). Old
SNMF results at deltas 4, 10, or 20 must not be called removal: the corrected
operator removes the selected component at delta 1. The expanded Rome EMBER
grid turns over beyond delta 200, so the former ``unbracketed optimum'' claim
is retired. The new AI question result prevents a blanket claim that EMBER
never approaches its excluded twin. The AI result does not establish text-level
or parameter-space equivalence.

New questions to resolve before submission:

- **MQ-1:** What exact AI EMBER delta candidates, selection items, and
  chronology led to delta 5? Archive the run-local report and judge manifest.
- **MQ-2:** What are the final AI RMU/SNMF checkpoint hashes, learning rates,
  factorization layers, and pre-comparison selection rules? The result note
  gives settings and paths but not one complete publication-ready config table.
- **MQ-3:** Can all AI methods be scored on identical held-out/control chunk
  IDs and compared in parameter space? The current common comparison is only
  the question harness; the AI twin's own chunk-loss validation is a different
  result.
- **MQ-4:** Can Rome and Baseball RMU/SNMF checkpoints be rescored alongside
  their excluded twins with the same prompt bank, chunk IDs, and scoring code?
  The corrected MLP investigation has useful null probes, but some use a
  different question scale or Rome chunk sample.
- **MQ-5:** How many distinct edited token *types*, rather than occurrences,
  appear per chunk? The lexical-tail analysis raises this as a hypothesis for
  the Rome/Baseball high-density contrast, not a demonstrated mechanism.
- **MQ-6:** Can the raw per-chunk delta-sweep outputs, AI completion outputs,
  MLP sanity reports, and run-local configs be put in a versioned supplement?
  Current summary tables cite cluster or gitignored artifacts.

## Experimental design

### ED-1: What exactly counts as removing a concept?

**Status: Answered from repository, with one framing decision to confirm.**

The framework does not physically delete documents. It resolves a blacklist of
Wikidata QIDs to LMEnt chunk IDs using hyperlinks, entity linking, coreference,
and coreference-cluster scores. Matching rows remain in their original batch
positions, but their labels are set to the ignore index, so they contribute no
language-model loss or gradient.

Evidence:

- `Untaught/README.md`: "What this actually does," "Why mask instead of
  delete," and "Changing the concept."
- `Untaught/framework/node/train_untaught.py`: blacklist loading and masking.
- `Untaught/blacklists/ancient_rome_core.json` and
  `Untaught/blacklists/baseball_core_teams.json`: final QIDs, footprints, and
  construction rationale.

Paper caveat: define "never-trained" precisely. These are more conservatively
called **data-ablated** or **concept-excluded** models: they saw the same batch
slots but received no learning signal from blacklisted chunks. The blacklist is
an operational approximation to a concept, not proof that all related facts
were absent from training.

Known construction history:

- Ancient Rome uses 56 QIDs and masks 65,844 chunks (0.628% of the corpus). It
  replaced a one-QID blacklist after a preliminary corpus audit found that the
  narrow set covered only about 8% of sampled Rome-bearing text. The wider set
  was estimated at about 46% coverage and 62.5% precision. Thus, the final set
  was informed by a corpus audit and failure of the earlier narrow ablation; it
  was not fully pre-registered.
- Baseball uses core institutional entities plus 30 MLB franchises: 43 QIDs
  and 80,466 chunks (about 0.767%). The final training configuration records a
  corpus-only precision/coverage audit on 2026-09-05 and explicitly says the
  core-plus-teams set was chosen on that audit, not on QA results.

**Ask Adam/team:** Should we describe blacklist construction as an exploratory
corpus audit? This is the most transparent framing.

### ED-2: Were the control and concept-excluded models genuinely matched?

**Status: Answered by the project team; run-local files remain to be archived.**

The intended twins share the OLMo-2 1B architecture, initialization seed
(12536), data seed (0), optimizer, schedule, corpus order, batch size, and
duration. Because masking happens after batch composition, each arm traverses
the same batches and steps. One completed control is shared by Ancient Rome,
Baseball, and AI. The project team confirms that this matched setup was used
in the completed runs. The tracked-template comparison below covers
Rome/Baseball.

A semantic comparison of the tracked YAML files confirms that all three have
identical `train` blocks. Ancient Rome differs from control only in `job.name`
and `untaught`; Baseball additionally differs in scheduling fields such as
partition and time limit. The Rome and Baseball records also give the same
final step (54,832) and core training recipe.

Evidence:

- `Untaught/configs/train_1b_control_2e_b131k.yaml`
- `Untaught/configs/train_1b_no_ancient_rome_core_2e_b131k.yaml`
- `Untaught/configs/train_1b_no_baseball_core_teams_2e_b131k_h100.yaml`
- `Untaught/model_cards/lment-1b-nobaseball-2e-b131k.README.md`, which explicitly
  identifies the shared control and identical batch composition.

Known qualification: the new Rome and AI model cards place the shared control
on H200 throughout. Rome also used H200 throughout; Baseball switched from
H100 to H200 for its final 2,832 steps, and AI had an A6000 window. The older
Baseball card instead says the control switched GPU late. Reconcile that card
against run logs before interpreting directional weight comparisons.

Optional provenance collection, not a condition for reporting the team's
confirmed training setup: copy each final run's `config.yaml` and
`run_environment.json` into the project archive for the shared control,
no-Rome, no-Baseball, and no-AI models. If available, also collect:

- checkpoint IDs and resume lineage across training windows;
- the guard-leak metric or warning logs for each excluded arm.

### ED-3: How were method hyperparameters selected?

**Status: EMBER partly answered; RMU/SNMF runs exist, but final selection
chronology and common-protocol scope need teammate verification.**

The EMBER notes say edit strength was selected using EMBER's own
train/specificity objective, not by optimizing similarity to the never-trained
model. Baseball feature/cell selection used feature quality, and its delta was
not re-selected after inspecting the twin comparison.

Evidence:

- `ember_eval/ERASURE_RESULTS.md`: "Which split the delta was chosen on."
- `ember_eval/BASEBALL_RESULTS.md`: feature, integrity, and delta-selection
  notes.
- Relevant configurations under `Ember-on-LMEnt/configs/`.

Questions for teammates before RMU/SNMF enter the main comparison:

1. What candidate grid was searched for each method and concept?
2. Which data and metric selected the final configuration?
3. Were never-trained outputs inspected before selection?
4. Was one rule used across both concepts, or was each tuned separately?
5. Which pre-specified integrity rule excluded failed runs?

Recommended policy: select configurations using method-internal training data
or a validation split derived from the original model, freeze them, and reserve
never-trained comparisons for evaluation. If tuning already used those results,
disclose it and label the analysis exploratory.

## Evaluation and results

### EV-1: What is the final common set of examples and metrics?

**Status: AI question comparison verified as common-harness in its result
summary; other dimensions and Rome/Baseball MLP comparisons need verification.**

Confirm that control, concept-excluded, EMBER, RMU, and SNMF use identical
examples, tokenization, prompts, scoring, and aggregation within each behavioral,
text-loss, parameter-space, and specificity evaluation. Observed-text NLL is
not a full output-distribution divergence.

### EV-2: What uncertainty will be reported?

**Status: Open analysis decision.**

Choose paired confidence intervals, tests, or effect sizes where appropriate.
Do not treat tokens from one passage as independent observations without
justification.

### EV-3: What parameter-space comparison is defensible?

**Status: Open analysis decision.**

Pre-specify tensors, normalization, distances, and baselines. Distinguish
whole-model and embedding-only distances, and compare each erasure displacement
with the control-to-never-trained displacement. Account for training
nondeterminism and the H100/H200 qualification.

## Writing and scope

### WR-1: Will robustness remain in the paper?

**Status: Optional; pending a concrete experiment.**

Do not reserve substantial main-paper space until a recovery/attack protocol is
defined and applied consistently.

### WR-2: Where do general descriptions belong?

**Status: Answered as a writing rule.**

- **Background/Related Work:** what LMEnt offers generally, what OLMo-2 is, the
  general idea of controlled training, and how EMBER/RMU/SNMF work in prior work.
- **Experimental Design:** the exact corpus, checkpoints, OLMo-2 1B setup,
  blacklists/training construction, and method configurations used here.
- **Evaluation Framework:** test sets and metric definitions.
- **Results:** observed outcomes after validity checks.

Some topics therefore appear twice at different levels: conceptual context in
Background and a reproducible project-specific specification in Experimental
Design. Do not repeat the same prose.

## Full-paper completion checklist

The LaTeX contains a complete narrative without visible drafting placeholders.
The questions below track remaining evidence and submission decisions. Resolve
the **P0** items before making stronger comparative claims; they can change the
conclusions.

### P0: Results that can change the paper's conclusion

#### CQ-1: Did RMU produce a valid persistent checkpoint for both concepts?

**Owner: teammates running RMU. Status: AI persistent checkpoints completed;
Rome/Baseball common-protocol twin comparison pending.**

For each concept, record the final model path/hash, edited layer/tensor names,
forget and retain datasets, layer window, steering coefficient, retain weight,
learning rate, batch size, update count, seed, grid, and selection metric. Then
run the identical question, chunk-NLL, control, OLMES, and weight-space scripts.
An activation diagnostic without a saved and reloaded checkpoint is not enough.

#### CQ-2: Did SNMF produce a valid persistent checkpoint for both concepts?

**Owner: teammates running SNMF. Status: AI persistent checkpoint completed;
Rome/Baseball common-protocol twin comparison pending.**

Record the final model hash, factorized layers, rank/sparsity, concept and neutral
sentence sets, judge version, selected features, edited projections, delta grid,
selection rule, and integrity result. Confirm the erased model survives save and
reload and changes only intended tensors before evaluating it.

#### CQ-3: Were RMU/SNMF selected independently of the never-trained outcomes?

**Owner: teammates. Status: Ask teammates.**

Request a chronological answer, not only the final rule: what results had been
seen when each configuration was chosen? If the never-trained comparison guided
selection, label the result exploratory and do not present it as held-out.

#### CQ-4: What is the final cross-method conclusion?

**Owner: paper group. Status: Partly answered on AI questions; blocked on
remaining common evaluations in CQ-1--CQ-3.**

Decide whether EMBER's mismatch is smaller, larger, or qualitatively different
from RMU/SNMF on each dimension. Do not rank methods using one arbitrary combined
score. Update the abstract, introduction preview, overall results, discussion,
limitations, and conclusion together.

### P0: Validity and metric verification

#### CQ-5: Does Ancient Rome have a verified parameter-space result?

**Status: Repository audit complete; pending experiment.**

No Ancient Rome weight-comparison JSON, table, or reported values are tracked.
`ROME_RESULTS.md` covers the trained twins and `ERASURE_RESULTS.md` covers
behavioral and chunk-NLL comparisons, but neither contains the directional
parameter calculation. This is not a hidden completed result: run
`compare_weights.py` on the final aligned control, no-Rome, and Rome-EMBER
checkpoints, then archive its JSON. Report global and embedding-only cosine,
progress, and residual.

#### CQ-6: What does `rel_edit_size = 0.0250` mean in the Baseball notes?

**Status: Answered from repository; misleading result-note gloss corrected.**

`compare_weights.py` defines `rel_edit_size` as
`||D_erase|| / ||W_base||`. Thus 0.0250 means that the erasure update norm is
2.5% of the base-model parameter norm; it does **not** mean that erasure moved
2.5% as far as the ablation. Progress along the ablation displacement is the
separate `progress_along_target` field (2.64e-05 in the Baseball summary). The
incorrect prose in `ember_eval/BASEBALL_RESULTS.md` has been corrected. The raw
JSON is not tracked, so the numeric value itself still depends on that result
summary's provenance.

#### CQ-7: Are all methods evaluated on byte-identical items and settings?

**Status: Needs verification after RMU/SNMF complete.**

Compare item IDs, prompt/stem versions, tokenizer hashes, null context, fp32
scoring mode, max sequence length, and aggregation code across every model.
Store the check result. A shared script name is not sufficient if inputs differ.

#### CQ-8: Which question statistic is the final primary outcome?

**Status: Answered from repository and wording standardized.**

The primary three-model comparisons use the gold option's `pmi_per_char`, as
specified by `RULE = "pmi_per_char"` and `STAT = "gold"` in
`erasure_vs_twins.py`; the Rome and Baseball three-way result tables say the
same. It is conditional-minus-null log-likelihood per character.
`gold_per_char` in `evaluate_completion.py` is a distinct earlier diagnostic
containing conditional log-likelihood only. Keep that name only when discussing
its diagnostic table; do not use it for the paper's headline three-model
values. Argmax accuracy remains secondary.

#### CQ-9: Which confidence intervals will accompany headline estimates?

**Status: Open analysis task.**

Add paired bootstrap 95% intervals for question and same-set chunk contrasts.
For the held-out/control difference-in-differences, resample at the chunk level
within each set. State clearly that these intervals do not include training-seed
uncertainty. Also choose one primary inferential convention: current question
summaries quote paired t-test p-values while also computing sign-flip tests;
chunk summaries use sign-flip tests and the difference-in-differences uses a
label permutation. Name the test beside every reported p-value, and never treat
a non-significant residual as evidence of equivalence without a pre-specified
equivalence margin.

#### CQ-10: Is the Ancient Rome zero-guard-leak claim verified?

**Status: Repository audit complete; needs archived-log check.**

The updated Rome model card confirms that the strict guard was enabled but
does not report a direct zero-leak tally. The code emits a guard-leak metric
only if an all-masked batch occurs. `ROME_RESULTS.md` verifies that 65,844
chunk IDs were loaded and 131,624 instance-slots were excluded (99.95% of the
two-epoch expectation); this is not a substitute for auditing the warnings or
metric records. Keep any claim of zero Rome guard leaks out of the paper until
the cluster logs are checked.

### P1: Experimental-design history

#### CQ-11: When and why was the final Baseball QID set fixed?

**Status: Answered from repository.**

This asks how Baseball was *labeled in the corpus*, not why Baseball was chosen
as a project topic. The final configuration documents a corpus-only audit on
2026-09-05 (jobs 852479/852602/852603): the 13-QID core achieved 94.0% estimated
precision and 43.1% coverage; adding 30 MLB teams produced the selected 43-QID
set at 87.2% precision and 56.5% coverage. Adding players gained only 0.1
percentage points of coverage while losing 2.0 points of precision, so it was
rejected. The configuration explicitly says this choice was made on the audit,
not the QA evaluation. The Experimental Design section now records this.

#### CQ-12: How should the Ancient Rome exploratory widening be described?

**Status: Draft framing implemented; paper group should confirm.**

Recommended wording: an initial one-QID intervention failed a corpus-coverage
audit, so the final 56-QID operationalization was developed through exploratory
corpus analysis before the reported matched comparison. State estimated recall
and precision and avoid calling it exhaustive. The Experimental Design and
Limitations sections now use this framing.

#### CQ-13: Archive final run-local configurations

**Status: Training match confirmed by the project team; archival remains open.**

The tracked configurations' `train` blocks are semantically identical. The
Rome template differs from control only by job name and exclusion settings;
Baseball also changes scheduling fields. The run summaries agree on model,
seed, optimizer recipe, corpus, batch size, duration, and final step. For
long-term provenance, archive `config.yaml` and `run_environment.json` for the
shared control, no-Rome, no-Baseball, and no-AI final runs, plus records of
their resumed checkpoint lineage. This is an artifact-collection task, not an
unresolved claim in the manuscript.

#### CQ-14: Which training runs changed accelerator type?

**Status: Partially answered; required framing identified.**

The newer Rome and AI cards report that the shared control ran on H200
throughout and Rome did likewise. Baseball ran its first four windows on H100
and final 2,832 steps on H200; AI had an A6000 window. The Baseball card's
older statement about a late control-GPU switch conflicts with those newer
records and appears to conflate this control with the retired Pornography
pair. Verify the control's run logs and correct the stale card. The drift
artifact behind the reported approximately 3e-5 cross-GPU effect is not
tracked. Do not numerically compare that drift with weight-space cosine until
their normalizations are verified.

### P1: Specificity, controls, and robustness

#### CQ-15: What is the final general-capability suite?

**Status: Open analysis decision.**

Choose a compact, common set from SciQ, ARC-Easy, HellaSwag, and PIQA and run it
on every final erased and excluded model. Report the same formulation and metric
for every condition. Do not infer broad preservation from one SciQ result.

#### CQ-16: How should Baseball's positive neighboring-sports shift be treated?

**Status: Answered from repository; do not interpret it as specificity.**

The excluded model improves unexpectedly on both small Simdom splits, but the
no-Rome twin---which excluded no Baseball data---shows almost the same shift on
the identical items (+0.1920/+0.2083 versus +0.2395/+0.2283). The same direct
control reproduces most of the apparent Baseball target-QA exclusion effect.
Because every arm shares one control, these are between-run/question-set
offsets rather than evidence of beneficial specificity. The paper now treats
the Baseball EMBER-minus-control question effect as real, but does not claim a
verified question-side erasure-versus-never-training ratio.

#### CQ-17: Will there be a robustness experiment?

**Status: Optional; subsection removed from the current draft.**

If yes, predefine one threat model (for example paraphrased prompts, relearning,
or prompt-based recovery), data, budget, and success measure, then apply it to
erased and concept-excluded models. The main-text subsection is currently
removed; the Limitations section states that robustness was not evaluated.

### P1: Figures, tables, and artifacts

#### CQ-18: Experimental-design figure

**Status: Optional figure not included in the manuscript.**

If space permits, add a vector diagram showing the shared
initialization/batch stream, control and three exclusion branches, post-hoc
method branches, and completed versus pending comparisons. It should explain
design, not contain result values.

#### CQ-19: Per-chunk distribution figure

**Status: Optional figure not included in the manuscript.**

Plot empirical CDFs or paired-shift distributions for Rome and Baseball EMBER
against exclusion; include MLP methods only on identical chunk IDs.

#### CQ-20: Main synthesis table

**Status: EMBER Rome/Baseball cells populated in the table fragment; table not
currently included in the eight-page main text. MLP common-protocol cells
pending.**

Fill each cell with the erased-minus-excluded residual and uncertainty, not only
erased-minus-control efficacy. Include behavioral, text-NLL, parameter, and
specificity columns. Keep the two concepts as separate rows.

#### CQ-21: Stable result provenance

**Status: Required before submission.**

Many summaries point to private cluster paths and gitignored JSONs. Copy the
final aggregate outputs, metadata, and configuration hashes into a stable,
versioned supplement. The paper must remain verifiable after cluster cleanup.

#### CQ-21a: Are EMBER's selected feature artifacts stably archived?

**Status: Needs artifact collection.**

The repository records the judge (`google/gemma-4-12B-it`), its revision for
the Rome run, the selected cells/features, and the fact that both erasures
reused cached factorizations. However, the chosen `feature_manifest.json`, raw
judge verdicts/traces, `potential_features.csv`, and Baseball's detailed
`GRID_NOTES.md` live on private cluster paths. Copy these artifacts into the
versioned supplement and record their hashes. This matters because the Rome
audit found that a fixed seed reproduced byte-identically across two RTX 3090
nodes but produced different features on an A6000.

### P2: Writing and submission details

#### CQ-22: Authors, affiliations, and exact title

**Status: Author details are present; confirm them with the group.**

The author block now lists Gal Barak, Tamar Tabbach, Itamar Stahl, and Adam
Fleisher, all affiliated with Tel Aviv University. Confirm the listed email
addresses and whether the course requires named or anonymous ACL mode. Revisit
the title only after the final comparative claim is known.

#### CQ-23: Citation audit

**Status: In progress.**

Verify all BibTeX metadata against primary pages. Add any course-required paper,
the exact source for the project definitions, and further related work needed to
support distinctions among unlearning, factual editing, and refusal tuning.
The Background paragraph currently makes these distinctions without dedicated
foundational citations.

#### CQ-24: AI disclosure

**Status: Ask every teammate.**

Collect tool/model names and versions, tasks assisted, verification practices,
one useful outcome, and one failure or limitation. Replace the current generic
draft without exposing secrets or copying private prompts unnecessarily.

#### CQ-25: Eight-page budget

**Status: Revisit after results stabilize.**

The current draft puts references on page 8 and has nine PDF pages including
the appendix. Recheck the eight-page main-text limit after inserting real
figures, tables, and final method details. Keep any deferred table fragment
visible in the repository until the final layout decision.
