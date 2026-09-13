# Paper Open Questions

Track unsettled facts here instead of guessing in the LaTeX. Status labels are
**answered from repository**, **partially answered**, **needs verification**,
**ask teammates**, and **pending experiment**.

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

**Status: Partially answered: tracked configurations match; run-local artifacts
are absent.**

The intended twins share the OLMo-2 1B architecture, initialization seed
(12536), data seed (0), optimizer, schedule, corpus order, batch size, and
duration. Because masking happens after batch composition, each arm traverses
the same batches and steps. One completed control is shared by Ancient Rome and
Baseball.

A semantic comparison of the tracked YAML files confirms that all three have
identical `train` blocks. Ancient Rome differs from control only in `job.name`
and `untaught`; Baseball additionally differs in scheduling fields such as
partition and time limit. The Rome and Baseball records also give the same
final step (54,832) and core training recipe. This resolves what the repository
specifies, but not whether every resumed job used precisely those files.

Evidence:

- `Untaught/configs/train_1b_control_2e_b131k.yaml`
- `Untaught/configs/train_1b_no_ancient_rome_core_2e_b131k.yaml`
- `Untaught/configs/train_1b_no_baseball_core_teams_2e_b131k_h100.yaml`
- `Untaught/model_cards/lment-1b-nobaseball-2e-b131k.README.md`, which explicitly
  identifies the shared control and identical batch composition.

Known qualification: final portions of some runs used different GPU types
(H100/H200). The Baseball model card documents this and reports small measured
cross-GPU drift. State this where parameter-space comparisons are discussed.

Not answerable from this checkout because no archived `config.yaml` or
`run_environment.json` is present:

- Compare archived `config.yaml` and `run_environment.json` from each final run,
  rather than relying only on repository templates.
- Confirm initial-checkpoint provenance across resumed windows.
- Record exact checkpoint IDs/steps and verify zero all-masked guard leaks for
  both concepts.

### ED-3: How were method hyperparameters selected?

**Status: EMBER partly answered; RMU and SNMF pending experiment/teammates.**

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

**Status: Needs code/result-artifact verification.**

Confirm that control, concept-excluded, EMBER, RMU, and SNMF use identical
examples, tokenization, prompts, scoring, and aggregation within each behavioral,
distributional, parameter-space, and specificity evaluation.

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

The LaTeX now contains a complete first-pass narrative. The questions below map
directly to its red `TODO` placeholders. Resolve the **P0** items before polishing
language; they can change the claims.

### P0: Results that can change the paper's conclusion

#### CQ-1: Did RMU produce a valid persistent checkpoint for both concepts?

**Owner: teammates running RMU. Status: Pending experiment.**

For each concept, record the final model path/hash, edited layer/tensor names,
forget and retain datasets, layer window, steering coefficient, retain weight,
learning rate, batch size, update count, seed, grid, and selection metric. Then
run the identical question, chunk-NLL, control, OLMES, and weight-space scripts.
An activation diagnostic without a saved and reloaded checkpoint is not enough.

#### CQ-2: Did SNMF produce a valid persistent checkpoint for both concepts?

**Owner: teammates running SNMF. Status: Pending experiment.**

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

**Owner: paper group. Status: Blocked on CQ-1--CQ-3.**

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

No tracked Rome model card or log excerpt states a zero guard-leak count.
`ROME_RESULTS.md` verifies that 65,844 chunk IDs were loaded and 131,624
instance-slots were excluded (99.95% of the two-epoch expectation), but that is
not equivalent to proving that the guard counter stayed zero. The paper must
retain its placeholder until the cluster logs are checked. Also archive the
final checkpoint step, blacklist hash, and per-window guard counts.

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

#### CQ-13: Are final run-local configurations truly matched?

**Status: Partially answered; tracked templates match, cluster artifacts absent.**

The tracked configurations' `train` blocks are semantically identical. The
Rome template differs from control only by job name and exclusion settings;
Baseball also changes scheduling fields. The run summaries agree on model,
seed, optimizer recipe, corpus, batch size, duration, and final step. Exact
run-local verification remains impossible from this checkout because it has no
archived `config.yaml` or `run_environment.json`. Diff those files for the shared
control, no-Rome, and no-Baseball models after removing scheduling-only fields.
Confirm initialization seed, data indices/order, optimizer state continuity,
schedule, duration, precision, and resume lineage.

#### CQ-14: How much should the H100/H200 difference affect claims?

**Status: Partially answered; required framing identified.**

The Baseball model card records that the control's final roughly 9.7% and the
Baseball twin's final 2,832 updates ran on H200 rather than H100, and reports a
measured cross-GPU drift of about 3e-5. The underlying drift artifact is not
tracked. Behavioral and NLL conclusions can therefore cite it as a small
qualification, but directional weight-space claims must note the asymmetry and
cannot present the displacement as free of hardware noise. A same-device
sensitivity analysis would strengthen this result; do not numerically compare
the reported drift and cosine until their normalizations are verified.

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

**Status: Placeholder present.**

Replace the box in Section 3 with a vector diagram showing the shared
initialization/batch stream, control and two exclusion branches, post-hoc method
branches, and common evaluation suite. It should explain design, not contain
result values.

#### CQ-19: Per-chunk distribution figure

**Status: Placeholder present.**

Plot empirical CDFs or paired-shift distributions for both concepts and every
completed method. Use identical axes where possible and mark zero, mean, and
median. This figure should make Ancient Rome's EMBER tail visually explicit.

#### CQ-20: Main synthesis table

**Status: Value-free placeholder present.**

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

**Status: Partially answered; one email and submission-mode decision remain.**

The author block now lists Gal Barak, Tamar Tabbach, Itamar Stahl, and Adam
Fleisher, all affiliated with Tel Aviv University. Confirm Itamar Stahl's email
address and whether the course requires named or anonymous ACL mode. Revisit
the title only after the final comparative claim is known.

#### CQ-23: Citation audit

**Status: In progress.**

Verify all BibTeX metadata against primary pages. Add any course-required paper,
the exact source for the project definitions, and further related work needed to
support distinctions among unlearning, factual editing, and refusal tuning.
The first Background paragraph now contains a specific citation placeholder for
these three claims.

#### CQ-24: AI disclosure

**Status: Ask every teammate.**

Collect tool/model names and versions, tasks assisted, verification practices,
one useful outcome, and one failure or limitation. Replace the current generic
draft without exposing secrets or copying private prompts unnecessarily.

#### CQ-25: Eight-page budget

**Status: Revisit after results stabilize.**

The current draft intentionally prioritizes completeness. After tables and
figures are real, compress repeated setup, move exhaustive configurations to the
appendix, and verify that content before references fits the course's eight-page
limit.
