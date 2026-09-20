# Planned evaluation paper: fill-in version

This is a separate, prospective paper based on the team's updated evaluation
plan. It assumes that the shared full model, a concept-excluded twin, and
EMBER, RMU, and SNMF checkpoints will all be scored on the **same held-out
sentence completions** for Ancient Rome, Baseball, and AI. It does not claim
that every comparison has been completed. Replace the blue planning notes and
empty exhibits as results become available. The existing `paper/main.tex` is
untouched.

Build from this folder with `latexmk -pdf main.tex`. The three symlinks reuse
the parent paper's ACL style and bibliography. Main-paper target: **at most
eight pages**, excluding references and appendices. The lengths below are
writing targets, not a claim about the current placeholder PDF.

| Part | Target length | Reader should learn | Planned exhibit |
|---|---:|---|---|
| Title and abstract | 0.5 page | Question, twin benchmark, three concepts, five models, two answer-token metrics, central findings once known | None |
| 1 Introduction | 0.7 page | What erasure is being tested; why a concept-excluded twin is the reference; scope and contributions | Fig. 1: compact study-design schematic |
| 2 Background and related work | 0.5 page | What EMBER, RMU, SNMF change; how exclusion differs; relevant evaluation precedents | None |
| 3 Experimental design | 1.1 pages | Shared control, three twins, exclusion scope, erasure setup, held-out completion groups | Table 1: concepts, exclusion footprints, and completion counts |
| 4 Evaluation protocol | 1.0 page | Fixed correct continuations; answer NLL difference; full-vocabulary KL; model references and uncertainty | One stem/answer example and equations |
| 5 Results | 2.2 pages | Twin effects; target resemblance; neighboring/unrelated preservation; agreement and disagreement between NLL and KL | Table 2: compact target comparison; Fig. 2: three-group answer-NLL changes; optional KL panel |
| 6 Analysis and discussion | 0.8 page | How to interpret similarity to the twin and preservation; per-item cancellation and metric disagreements | A small paired-change plot only if it explains a central finding |
| 7 Limitations | 0.4 page | Blacklist coverage, fixed-answer scope, one training seed, finite questions, teacher-forced KL | None |
| 8 Conclusion | 0.2 page | Answer the research question at the level actually supported | None |
| AI disclosure and reflection | 0.2 page | Name actual tools/models and their role, per course instructions | None |
| Appendix A–F | About 3–5 pages as needed | Exact protocol, full tables, configurations, diagnostics, reproducibility | Tables A1–A5; optional diagnostic plots |

Subsection budgets within those totals (including nearby exhibit space where
applicable):

| Section | Subsection targets |
|---|---|
| Background | Exclusion reference 0.2 page; erasure methods 0.3 |
| Experimental design | Shared model/twins 0.3; concept scope 0.25; erasure checkpoints/selection 0.35; evaluation separation 0.2 |
| Evaluation protocol | Three groups and fixed answers 0.2; answer NLL 0.25; teacher-forced KL 0.25; comparisons/uncertainty 0.3 |
| Results | Exclusion reference 0.4; target match 0.7; neighboring/unrelated preservation 0.6; NLL/KL agreement 0.3; about 0.2 for exhibits |
| Discussion | Criteria for matching exclusion 0.35; concept/metric disagreements 0.45 |
| Appendix | A configurations 0.5; B evaluation sets 0.5; C scoring/statistics 0.7; D complete results 1.5–2.5; E diagnostics 0.5; F reproducibility 0.3 pages, adjusted to actual material |

## Main-paper exhibits

- **Figure 1, study design:** the full model branches into three
  concept-excluded twins; EMBER, RMU, and SNMF edit the full model for each
  concept. All five models receive identical target, neighboring, and
  unrelated sentence completions. Draw it only if one column is clearer than
  prose.
- **Table 1, study inventory:** one row per concept, with blacklist definition
  and size and completion counts in each of the three groups. Exact QID lists
  and question IDs belong in Appendix A/B.
- **Table 2, target results:** one row per concept and erasure method. Show
  target answer-NLL difference from full, answer-NLL difference from twin, and
  KL(twin → method), with paired intervals. Include or display nearby the
  twin-minus-full NLL and KL(full → twin) reference effects. Do not call a
  near-zero mean NLL difference equivalence without examining its uncertainty
  and per-question distribution.
- **Figure 2, preservation:** faceted by concept, show signed answer-NLL
  differences from full for twin, EMBER, RMU, and SNMF across target,
  neighboring, and unrelated groups. Add paired intervals. A corresponding KL
  panel can be added if it clarifies differences; KL measures distributional
  change, not better or worse factual prediction.

No stock images are needed. Figures should be drawn from the actual protocol
and results when available. Keep essential results and interpretation in the
main paper; put complete matrices in the appendix.

## Results to fill, in order

1. Inventory exact held-out question counts and verify every checkpoint uses
   the same prompts, fixed answers, tokenization, and context settings.
2. Establish each twin's signed answer-NLL change from full and
   KL(full → twin) in all three groups. This shows what the reference
   intervention measurably changed before ranking erasure methods.
3. Fill target NLL and KL for EMBER, RMU, and SNMF relative to **both** full
   and twin. A near-zero *average* signed NLL residual can hide canceling
   question-level changes; report intervals and inspect their distribution.
4. Fill neighboring and unrelated results separately. The neighboring group
   is where semantically related spillover may appear; it is not
   interchangeable with the unrelated group.
5. Examine where answer NLL and KL disagree. A low KL means similar
   distributions; a positive NLL difference from full means lower probability
   on correct answer tokens. Neither metric measures generated-answer accuracy.
6. Interpret concept differences after checking exclusion footprint, question
   construction, and any limits of this answer-only instrument.

## Metric conventions fixed by the updated plan

- Each item is a sentence-completion prompt with **one fixed correct
  continuation**. No options are ranked and no answer is generated.
- Per-question answer NLL is the mean negative log-probability of the correct
  answer tokens, conditioned on the prompt and preceding correct tokens. The
  signed group mean difference is **evaluated − reference**. Positive means
  less probability assigned to the correct answer on average.
- Full-answer KL compares the **entire next-token vocabulary distribution** at
  each correct-answer position, using identical teacher-forced prefixes. Its
  direction is **reference → evaluated**; average positions within a question,
  then average questions. It is not a divergence between generated answers.
- Calculate both metrics separately for target, neighboring, and unrelated
  groups. Compare twin against full as reference, and each erasure against
  both twin and full. Preserve the NLL sign and KL direction above.
- Use paired bootstrap confidence intervals over questions, clustering any
  paraphrases of the same fact. A near-zero signed mean does not establish
  identical per-question behavior or an equivalence claim.
- This protocol evaluates correct-answer probability and prediction-distribution
  similarity on factual sentence completions. It does **not** measure
  generated-answer accuracy or general passage-prediction ability.

## Appendix plan

- **A. Corpus exclusion and checkpoints:** QID rules, blacklist audits,
  shared training recipe, checkpoint identifiers, erasure settings and
  selection sets (Table A1).
- **B. Evaluation sets:** per-group question counts, inclusion/exclusion
  rules, representative prompt/answer, question provenance and overlap checks
  (Table A2).
- **C. Scoring and inference:** exact answer-token NLL and teacher-forced KL
  equations, reference direction, tokenization/context, aggregation,
  bootstrap and paraphrase-clustering details, and a worked item.
- **D. Full results:** separate Ancient Rome, Baseball, and AI tables with
  five models × three subject groups × two metrics, underlying group means,
  both full/twin contrasts and intervals (Tables A3–A5). Split wide tables
  for legibility.
- **E. Diagnostics:** per-question signed NLL differences, cancellation,
  NLL/KL disagreement and any supported case studies. Include only analyses
  actually run.
- **F. Reproducibility:** final scoring code, model IDs, seeds, exact question
  manifest and split chronology, context settings, and deviations from the
  common protocol. Do not paste raw cluster logs.

The older manuscript uses gold-option PMI per character and option accuracy.
Those values cannot be copied into this plan's answer-NLL or KL tables. They
must be recomputed under the updated protocol or explicitly labeled as
separate exploratory results.
