> **Archived 2026-09-30.** This audit was run against the PDF build of
> 2026-09-24 and does not describe the current paper, which was substantially
> rewritten on the 27th and 29th. It is kept as the record that 24 attributions
> were checked against primary text.
>
> Status of what it raised:
>
> - **1.1 Arditi citation** and **1.2 RMU's scaling term** — fixed in Overleaf on
>   2026-09-30, not yet synced to git at the time of archiving.
> - **Section 4, Broken** — resolved. `score_mc.py`, `finalize_tables.py`,
>   `questions_with_options_TEST.csv` and `sciq_unrelated.csv` are all in `main`.
> - **Section 4, Mismatch** — resolved, and its premise was wrong. The paper's AI
>   checkpoint is delta 500 and is published as
>   `itamarstahl/lment-1b-ai-ember-d500-b131k`. The delta 5.0 card describes
>   EMBER's own delta-search pick, a separate comparison condition. See
>   `REPRODUCE.md`.
> - **Section 2** — judgment calls, left as written.

# Reference and citation audit

**Paper:** "Can Concept Erasure Reproduce Concept Exclusion? A Matched Evaluation of EMBER,
RMU, and SNMF"
**Audited:** 2026-09-24, against the PDF build of that date.
**Scope:** not "do the cited papers exist" — they all do — but whether each citation is
placed on the right clause, points at the work that actually supports it, and whether our
wording survives comparison with what the source really says. 24 attributions were checked
against primary text: equations, appendices and source files, not summaries. The paper's own
internal artifact references (Appendices A and B) were checked against this repo.

**Verdict: 2 items to fix, 3 minor, the rest sound.** No misattributed claim, no citation
pointing at the wrong work, no source that fails to support what it is cited for.

---

## 1. Fix these

### 1.1 Missing citation — Arditi et al. (2024)

We describe the SNMF baseline's erasure operation twice:

- §2.2 — "concept-associated features are mapped to directions that can be removed through
  MLP weight edits (Suslik et al., 2026)"
- §3.3 — "the SNMF-based method removed concept-associated directions through MLP weight
  edits"

That operation is **directional ablation**, `W ← W − (Wf̂)f̂ᵀ`. EMBER §4.2 credits it
explicitly — its SNMF variant "finds semi-nonnegative MF features (Shafran et al., 2026)
from MLP activations and applies **directional ablation (Arditi et al., 2024)** to MLP
weights across layers" — and §A.4 calls its own edit "the pure projection-out of Arditi
et al. (2024)". EMBER cites Arditi in §4.2 and twice in §A.4.

Attributing the *erasure adaptation* to Suslik et al. is correct; EMBER introduced that
baseline. But we reimplement the underlying operation and never name its source. Add:

> Arditi, A., Obeso, O., Syed, A., Paleka, D., Panickssery, N., Gurnee, W., and Nanda, N.
> 2024. Refusal in language models is mediated by a single direction.
> *Advances in Neural Information Processing Systems* 37, pp. 136037–136083.

### 1.2 RMU's mechanism is described without its scaling term

§2.2 — *"RMU fine-tunes selected transformer weights so that forget-set representations
approach a random direction, while a retain objective keeps other representations close to
those of the original model."*

RMU's forget loss is `E[ ‖M_updated(t) − c·u‖² ]`: `u` is a fixed random unit vector **and
`c` is a scaling hyperparameter**. The norm increase is not incidental, it is the stated
mechanism. WMDP §4: "increasing the norm of the model's activations on hazardous data in
earlier layers makes it difficult for later layers to process the activations." Their
Figure 7 caption describes the forget term as one "which changes direction **and scales up
the norm** of model activations."

"A random direction" drops the half of the mechanism WMDP says does the work. One-word fix:
"approach a **scaled** random direction", or "a fixed random vector with amplified norm".

Our own implementation description is already correct — §3.3's "updated MLP down-projection
weights in selected layers" matches RMU's actual implementation. Only the §2.2 sentence
needs the word.

---

## 2. Minor — judgment calls, all defensible as written

### 2.1 Concept-set provenance is undisclosed, and two sections sit oddly together

§3.2 — "We selected Ancient Rome, Baseball, and artificial intelligence (AI) **by hand**
before training the erasure models… Because this was a purposive choice, our findings don't
necessarily generalize to every concept."

§4.1 — "The target and neighboring questions **followed EMBER's existing splits**."

Both are true, and compatible — but only because all three concepts are among EMBER's 18.
That unstated fact is what makes EMBER's questions reusable. From EMBER §B:

- EMBER's first 11 concepts are adopted from **Gur-Arieh et al. (2025b)** and include
  **Ancient Rome** and **Baseball**.
- EMBER's 7 additional concepts come from **ConceptVectors (Hong et al., 2025)** and include
  **Artificial Intelligence**.
- EMBER then wrote its own questions for all 18 from scraped Wikipedia pages.

So the questions and splits are genuinely EMBER's (§4.1 is correct), while the concept names
trace back through EMBER to two earlier datasets. Suggest one clause in §3.2 — "by hand from
EMBER's 18 concepts" — which removes the apparent tension and strengthens §4.1. Optionally
credit Gur-Arieh et al. (2025b) and Hong et al. (2025) as the concept-set origin.

### 2.2 The intro's three-way citation compresses one attribution

Intro — "we apply EMBER, RMU, and an SNMF-based MLP erasure to copies of the full model
(Suslik et al., 2026; Li et al., 2024; Shafran et al., 2026)."

Citation order matches method order exactly, which is good practice. But Shafran et al. is
an *interpretability* paper: it introduces SNMF for feature discovery and causal steering
and proposes no erasure method. The erasure adaptation is EMBER's. Suslik is first in the
list so the triple is covered, and §2.2 gets this exactly right. Flagged only because a
reviewer skimming the intro alone could read it as crediting Shafran with an erasure method.

### 2.3 A claim about the EMBER study cited to other papers

§2.2 — "It evaluated EMBER alongside RMU and an MLP weight-erasure adaptation of
SNMF-derived features (Li et al., 2024; Shafran et al., 2026)."

The subject is the EMBER study, so a strict reader wants (Suslik et al., 2026) on the claim
with Li/Shafran as method identifiers. The two preceding sentences both cite Suslik, so
context carries it. Note also that EMBER's full baseline set is RMU, CRISP, PISCES and SNMF,
each tested alone and ensembled with EMBER; "alongside" is not a completeness claim, so the
sentence is accurate, just selective.

---

## 3. Confirmed sound — checked adversarially, nothing to change

These were checked precisely because they were the ones most likely to break.

**EMBER mechanism (§2.2, §3.3)**

- "edits token embeddings rather than internal transformer layers" ✅
- "subtracts their scaled contributions from the embeddings of tokens that express those
  factors" ✅ — EMBER: "Features related to the target concept are then subtracted from the
  embeddings of those tokens, removing the concept-related component while leaving the rest
  of each embedding intact." Near-verbatim faithful; "scaled" corresponds to EMBER's δ.
- **"changes a selected set of input representations without updating the rest of the
  model"** ✅ — this could easily have been wrong. EMBER's Limitations, *Editing the input
  embedding only*: "We use EMBER to edit only the input embedding matrix, and **not the
  unembedding matrix**." Our §3.3 "EMBER edited selected input embeddings" is precisely right.
- *Corroboration:* EMBER flags a caveat for **tied** embeddings (Gemma). OLMo-2 sets
  `tie_word_embeddings: False`, and our own erased model cards state "the input embedding
  only — `lm_head.weight` is a separate, untied tensor and is carried across untouched."
  EMBER: "Most modern LLMs use untied embeddings, where this concern does not apply." The
  port is clean on this axis — worth a footnote if a reviewer raises it.

**Inherited scoring protocol (§3.3) — the most load-bearing citation in the paper**

- "chance-corrected accuracy normalization" ✅ — EMBER Eq. 22:
  `(Acc(M′) − 0.25)/(Acc(M) − 0.25)` for multiple-choice, with the raw ratio (Eq. 21)
  reserved for open-ended. Our Eq. (1) is the same formula with F in place of M.
- "nested harmonic aggregation" ✅ — EMBER Eq. 23: `H = HM(φ_eff, φ_spec, φ_coh)` where
  `φ_spec = HM(Sim, MMLU)` and `φ_coh = HM(Ins, Flu)`. Genuinely a harmonic mean of harmonic
  means. Our `H_selection = HM(φ_eff, HM(neighbor, SciQ))` is a faithful two-axis reduction.
- `φ_eff = 1 − Acc~_target` ✅ identical to EMBER's `φ_efficacy = 1 − Acc~_C`.
- "adapts its validation score to our **base-model** setting… we omit the AlpacaEval
  coherence component" ✅ — and the contrast is real: EMBER evaluates **Gemma-2-2B-it** and
  **Llama-3.1-8B-Instruct**, both instruction-tuned. Dropping AlpacaEval for a 1B base model
  is correctly motivated, not hand-waved.
- "followed EMBER's existing splits" ✅ — EMBER §B: "Concept and Similar-Domain MC/OE
  questions: 50 validation / 50 test per concept", matching our 50 selection / 50 test.
- "the study did not compare edits with a matched model trained without loss on
  concept-associated data" ✅ — negative claims are the riskiest kind; this one is safe,
  since EMBER edits off-the-shelf instruct models and retrains nothing.

**RMU / WMDP** — "measured suppression on WMDP alongside retention on MMLU and MT-Bench
relative to the original model" ✅ verbatim-level match with WMDP's own summary. "a retain
objective keeps other representations close to those of the original model" ✅.
"fine-tunes selected transformer weights" ✅.

**SNMF** — "identifies interpretable features in MLP activations from groups of co-activated
neurons" ✅ — SNMF §2: "we apply SNMF to MLP activations to recover features defined by
groups of co-activated neurons." Near-verbatim.

**Controlled-pretraining pair** — "compared models whose corpora differ through targeted data
insertion or filtering (Wei et al., 2025; O'Brien et al., 2025)" ✅, and the two citations are
in the same order as the two mechanisms. Hubble: "standard models are pretrained on a large
English corpus, and perturbed models are trained in the same way but with controlled
insertion of text." Deep Ignorance: pretraining-data filtering. Both are pretraining studies.

**SciQ** — validation split is 1,000 items (so sampling 100 is fine); each item is
`question` + `correct_answer` + `distractor1–3` = **four options** ✅, matching "retained
their original question text and four answer options"; domains are physics / chemistry /
biology and other sciences ✅, matching "general science".

**LMEnt** — "Wikipedia-based pretraining corpus annotated with entity mentions, an index for
retrieving chunks…, and language models trained on the corpus" ✅ exactly LMEnt's three
contributions. Two-epoch training ✅ (LMEnt releases 1/2/4/6-epoch variants). "OLMo-2 1B
models (Team OLMo et al., 2025; Gottesman et al., 2026)" ✅ — LMEnt trains 170M/600M/1B
"based on the OLMo-2 architecture", citing the OLMo paper for exactly that; the dual citation
mirrors LMEnt's own usage. The OLMo paper itself describes 7B/13B/32B and no 1B, which is
correct as architecture attribution — **do not "fix" this into an error**.

**Intro framing** — "Post-training concept erasure aims to remove a model's learned
association with a target concept while preserving its behavior elsewhere (Li et al., 2024;
Suslik et al., 2026)" ✅. WMDP's framing is "unlearning", not "concept erasure", so this
looked like a stretch — but EMBER itself lists Li et al. (2024) among "the common problem
setup of concept erasure". Using it this way follows the cited literature.

**Completeness** — 8 entries, 8 works cited, every citation resolves, no orphans. Counts:
Suslik ×7, Li ×5, Shafran ×3, Gottesman ×2, Welbl ×2, O'Brien ×1, Wei ×1, Team OLMo ×1.

---

## 4. Internal artifact references (Appendices A and B)

The paper cites its own code, data and checkpoints as reproducibility artifacts. Those are
citations too. **The scientific artifacts all verify; the code paths do not.**

### Verified ✅

- **Table 1 reconciles perfectly with the blacklist files.** Entity-ID counts
  (Rome 56 / Baseball 43 / AI 31) equal `len(entities)` in
  `Untaught/blacklists/{ancient_rome_core,baseball_core_teams,ai_core}.json`, and the
  excluded-chunk counts (65,844 / 80,466 / 19,818) match each file's own header comment. The
  derived shares (0.628% / 0.767% / 0.189%) are consistent against the 10,491,928-chunk
  corpus, which matches LMEnt's reported ~10.5M-chunk index. *(A regex finds 59 and 32 QIDs
  in the Rome and AI files, but the extras appear only in prose describing deliberately
  excluded entities — Latin Q397, Byzantine Q12544, Holy Roman Empire Q12548 — and one
  mis-resolution note. The operative counts are right.)*
- All three blacklist paths cited in Appendix A exist at HEAD.
- All four model identifiers in Table 5 resolve to cards in `Untaught/model_cards/`.
- The NLL/KL scripts referenced in Appendix A (`ember_eval/nll_kl/score_model.py`,
  `compare.py`, `finalize.py`) exist at HEAD.

### Broken ❌

- `ember_eval/acc_selection/score_mc.py` and selection revision **`92a39d3`** (Appendix A):
  the revision is real, but it sits on `origin/eval/accuracy-selection` and is **not an
  ancestor of HEAD**. The directory does not exist in the working tree.
- `ember_eval/acc_selection/finalize_tables.py` (Appendix A): exists only at that branch's
  **tip** (`bfa7f54`), not at the cited revision `92a39d3`.
- `questions_with_options_TEST.csv` and `sciq_unrelated.csv` (Appendix B): do not exist under
  those names anywhere in the repo, including the branch tip. Only
  `questions_with_options.csv` is present.

A reader following the reproducibility appendix today finds nothing. Fix: merge or
cherry-pick the selection pipeline onto the publication branch, pin a revision reachable from
it, and correct the two manifest filenames.

### Mismatch ⚠️

Table 7's selected EMBER checkpoints, against the released model cards:

| Table 7 | Model card | δ | Match |
|---|---|---|---|
| `ember_rome_d200` | `lment-1b-rome-erased-b131k` | 200 | ✅ |
| `ember_baseball_d10` | `lment-1b-baseball-erased-b131k` | 10 | ✅ |
| `ember_ai_d500` | `lment-1b-ai-erased-b131k` | **5.0** | ❌ |

The naming convention is unambiguous — Rome ships separate `d50` and `d500` cards — so `d500`
means δ=500, and the only released AI erasure is δ=5.0. Either Table 7's AI entry is a typo
for `ember_ai_d5`, or the checkpoint behind the reported AI numbers was never released.
Resolve before publication: it determines whether the AI column is reproducible.

---

## 5. Internal consistency (outside the brief, checked while the tables were open)

All of §5.1's numbers reconcile with Appendix C, **including signs**: Rome 0.42→0.26 /
+1.016 / KL 0.955; Baseball 0.54→0.50 / +0.377; AI 0.54→0.48 / +0.595; neighboring
+0.171 / +0.093 / −0.139; SciQ +0.046 / −0.093 / +0.145.

The four headline claims are arithmetically correct against the appendix:

- "RMU most closely matches the twins' signed mean target NLL" — RMU has the smallest
  |ΔNLL_{M−T}| on target in all three concepts (−0.303 / +0.247 / −0.013) ✅
- "SNMF yields the lowest target KL" — 0.851 / 0.680 / 0.588, lowest of the three methods in
  all three concepts ✅
- "EMBER produces the strongest target-specific NLL changes" — +2.402 / +1.294 / +2.220,
  largest in all three ✅
- "the highest-scoring method is closest in mean target NLL for only one concept and never
  has the lowest target KL" — closest only for Rome (RMU); the top scorer's target KL is
  1.143 / 1.472 / 1.964 against SNMF's 0.851 / 0.680 / 0.588 ✅

Still open: the **AI Disclosure** section is an unfilled placeholder.

---

## 6. Bibliographic layer (all exact)

| Reference | Verdict |
|---|---|
| Gottesman et al. 2026, LMEnt, TACL **14:1685–1722** | ✅ aclanthology.org/2026.tacl-1.76, DOI 10.1162/tacl.a.746, 7 authors |
| Li et al. 2024, WMDP, PMLR **235:28525–28550** | ✅ proceedings.mlr.press/v235/li24bc.html |
| O'Brien et al. 2025, Deep Ignorance, arXiv:2508.06601 | ✅ 2025-08-08, 10 authors in order |
| Shafran, Geiger, Geva 2026, ACL **42326–42348**, San Diego | ✅ incl. byline "Or **David** Shafran" |
| Suslik, Shafran, Geva 2026, EMBER, arXiv:2606.03695 | ✅ 2026-06-02 |
| Team OLMo et al. 2025, arXiv:2501.00656 | ✅ |
| Wei et al. 2025, Hubble, arXiv:2510.19811 | ✅ 2025-10-22, 10 authors in order |
| Welbl, Liu, Gardner 2017, SciQ, W-NUT **94–106** | ✅ aclanthology.org/W17-4413 |

Cosmetic only: "Or Shafran" (entry 5, arXiv byline) vs "Or David Shafran" (entry 4, ACL
byline) is the same person; each entry reproduces its own source, so neither is wrong.
Sentence-cased titles in entries 6–7 are ACL style.

---

## 7. Method and sources

Full text read, not summarized: EMBER `arxiv.org/html/2606.03695v1` (§4.2, §B, §C.1
Eq. 21–23, §C.2, Limitations, §A.4); WMDP `arxiv.org/html/2403.03218v3` (§4, Fig. 7);
LMEnt `arxiv.org/html/2509.03405v1` (§4); SNMF `arxiv.org/html/2506.10920v2` (abstract, §2).
Metadata: aclanthology.org 2026.tacl-1.76 / 2026.acl-long.1959 / W17-4413;
proceedings.mlr.press/v235/li24bc.html; arxiv.org/abs/2508.06601, /2501.00656, /2510.19811;
huggingface.co/datasets/allenai/sciq; huggingface.co/allenai/OLMo-2-0425-1B `config.json`.

**One caveat for anyone repeating this.** Automated page-summarization was wrong on the two
most load-bearing claims in the paper: it reported that EMBER's normalization was *not*
chance-corrected and that its harmonic mean was *flat* — i.e. that both claims in §3 above
were false. Reading EMBER's Eq. 21–23 directly showed the opposite. A summarizer also missed
that EMBER's concept list contains all three of our concepts. Single-pass summaries of these
papers are unreliable on formula- and list-level detail; every finding here came from the
primary text.
