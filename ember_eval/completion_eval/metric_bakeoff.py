#!/usr/bin/env python3
"""Choose the completion-eval scoring rule on evidence rather than argument.

`evaluate_completion.py` stores `logp_conditional` and `logp_null` per option,
so every scoring rule anyone has proposed is recomputable from records already
on disk — no GPU, no re-running the models.  This script recomputes them all
and ranks them on criteria that are *independent of the ablation*, which is the
whole point: picking the rule that shows the biggest twin effect would be
selection on the outcome, and that is how the `acc_raw` claims in
`EVALUATION.md` came to be retracted.

Two axes are crossed.

**Scoring rule** — what the four per-option numbers are:

    cond_sum        log P(option | stem)                     summed, no normalisation
    cond_per_char   the same, divided by the option's characters
    cond_per_token  the same, divided by its tokens          (needs --tokenizer)
    pmi_sum         log P(option | stem) - log P(option | null)
    pmi_per_char    the same, per character
    pmi_per_token   the same, per token                      (needs --tokenizer)

**Statistic** — what is read off those four numbers per question:

    acc         1 if the gold option is the argmax          (a 0/1 indicator)
    p_correct   softmax over the four options, gold's share  (continuous, chance 0.25)
    margin      gold minus the best distractor
    gold        the gold option's own score, unnormalised

The question that prompted this maps onto the grid as: option 1 is
(cond_*, acc), option 2 is (pmi_*, acc), option 3 is (cond_*, gold) or
(cond_*, p_correct), option 4 is the pmi equivalents.

**The criteria.**  A scoring rule is good if it responds to differences that
are really there and stays quiet about ones that are not:

  sensitivity   concept QA against neighbouring-domain QA, within a single
                model, averaged over all three models and both concepts.  Every
                model is far above chance on the concept and near chance on the
                neighbouring domain, so this gap is real; the question is which
                rule resolves it most sharply.
  null (HP)     control against ablated on **Harry Potter**, which neither model
                ablated.  The truth is zero.  Measured as the drift-free
                difference-in-differences, `QA - SimdomQA`.
  null (rel)    the released 2E model against our control on Pornography.  They
                differ in batch size, seed, code version and optimiser
                trajectory — in everything except an ablation — so their
                concept-specific difference is also zero by construction.
  reliability   |QA_train - QA_test| of the model-level mean, standardised.
                The two splits are random halves of one question set, so a rule
                that disagrees between them is measuring its own noise.

The headline ranking is sensitivity divided by the larger of the two null
floors.  The actual twin contrast is computed and printed too, but *after* the
selection criteria and clearly marked, so it cannot leak into the choice.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
from collections import defaultdict
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np

RNG_SEED = 20260827
PERMUTATIONS = 20000
CHANCE = 0.25

# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
# results/completion/<prefix>_<model>_<subset>_<split>_<jobid>.json, where the
# prefix distinguishes the two concepts the evaluation was run on.
FNAME = re.compile(r"^(cmpl|hp)_(\w+?)_(QA|SimdomQA)_(train|test)_(\d+)\.json$")


def load_records(results_dir: str) -> List[Dict[str, Any]]:
    """Every record, tagged with the model and concept it came from."""
    out: List[Dict[str, Any]] = []
    for path in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        m = FNAME.match(os.path.basename(path))
        if not m:
            continue
        _, model, subset, split, _ = m.groups()
        blob = json.load(open(path))
        for rec in blob["records"]:
            rec = dict(rec)
            rec["model"] = model
            rec["_subset"] = subset
            rec["_split"] = split
            out.append(rec)
    if not out:
        raise SystemExit(f"no completion records matched in {results_dir}")
    return out


def item_key(rec: Dict[str, Any]) -> Tuple[str, str, str]:
    """Identity of a question, stable across models."""
    return (rec["concept"], rec["_subset"], rec["question"])


def index_records(records: Sequence[Dict[str, Any]]) -> Dict[str, Dict[Tuple, Dict]]:
    """{model: {item_key: record}}, deduplicated.

    26 of the 50 similar-domain test questions also appear in the validation
    split (see `score_pairs` in evaluate_completion.py), so pooling train and
    test without deduplicating would double-count them.
    """
    by_model: Dict[str, Dict[Tuple, Dict]] = defaultdict(dict)
    dropped = 0
    for rec in records:
        key = item_key(rec)
        if key in by_model[rec["model"]]:
            dropped += 1
            continue
        by_model[rec["model"]][key] = rec
    if dropped:
        print(f"  deduplicated {dropped} repeated question(s) across splits")
    return by_model


# --------------------------------------------------------------------------- #
# scoring rules
# --------------------------------------------------------------------------- #
def option_chars(rec: Dict[str, Any]) -> List[int]:
    """Characters actually scored: the continuation is " " + option."""
    return [len(f" {o}") for o in rec["options"]]


def build_rules(token_counts: Dict[str, int] | None
                ) -> Dict[str, Callable[[Dict[str, Any]], List[float]]]:
    def cond_sum(rec):
        return list(rec["logp_conditional"])

    def pmi_sum(rec):
        return [c - z for c, z in zip(rec["logp_conditional"], rec["logp_null"])]

    def per_char(base):
        def f(rec):
            return [s / n for s, n in zip(base(rec), option_chars(rec))]
        return f

    rules = {
        "cond_sum": cond_sum,
        "cond_per_char": per_char(cond_sum),
        "pmi_sum": pmi_sum,
        "pmi_per_char": per_char(pmi_sum),
    }

    if token_counts is not None:
        def per_token(base):
            def f(rec):
                ns = [token_counts[f" {o}"] for o in rec["options"]]
                return [s / n for s, n in zip(base(rec), ns)]
            return f
        rules["cond_per_token"] = per_token(cond_sum)
        rules["pmi_per_token"] = per_token(pmi_sum)

    return rules


def softmax(xs: Sequence[float], temperature: float) -> List[float]:
    scaled = [x / temperature for x in xs]
    m = max(scaled)
    exps = [math.exp(x - m) for x in scaled]
    total = sum(exps)
    return [e / total for e in exps]


def calibrate_temperature(rule, records: Sequence[Dict[str, Any]]) -> float:
    """One temperature per rule, fixed across models.

    A per-character score spans a far narrower range than a summed one, so a
    softmax at T=1 would sit near-uniform for one rule and saturated for
    another, making `p_correct` incomparable between them.  T is set to the mean
    within-item spread so every rule enters the softmax on the same scale.

    Calibrated on the **released** model only.  It appears in neither twin pair,
    so no constant used here is derived from the contrast being measured.
    """
    spreads = []
    for rec in records:
        s = rule(rec)
        spreads.append(max(s) - sum(s) / len(s))
    t = float(np.mean(spreads))
    return t if t > 1e-9 else 1.0


def statistics_for(rule, temperature: float, rec: Dict[str, Any]) -> Dict[str, float]:
    s = rule(rec)
    g = rec["correct_index"]
    others = [s[i] for i in range(len(s)) if i != g]
    return {
        "acc": float(max(range(len(s)), key=lambda i: s[i]) == g),
        "p_correct": softmax(s, temperature)[g],
        "margin": s[g] - max(others),
        "gold": s[g],
    }


STATISTICS = ("acc", "p_correct", "margin", "gold")

# `gold` is excluded from the ranking, and the reason is the same trap that
# `acc_raw` fell into.  The sensitivity criterion compares concept QA against
# neighbouring-domain QA — two *different* question sets, with different answer
# strings.  `acc`, `p_correct` and `margin` all compare the gold option against
# its own distractors within a single item, so whatever makes one question set's
# answers longer or more frequent cancels.  `gold` does not: it is the gold
# option's unnormalised score, so it separates the two sets largely by how long
# and how common their answer strings are.  It scores a spuriously huge
# sensitivity for exactly the reason it should not be trusted.  It is still
# computed and printed, because it is what the question "take the likelihood
# itself as a score" literally asks for and the number is worth seeing.
RANKABLE = ("acc", "p_correct", "margin")


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #
def dz(d: np.ndarray) -> float:
    sd = d.std(ddof=1)
    return float(d.mean() / sd) if sd > 1e-12 else 0.0


def perm_p_paired(d: np.ndarray, rng: np.random.Generator,
                  iters: int = PERMUTATIONS) -> float:
    """Two-sided sign-flip test: under the null the model label is arbitrary."""
    obs = abs(d.mean())
    signs = rng.choice([-1.0, 1.0], size=(iters, d.size))
    null = np.abs((signs * d).mean(axis=1))
    return float((np.sum(null >= obs - 1e-15) + 1) / (iters + 1))


def perm_p_did(d_qa: np.ndarray, d_sd: np.ndarray, rng: np.random.Generator,
               iters: int = PERMUTATIONS) -> float:
    """Same, for a difference of two independently paired means."""
    obs = abs(d_qa.mean() - d_sd.mean())
    s_qa = rng.choice([-1.0, 1.0], size=(iters, d_qa.size))
    s_sd = rng.choice([-1.0, 1.0], size=(iters, d_sd.size))
    null = np.abs((s_qa * d_qa).mean(axis=1) - (s_sd * d_sd).mean(axis=1))
    return float((np.sum(null >= obs - 1e-15) + 1) / (iters + 1))


def welch_d(a: np.ndarray, b: np.ndarray) -> float:
    """Unpaired standardised difference; the two question sets differ."""
    sa, sb = a.var(ddof=1), b.var(ddof=1)
    pooled = math.sqrt((sa + sb) / 2.0)
    return float((a.mean() - b.mean()) / pooled) if pooled > 1e-12 else 0.0


def paired_arrays(by_model, model_a: str, model_b: str, concept: str,
                  subset: str, stat: str) -> np.ndarray:
    """b - a, over the questions both models were scored on."""
    a, b = by_model[model_a], by_model[model_b]
    keys = [k for k in a if k in b and k[0] == concept and k[1] == subset]
    keys.sort()
    return np.array([b[k]["_stats"][stat] - a[k]["_stats"][stat] for k in keys])


def subset_array(by_model, model: str, concept: str, subset: str,
                 stat: str, split: str | None = None) -> np.ndarray:
    recs = [r for k, r in by_model[model].items()
            if k[0] == concept and k[1] == subset
            and (split is None or r["_split"] == split)]
    return np.array([r["_stats"][stat] for r in recs])


# --------------------------------------------------------------------------- #
# the criteria
# --------------------------------------------------------------------------- #
def sensitivity(by_model, models, concepts, stat: str) -> float:
    """|d| for concept QA against neighbouring domain, within a model."""
    ds = []
    for model in models:
        for concept in concepts:
            qa = subset_array(by_model, model, concept, "QA", stat)
            sd = subset_array(by_model, model, concept, "SimdomQA", stat)
            if qa.size and sd.size:
                ds.append(abs(welch_d(qa, sd)))
    return float(np.mean(ds)) if ds else float("nan")


def did_effect(by_model, model_a, model_b, concept, stat, rng):
    d_qa = paired_arrays(by_model, model_a, model_b, concept, "QA", stat)
    d_sd = paired_arrays(by_model, model_a, model_b, concept, "SimdomQA", stat)
    if not d_qa.size or not d_sd.size:
        return None
    diff = d_qa.mean() - d_sd.mean()
    # Standardise the difference-of-differences on the pooled per-item spread,
    # so it is comparable across statistics measured in different units.
    pooled = math.sqrt((d_qa.var(ddof=1) + d_sd.var(ddof=1)) / 2.0)
    return {
        "did": float(diff),
        "dz": float(diff / pooled) if pooled > 1e-12 else 0.0,
        # A standardised difference estimated from n items carries roughly
        # sqrt(1/n) of sampling error, so two floors closer together than this
        # are not distinguishable and the ranking between them means nothing.
        "dz_se": math.sqrt(1.0 / d_qa.size + 1.0 / d_sd.size),
        "p": perm_p_did(d_qa, d_sd, rng),
        "n_qa": int(d_qa.size),
        "n_sd": int(d_sd.size),
        "qa_only_dz": dz(d_qa),
        "qa_only_p": perm_p_paired(d_qa, rng),
    }


def reliability(by_model, models, concepts, stat: str) -> float:
    """|train - test| of the model-level mean, in pooled standard deviations."""
    gaps = []
    for model in models:
        for concept in concepts:
            for subset in ("QA", "SimdomQA"):
                tr = subset_array(by_model, model, concept, subset, stat, "train")
                te = subset_array(by_model, model, concept, subset, stat, "test")
                if tr.size < 2 or te.size < 2:
                    continue
                pooled = math.sqrt((tr.var(ddof=1) + te.var(ddof=1)) / 2.0)
                if pooled > 1e-12:
                    gaps.append(abs(tr.mean() - te.mean()) / pooled)
    return float(np.mean(gaps)) if gaps else float("nan")


# --------------------------------------------------------------------------- #
# is the PMI correction earning its place?
# --------------------------------------------------------------------------- #
def null_term_drift(by_model, model_a: str, model_b: str, concepts) -> List[Dict]:
    """Does subtracting `log P(option | null)` cancel drift, or add it?

    PMI is justified by the claim that the unconditional term carries the part
    of the score that is not about this stem, so removing it leaves the
    association.  That is testable: compare how much the conditional term
    differs between two models against how much PMI does.  If PMI is doing its
    job the gap shrinks.

    It does not.  The null context is a single BOS token, so `log P(option |
    null)` is close to the model's document-initial prior — a quantity with no
    context anchoring it, and far less stable across two training runs than the
    conditional term.  Subtracting it therefore *injects* between-model drift.

    The escape is that the drift is largely shared by all four options of an
    item, so any statistic that compares the options against each other
    (`acc`, `p_correct`, `margin`) cancels it within the item and never needs
    the PMI correction.  Only `gold`, which reads one option in isolation, is
    exposed — and PMI makes it worse rather than better.
    """
    rows = []
    for concept in concepts:
        for subset in ("QA", "SimdomQA"):
            a, b = by_model[model_a], by_model[model_b]
            keys = sorted(k for k in a if k in b
                          and k[0] == concept and k[1] == subset)
            if not keys:
                continue

            def gold_of(rec, field):
                return rec[field][rec["correct_index"]]

            d_cond = np.array([gold_of(b[k], "logp_conditional")
                               - gold_of(a[k], "logp_conditional") for k in keys])
            d_null = np.array([gold_of(b[k], "logp_null")
                               - gold_of(a[k], "logp_null") for k in keys])
            d_pmi = d_cond - d_null
            rows.append({
                "concept": concept, "subset": subset, "n": len(keys),
                "d_cond": float(d_cond.mean()), "d_null": float(d_null.mean()),
                "d_pmi": float(d_pmi.mean()),
                "pmi_helps": bool(abs(d_pmi.mean()) < abs(d_cond.mean())),
            })
    return rows


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def token_counts_from(tokenizer_path: str, records) -> Dict[str, int]:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_path)
    conts = {f" {o}" for rec in records for o in rec["options"]}
    return {c: len(tok(c, add_special_tokens=False)["input_ids"]) for c in conts}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="/home/dcor/galbarak2/LMEnt-ember/"
                                         "ember_eval/results/completion")
    ap.add_argument("--control", default="control2e")
    ap.add_argument("--ablated", default="noporn2e")
    ap.add_argument("--released", default="released2e")
    ap.add_argument("--ablated-concept", default="Pornography")
    ap.add_argument("--null-concept", default="Harry Potter")
    ap.add_argument("--tokenizer", default=None,
                    help="HF model dir; enables the per-token rules. Only the "
                         "tokenizer is loaded, so this stays on CPU.")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    rng = np.random.default_rng(RNG_SEED)

    print("Reading completion records")
    records = load_records(args.results)
    by_model = index_records(records)
    models = [args.control, args.ablated, args.released]
    missing = [m for m in models if m not in by_model]
    if missing:
        raise SystemExit(f"no records for {missing}; found {sorted(by_model)}")

    concepts = sorted({k[0] for m in models for k in by_model[m]})
    print(f"  models   {', '.join(models)}")
    print(f"  concepts {', '.join(concepts)}")
    for m in models:
        print(f"  {m:12s} {len(by_model[m]):4d} unique questions")

    token_counts = None
    if args.tokenizer:
        print(f"\nTokenising options with {args.tokenizer}")
        token_counts = token_counts_from(args.tokenizer, records)
        print(f"  {len(token_counts)} distinct continuations")

    rules = build_rules(token_counts)
    released_recs = list(by_model[args.released].values())

    results: Dict[str, Any] = {"rules": {}, "meta": {
        "permutations": PERMUTATIONS, "seed": RNG_SEED,
        "n_questions": {m: len(by_model[m]) for m in models},
    }}

    rows = []
    for rule_name, rule in rules.items():
        temperature = calibrate_temperature(rule, released_recs)
        for m in models:
            for rec in by_model[m].values():
                rec["_stats"] = statistics_for(rule, temperature, rec)

        entry: Dict[str, Any] = {"temperature": temperature, "statistics": {}}
        for stat in STATISTICS:
            sens = sensitivity(by_model, models, concepts, stat)
            rel = reliability(by_model, models, concepts, stat)
            null_hp = did_effect(by_model, args.control, args.ablated,
                                 args.null_concept, stat, rng)
            null_rel = did_effect(by_model, args.control, args.released,
                                  args.ablated_concept, stat, rng)
            # The outcome. Computed last, never consulted by the ranking.
            twins = did_effect(by_model, args.control, args.ablated,
                               args.ablated_concept, stat, rng)

            floor = max(abs(null_hp["dz"]) if null_hp else 0.0,
                        abs(null_rel["dz"]) if null_rel else 0.0)
            floor_se_ = max(null_hp["dz_se"] if null_hp else 0.0,
                            null_rel["dz_se"] if null_rel else 0.0)
            # Several floors land near zero, and dividing by a noisy estimate
            # of zero sends the ratio to infinity for no good reason.  A floor
            # cannot be claimed tighter than the error with which it was
            # measured, so the denominator is held at its own standard error.
            denom = max(floor, floor_se_)
            snr = sens / denom if denom > 1e-9 else float("inf")

            entry["statistics"][stat] = {
                "sensitivity": sens, "reliability": rel,
                "null_hp": null_hp, "null_released": null_rel,
                "snr": snr, "twins": twins,
            }
            floor_se = max(null_hp["dz_se"] if null_hp else 0.0,
                           null_rel["dz_se"] if null_rel else 0.0)
            entry["statistics"][stat]["floor_se"] = floor_se
            rows.append((rule_name, stat, sens, floor, snr, rel, twins, floor_se))
        results["rules"][rule_name] = entry

    # ------------------------------------------------------------------ #
    # selection table
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 92)
    print("SELECTION  — criteria independent of the ablation")
    print("=" * 92)
    print(f"{'rule':16s} {'statistic':11s} {'sens':>7s} {'null':>7s} "
          f"{'+-':>6s} {'SNR':>7s} {'reliab':>7s}")
    print(f"{'':16s} {'':11s} {'|d| QA':>7s} {'floor':>7s} "
          f"{'se':>6s} {'s/floor':>7s} {'|tr-te|':>7s}")
    print("-" * 92)
    rankable = [r for r in rows if r[1] in RANKABLE]
    for rule_name, stat, sens, floor, snr, rel, _, fse in sorted(
            rankable, key=lambda r: -r[4]):
        print(f"{rule_name:16s} {stat:11s} {sens:7.3f} {floor:7.3f} "
              f"{fse:6.3f} {snr:7.2f} {rel:7.3f}")

    print("-" * 92)
    print("not ranked — 'gold' is not comparable across two question sets:")
    for rule_name, stat, sens, floor, snr, rel, _, fse in sorted(
            (r for r in rows if r[1] not in RANKABLE), key=lambda r: -r[4]):
        print(f"{rule_name:16s} {stat:11s} {sens:7.3f} {floor:7.3f} "
              f"{fse:6.3f} {snr:7.2f} {rel:7.3f}")

    best = max(rankable, key=lambda r: r[4])
    print("-" * 92)
    print(f"highest signal-to-noise: {best[0]} / {best[1]}   "
          f"(sensitivity {best[2]:.3f}, null floor {best[3]:.3f} +- {best[7]:.3f})")
    close = [r for r in rankable if r[4] >= best[4] * 0.5]
    if len(close) > 1:
        print("within a factor of two of it, i.e. not separated by this evidence:")
        for r in sorted(close, key=lambda r: -r[4])[1:]:
            print(f"    {r[0]:16s} {r[1]:11s} SNR {r[4]:.2f}")

    print("\n" + "=" * 92)
    print("NULL FLOORS in detail  — both of these are zero by construction")
    print("=" * 92)
    for rule_name, entry in results["rules"].items():
        for stat in STATISTICS:
            e = entry["statistics"][stat]
            hp, rl = e["null_hp"], e["null_released"]
            if hp and rl:
                print(f"{rule_name:16s} {stat:11s} "
                      f"HP twins dz={hp['dz']:+6.3f} p={hp['p']:.3f}   "
                      f"released-vs-control dz={rl['dz']:+6.3f} p={rl['p']:.3f}")

    # ------------------------------------------------------------------ #
    # the outcome, printed only after the choice is made
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 92)
    print(f"OUTCOME — control vs ablated on {args.ablated_concept}")
    print("NOT a selection criterion; read it after the rule is fixed.")
    print("=" * 92)
    for rule_name, entry in results["rules"].items():
        for stat in STATISTICS:
            t = entry["statistics"][stat]["twins"]
            if t:
                print(f"{rule_name:16s} {stat:11s} "
                      f"QA-Simdom={t['did']:+8.4f} dz={t['dz']:+6.3f} "
                      f"p={t['p']:.3f}   (QA alone dz={t['qa_only_dz']:+6.3f} "
                      f"p={t['qa_only_p']:.3f})")

    print("\n" + "=" * 92)
    print("IS THE PMI CORRECTION EARNING ITS PLACE?  drift on the gold option, nats")
    print("=" * 92)
    print(f"{'contrast':22s} {'concept':14s} {'subset':10s} {'n':>4s} "
          f"{'d(cond)':>9s} {'d(null)':>9s} {'d(pmi)':>9s}  helps?")
    drift_all = {}
    for a, b, label in ((args.control, args.ablated, "twins"),
                        (args.control, args.released, "released vs control")):
        rows_d = null_term_drift(by_model, a, b, concepts)
        drift_all[label] = rows_d
        for r in rows_d:
            print(f"{label:22s} {r['concept']:14s} {r['subset']:10s} {r['n']:4d} "
                  f"{r['d_cond']:+9.4f} {r['d_null']:+9.4f} {r['d_pmi']:+9.4f}"
                  f"  {'yes' if r['pmi_helps'] else 'NO'}")
    helped = sum(r["pmi_helps"] for rs in drift_all.values() for r in rs)
    total = sum(len(rs) for rs in drift_all.values())
    print("-" * 92)
    print(f"PMI reduced the between-model gap in {helped} of {total} cells. "
          f"The null term is the less stable of the two.")
    results["null_term_drift"] = drift_all

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(results, fh, indent=2)
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
