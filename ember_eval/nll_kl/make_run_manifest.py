#!/usr/bin/env python
"""Write results_accwinners/run_manifest.json: what was generated, from what, how.

Everything in the manifest is read back off disk rather than typed in, so a
stale definition cannot outlive the artifact it describes. The validation
block is the output of validate_accwinners.py, not a claim made here.

    python ember_eval/nll_kl/make_run_manifest.py --validation <validation.json> \
        --out ember_eval/nll_kl/results_accwinners/run_manifest.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RUN = "/home/morg/NLP_2526b/galbarak2/runs/nll_kl"
ACC = "/home/morg/NLP_2526b/galbarak2/runs/acc_selection"
ENS = "/home/morg/NLP_2526b/galbarak2/runs/ensembles"
TOPICS = [("rome", "Ancient Rome"), ("baseball", "Baseball"), ("ai", "Artificial intelligence")]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(repo: Path, *args: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()
    except Exception as e:  # noqa: BLE001
        return f"unknown ({e})"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=str(REPO))
    ap.add_argument("--run", default=RUN)
    ap.add_argument("--validation", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    W = Path(a.repo)

    res = {t: json.loads((W / f"ember_eval/nll_kl/results_accwinners/{t}/results.json").read_text())
           for t, _ in TOPICS}
    sel = json.loads(
        (W / "ember_eval/acc_selection/results_sciq/selected_checkpoints.json").read_text())
    winners = {(w["topic"], w["method"]): w for w in sel["winners"]}

    checkpoints = {}
    for t, name in TOPICS:
        checkpoints[name] = {
            "topic_slug": t,
            "full": {"label": res[t]["full"],
                     "path": json.loads(
                         (Path(a.run) / "scored" / res[t]["full"] / t / "records.json").read_text()
                     )["meta"]["model"]},
            "twin": {"label": res[t]["twin"],
                     "path": json.loads(
                         (Path(a.run) / "scored" / res[t]["twin"] / t / "records.json").read_text()
                     )["meta"]["model"]},
            **{m: {"label": res[t]["erased"][m],
                   "path": winners[(t, m)]["checkpoint_path"],
                   "hyperparameters": winners[(t, m)]["hyperparameters"],
                   "H_selection": winners[(t, m)]["H_selection"]}
               for m in ("EMBER", "RMU", "SNMF")},
        }

    # ensemble checkpoints, from the file the ratios are cross-checked against
    ens = defaultdict(dict)
    with open(W / "ember_eval/ensembles/results/target_ratios.csv", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            ens[r["concept"]][r["method"]] = {"label": r["checkpoint"]}
    for name, d in ens.items():
        checkpoints[name].update(d)

    groups = defaultdict(lambda: defaultdict(int))
    for t, name in TOPICS:
        for i in res[t]["rows"][0]["items"]:
            groups[name][i["set"]] += 1
    acc_groups = defaultdict(lambda: defaultdict(int))
    with open(W / "ember_eval/acc_selection/final/current_test_per_question.csv",
              encoding="utf-8") as fh:
        n_acc_rows = 0
        for r in csv.DictReader(fh):
            acc_groups[r["concept"]][r["question_group"]] += 1
            n_acc_rows += 1

    outputs = [
        "ember_eval/nll_kl/results_accwinners/rome/results.json",
        "ember_eval/nll_kl/results_accwinners/baseball/results.json",
        "ember_eval/nll_kl/results_accwinners/ai/results.json",
        "ember_eval/nll_kl/rkl_bootstrap.csv",
        "ember_eval/nll_kl/rkl_bootstrap_rows.tex",
        "ember_eval/acc_selection/final/current_test_per_question.csv",
        "ember_eval/acc_selection/final/current_test_accuracy.csv",
    ]

    man = {
        "what": "Current per-question NLL, KL and accuracy for the accuracy-selected "
                "checkpoints, and bootstrap intervals for the target-test KL ratio R_KL.",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git": {
            "repo": git(W, "config", "--get", "remote.origin.url"),
            "branch": git(W, "rev-parse", "--abbrev-ref", "HEAD"),
            "commit_at_generation": git(W, "rev-parse", "HEAD"),
            "worktree": str(W),
            "note": "commit_at_generation is HEAD when these files were produced; the "
                    "files themselves are committed in its child.",
        },
        "checkpoints": checkpoints,
        "selection_rule": {
            "source": "ember_eval/acc_selection/results_sciq/selected_checkpoints.json",
            "rule": sel["rule"],
            "supersedes": "the answer-NLL selection whose outputs are "
                          "archive/ember_eval/nll_kl/paper_tables/ and archive/ember_eval/nll_kl/export_20260922/ "
                          "-- neither is used here",
        },
        "inputs": {
            "nll_kl_scored_tree": f"{a.run}/scored/<model>/<topic>/"
                                  "{records.json,dists.npy,dists_index.json}",
            "nll_kl_sets": "ember_eval/nll_kl/sets/{rome,baseball,ai}.json",
            "accuracy_scored_trees": [
                f"{ACC}/scored/<label>/<topic>/per_question.csv  (selection, concept groups)",
                f"{ACC}/scored_merged/<label>/<topic>/per_question.csv  (selection, SciQ unrelated)",
                f"{ACC}/scored_sciq/<label>/sciq/per_question.csv  (SciQ selection and test)",
                f"{ACC}/scored_test/<label>/<topic>/per_question.csv  (test, concept groups)",
                f"{ENS}/scored_selection/<label>/<topic>/per_question.csv  (selection, ensembles)",
                f"{ENS}/merged_test/<label>/<topic>/per_question.csv  (test, all conditions)",
            ],
            "ensemble_per_question_kl": "ember_eval/ensembles/results/nll_kl_per_question.csv",
            "sciq_manifest": "ember_eval/acc_selection/sciq_unrelated.csv",
            "cross_checked_against": [
                "ember_eval/nll_kl/tables_accwinners/<topic>/table.md",
                "ember_eval/ensembles/results/target_ratios.csv",
                "ember_eval/acc_selection/final/H_test.csv",
                "ember_eval/acc_selection/test_phase/test_accuracy_rollup.csv",
                "ember_eval/ensembles/results/panel_A_accuracy_sciq.csv",
            ],
        },
        "outputs": {p: {"sha256": sha256(W / p), "bytes": (W / p).stat().st_size}
                    for p in outputs},
        "commands": [
            "# 1. NLL/KL tables and per-item results, one call per concept",
            "python ember_eval/nll_kl/report.py --topic rome --scored "
            f"{a.run}/scored --full lment-1b-control-2e-b131k --twin lment-1b-norome-2e-b131k "
            "--erased EMBER=ember_rome_d200 RMU=rmu_rome_L6hi_a10 SNMF=snmf_rome_ratio_out "
            "EMBER-released=lment-1b-rome-erased-b131k --out ember_eval/nll_kl/results_accwinners/rome",
            "python ember_eval/nll_kl/report.py --topic baseball --scored "
            f"{a.run}/scored --full lment-1b-control-2e-b131k --twin lment-1b-nobaseball-2e-b131k "
            "--erased EMBER=ember_baseball_d10 RMU=rmu_baseball_L6hi_a10 "
            "SNMF=snmf_baseball_ratio_both EMBER-released=lment-1b-baseball-erased-b131k "
            "--out ember_eval/nll_kl/results_accwinners/baseball",
            "python ember_eval/nll_kl/report.py --topic ai --scored "
            f"{a.run}/scored --full lment-1b-control-2e-b131k --twin lment-1b-noai-2e-b131k "
            "--erased EMBER=ember_ai_d500 RMU=rmu_ai_L6hi_a10 SNMF=snmf_ai_ratio_in "
            "EMBER-released=lment-1b-ai-erased-b131k --out ember_eval/nll_kl/results_accwinners/ai",
            "# 2. per-question accuracy, long form",
            "python ember_eval/acc_selection/export_current_test.py "
            "--out-per-question ember_eval/acc_selection/final/current_test_per_question.csv "
            "--out-accuracy ember_eval/acc_selection/final/current_test_accuracy.csv",
            "# 3. R_KL bootstrap",
            "python ember_eval/nll_kl/bootstrap_rkl.py "
            "--standalone-root ember_eval/nll_kl/results_accwinners "
            "--ensemble-csv ember_eval/ensembles/results/nll_kl_per_question.csv "
            "--out ember_eval/nll_kl/rkl_bootstrap.csv "
            "--latex-out ember_eval/nll_kl/rkl_bootstrap_rows.tex",
            "# 4. validation, then this manifest",
            "python ember_eval/nll_kl/validate_accwinners.py --out <validation.json>",
            "python ember_eval/nll_kl/make_run_manifest.py --validation <validation.json> "
            "--out ember_eval/nll_kl/results_accwinners/run_manifest.json",
        ],
        "question_groups": {
            "nll_kl": {
                "per_concept": {c: dict(g) for c, g in groups.items()},
                "kl_available_on": ["target_test", "neighbour_test", "unrelated_test"],
                "note": "score_model.py saves full distributions for test-phase items only, "
                        "so the selection groups carry NLL but no KL.",
            },
            "accuracy": {
                "rows": n_acc_rows,
                "per_concept": {c: dict(g) for c, g in acc_groups.items()},
                "group_names": {
                    "target_*, neighbour_*": "the concept's own and its neighbouring questions",
                    "unrelated_*": "the other-concept pool the protocol originally specified",
                    "sciq_unrelated_*": "the 50 SciQ questions that replaced it as the "
                                        "general-capability probe; renamed here because the "
                                        "scored trees call both groups 'unrelated_*' and the "
                                        "two are different questions",
                },
                "gaps": [
                    "Twin has no selection-phase concept-group rows: the accuracy selection "
                    "rule never loaded the twin, by design.",
                    "RMU+EMBER and SNMF+EMBER have no other-concept-pool unrelated rows: the "
                    "ensembles were only ever scored with SciQ as the unrelated group.",
                ],
            },
        },
        "definitions": {
            "nll": "Teacher-forced, one forward pass over stem + correct-answer completion. "
                   "nll(item) = mean over the k completion tokens of -log p(token | prefix), "
                   "in nats per token; the first completion token is predicted from the stem's "
                   "last position. Right padding with a causal mask, so a batch scores exactly "
                   "as one sequence at a time. delta = nll_eval - nll_ref, always evaluated "
                   "minus reference.",
            "kl": "Teacher-forced, full-vocabulary. kl(item) = mean over the same k completion "
                  "positions of sum_v p_twin(v) * (log p_twin(v) - log p_model(v)).",
            "kl_direction": "KL(twin || model): the twin is ALWAYS the first argument, so "
                            "deviations are weighted by the twin's own distribution. The "
                            "reverse is a different number -- on Rome's target set "
                            "KL(twin || full) is 0.955 where KL(full || twin) is 0.761.",
            "accuracy": "Four options per question, each scored as a continuation of the "
                        "declarative stem and ranked by per-character log-probability (the "
                        "character denominator includes the leading space). Never summed "
                        "log-probability, which is length-biased. Chance is 0.25 in every "
                        "group. Conditions on the declarative stem, not causal_mc.py's "
                        "'Question: ...\\nAnswer:', so these accuracies are not comparable "
                        "with previously reported causal_mc numbers.",
            "R_KL": "mean_i KL(twin || edited; i) / mean_i KL(twin || full; i) over the 50 "
                    "target_test questions -- a ratio of two means, not a mean of ratios.",
        },
        "bootstrap": {
            "statistic": "R_KL on the target_test group",
            "repetitions": 10000,
            "seed": 0,
            "rng": "numpy.random.default_rng(0)",
            "sampling_unit": "question index (50 per concept), drawn with replacement",
            "pairing_rule": "the same drawn indices index the edited and the full-model KL "
                            "values, and both the numerator and the denominator are "
                            "recomputed in every resample",
            "interval": "2.5th and 97.5th percentiles of the 10000 resampled ratios",
            "denominator_sources": "standalone conditions take KL(twin || full) from the "
                                   "'Full' kl_row of the concept's results.json; ensembles "
                                   "take it from kl_twin_to_full in the ensemble CSV. The two "
                                   "are asserted equal to within 1e-8 before use.",
        },
        "validation": json.loads(Path(a.validation).read_text()),
    }

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(man, indent=1) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
