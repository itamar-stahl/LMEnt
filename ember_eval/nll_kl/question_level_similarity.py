#!/usr/bin/env python
"""How close is each erasure model to its twin, question by question?

The paper compares models with the absolute value of the *mean signed* answer-NLL
difference, |mean_i(NLL_M(i) - NLL_T(i))|. A model that is +2 nats off on half the
questions and -2 nats off on the other half scores 0 there: the mean cancels, and
the cancellation is invisible. This script reports the quantity that cannot cancel,

    D_abs(M,T) = (1/N) sum_i |NLL_M(i) - NLL_T(i)|

against the same distance for the unerased full model, D_abs(F,T), as a ratio

    R_abs(M,T) = D_abs(M,T) / D_abs(F,T)

    R_abs < 1  the erasure moved the model toward the twin
    R_abs = 1  it is no closer than doing nothing
    R_abs > 1  it is further from the twin than doing nothing

and as the fraction of individual questions on which the erasure model is the
closer of the two,

    P_closer(M,T) = (1/N) sum_i 1[ |NLL_M(i)-NLL_T(i)| < |NLL_F(i)-NLL_T(i)| ]

Exact ties are counted separately (P_tied) and never as "closer".

Inputs are the per-question answer NLLs already produced by score_model.py and
assembled by report.py -- the same teacher-forced mean-over-answer-tokens NLL the
paper's tables use. No model is loaded and nothing is re-scored. Only the 50
held-out `target_test` questions of each concept are read; selection-phase,
neighbour, unrelated and SciQ items are dropped at load.

    python ember_eval/nll_kl/question_level_similarity.py \
        --results-root <run>/results_accwinners \
        --out-dir ember_eval/nll_kl/question_level_similarity
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent

SPLIT = "target_test"
METHODS = ("EMBER", "RMU", "SNMF")

# The accuracy-selected winners, as ember_eval/acc_selection/results_sciq/
# selected_checkpoints.json declares them. --selected-checkpoints re-reads that
# file and refuses to run if it has moved on.
CONCEPTS: Tuple[Dict[str, Any], ...] = (
    {"slug": "rome", "name": "Ancient Rome",
     "full": "lment-1b-control-2e-b131k", "twin": "lment-1b-norome-2e-b131k",
     "methods": {"EMBER": "ember_rome_d200", "RMU": "rmu_rome_L6hi_a10",
                 "SNMF": "snmf_rome_ratio_out"}},
    {"slug": "baseball", "name": "Baseball",
     "full": "lment-1b-control-2e-b131k", "twin": "lment-1b-nobaseball-2e-b131k",
     "methods": {"EMBER": "ember_baseball_d10", "RMU": "rmu_baseball_L6hi_a10",
                 "SNMF": "snmf_baseball_ratio_both"}},
    {"slug": "ai", "name": "Artificial intelligence",
     "full": "lment-1b-control-2e-b131k", "twin": "lment-1b-noai-2e-b131k",
     "methods": {"EMBER": "ember_ai_d500", "RMU": "rmu_ai_L6hi_a10",
                 "SNMF": "snmf_ai_ratio_in"}},
)

PER_QUESTION_COLUMNS = ("topic", "method", "checkpoint", "item_id", "nll_method",
                        "nll_twin", "nll_full", "abs_method_twin", "abs_full_twin",
                        "is_closer", "is_tied")
SUMMARY_COLUMNS = ("topic", "method", "checkpoint", "n", "D_abs_M_T", "D_abs_F_T",
                   "R_abs", "R_abs_ci_low", "R_abs_ci_high", "P_closer",
                   "P_closer_ci_low", "P_closer_ci_high", "P_tied")


class Validation:
    """Every check is executed and recorded, pass or fail; nothing is asserted."""

    def __init__(self) -> None:
        self.checks: List[Dict[str, Any]] = []

    def record(self, scope: str, name: str, ok: bool, detail: str) -> bool:
        self.checks.append({"scope": scope, "check": name,
                            "result": "PASS" if ok else "FAIL", "detail": detail})
        return ok

    @property
    def failed(self) -> List[Dict[str, Any]]:
        return [c for c in self.checks if c["result"] == "FAIL"]

    def report(self) -> str:
        w = max(len(c["scope"]) for c in self.checks)
        lines = [f"  [{c['result']}] {c['scope']:<{w}}  {c['check']}: {c['detail']}"
                 for c in self.checks]
        return "\n".join(lines)


def git_commit(path: Path) -> Tuple[str, bool]:
    """HEAD at run time, and whether the tree was dirty. HEAD is necessarily the
    commit *before* the one carrying this manifest; a rerun on a clean checkout
    reproduces every number but records its own HEAD here."""
    try:
        head = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "-C", str(path), "status", "--porcelain"],
                                             text=True, stderr=subprocess.DEVNULL).strip())
        return head, dirty
    except Exception:  # noqa: BLE001
        return "unknown", False


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_selected_checkpoints(path: Path, v: Validation) -> None:
    """Refuse to run on checkpoints the selection file no longer names."""
    if not path.exists():
        v.record("checkpoints", "selected_checkpoints.json present", False, f"missing: {path}")
        return
    declared = {(w["topic"], w["method"]): w["model_label"]
                for w in json.loads(path.read_text())["winners"]}
    for c in CONCEPTS:
        for method in METHODS:
            want = c["methods"][method]
            got = declared.get((c["slug"], method))
            if got is None and method not in ("EMBER", "RMU", "SNMF"):
                continue          # an added condition is frozen in its own file
            v.record(c["name"], f"selected checkpoint {method}", got == want,
                     f"expected {want}, selection file says {got}")


def rows_by_label(results: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {r["label"]: r for r in results["rows"]}


def split_items(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The held-out target questions only -- both the set name and the phase flag."""
    return [i for i in row["items"] if i["set"] == SPLIT and i["phase"] == "test"]


def series(row: Dict[str, Any], field: str) -> Dict[str, float]:
    return {i["id"]: float(i[field]) for i in split_items(row)}


def finite(values: Dict[str, float]) -> bool:
    return all(np.isfinite(x) for x in values.values())


def load_concept(results_root: Path, concept: Dict[str, Any], v: Validation
                 ) -> Tuple[List[str], Dict[str, Dict[str, float]]]:
    """Per-question NLL for full, twin and the three methods, keyed by item id.

    results.json stores pairs, not models: each row carries the evaluated model's
    NLL and its reference's. The same model therefore appears in several rows, and
    the values must agree exactly -- they come from one scoring pass. Checking that
    is a free consistency test on the assembly, so it is checked.
    """
    name = concept["name"]
    path = results_root / concept["slug"] / "results.json"
    res = json.loads(path.read_text())
    rows = rows_by_label(res)

    v.record(name, "results.json names the expected full model",
             res["full"] == concept["full"], f"{res['full']}")
    v.record(name, "results.json names the expected twin",
             res["twin"] == concept["twin"], f"{res['twin']}")
    for method in METHODS:
        v.record(name, f"results.json names the expected {method} checkpoint",
                 res["erased"].get(method) == concept["methods"][method],
                 f"{res['erased'].get(method)}")

    twin_vs_full = rows["Twin vs Full"]
    nll: Dict[str, Dict[str, float]] = {
        "twin": series(twin_vs_full, "nll_eval"),
        "full": series(twin_vs_full, "nll_ref"),
    }
    for method in METHODS:
        vs_twin, vs_full = rows[f"{method} vs Twin"], rows[f"{method} vs Full"]
        a, b = series(vs_twin, "nll_eval"), series(vs_full, "nll_eval")
        v.record(name, f"{method} NLL identical in its vs-Twin and vs-Full rows",
                 a == b, f"{len(a)} items compared")
        v.record(name, f"twin NLL identical in Twin-vs-Full and {method}-vs-Twin",
                 series(vs_twin, "nll_ref") == nll["twin"], "exact match required")
        v.record(name, f"full NLL identical in Twin-vs-Full and {method}-vs-Full",
                 series(vs_full, "nll_ref") == nll["full"], "exact match required")
        nll[method] = a

    # split, uniqueness, pairing, finiteness
    ids = sorted(nll["twin"])
    raw = [i["id"] for i in split_items(twin_vs_full)]
    v.record(name, "held-out split only", True,
             f"set == '{SPLIT}' and phase == 'test' for every item read")
    v.record(name, "exactly 50 target_test items", len(ids) == 50, f"n = {len(ids)}")
    v.record(name, "no duplicate item ids", len(raw) == len(set(raw)),
             f"{len(raw)} rows, {len(set(raw))} distinct")
    for label, vals in nll.items():
        v.record(name, f"{label}: same 50 ids as the twin", sorted(vals) == ids,
                 f"{len(vals)} ids, {len(set(vals) ^ set(ids))} symmetric difference")
        v.record(name, f"{label}: all values finite and present", finite(vals) and len(vals) == 50,
                 "no NaN, no inf, no missing")
    return ids, nll


def check_nll_definition(scored_root: Path, concept: Dict[str, Any], ids: List[str],
                         nll: Dict[str, Dict[str, float]], v: Validation) -> None:
    """NLL is the mean over the correct continuation's answer tokens.

    Read back score_model.py's own records: `nll` must equal mean(token_nlls), and
    the value carried into results.json must be that same number.
    """
    name = concept["name"]
    labels = {"full": concept["full"], "twin": concept["twin"],
              **{m: concept["methods"][m] for m in METHODS}}
    for role, model in labels.items():
        rec_path = scored_root / model / concept["slug"] / "records.json"
        if not rec_path.exists():
            v.record(name, f"{role}: scored records readable", False, f"missing: {rec_path}")
            continue
        recs = {r["id"]: r for r in json.loads(rec_path.read_text())["records"]
                if r["set"] == SPLIT}
        dev_mean = max(abs(recs[i]["nll"] - float(np.mean(recs[i]["token_nlls"]))) for i in ids)
        dev_used = max(abs(recs[i]["nll"] - nll[role][i]) for i in ids)
        ntok = [len(recs[i]["token_nlls"]) for i in ids]
        v.record(name, f"{role}: NLL == mean over answer tokens", dev_mean < 1e-9,
                 f"max |nll - mean(token_nlls)| = {dev_mean:.2e}, "
                 f"answer lengths {min(ntok)}-{max(ntok)} tokens")
        v.record(name, f"{role}: value used matches the scored record", dev_used == 0.0,
                 f"max deviation {dev_used:.2e}")


def bootstrap(a_method: np.ndarray, a_full: np.ndarray, reps: int, seed: int
              ) -> Dict[str, Tuple[float, float]]:
    """Paired nonparametric bootstrap over questions: one index draw, both distances.

    Question indices are resampled with replacement; the identical indices index
    both |NLL_M - NLL_T| and |NLL_F - NLL_T|, so the pairing survives every draw.
    Tokens are never resampled -- the per-question NLL is the unit.
    """
    n = a_method.size
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(reps, n))
    boot_m, boot_f = a_method[idx].mean(axis=1), a_full[idx].mean(axis=1)
    boot_p = (a_method < a_full).astype(np.float64)[idx].mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        boot_r = np.where(boot_f > 0, boot_m / boot_f, np.nan)
    keep = np.isfinite(boot_r)
    return {
        "R_abs": (float(np.percentile(boot_r[keep], 2.5)),
                  float(np.percentile(boot_r[keep], 97.5))),
        "P_closer": (float(np.percentile(boot_p, 2.5)), float(np.percentile(boot_p, 97.5))),
        "n_usable": int(keep.sum()),
    }


def analyse(concept: Dict[str, Any], ids: List[str], nll: Dict[str, Dict[str, float]],
            reps: int, seed: int) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    twin = np.array([nll["twin"][i] for i in ids])
    full = np.array([nll["full"][i] for i in ids])
    a_full = np.abs(full - twin)
    per_question: List[Dict[str, Any]] = []
    summary: List[Dict[str, Any]] = []
    for method in METHODS:
        m = np.array([nll[method][i] for i in ids])
        a_m = np.abs(m - twin)
        closer, tied = a_m < a_full, a_m == a_full
        ci = bootstrap(a_m, a_full, reps, seed)
        for k, item in enumerate(ids):
            per_question.append({
                "topic": concept["name"], "method": method,
                "checkpoint": concept["methods"][method], "item_id": item,
                "nll_method": m[k], "nll_twin": twin[k], "nll_full": full[k],
                "abs_method_twin": a_m[k], "abs_full_twin": a_full[k],
                "is_closer": int(closer[k]), "is_tied": int(tied[k]),
            })
        summary.append({
            "topic": concept["name"], "method": method,
            "checkpoint": concept["methods"][method], "n": len(ids),
            "D_abs_M_T": float(a_m.mean()), "D_abs_F_T": float(a_full.mean()),
            "R_abs": float(a_m.mean() / a_full.mean()),
            "R_abs_ci_low": ci["R_abs"][0], "R_abs_ci_high": ci["R_abs"][1],
            "P_closer": float(closer.mean()),
            "P_closer_ci_low": ci["P_closer"][0], "P_closer_ci_high": ci["P_closer"][1],
            "P_tied": float(tied.mean()),
        })
    return per_question, summary


def write_csv(path: Path, columns: Tuple[str, ...], rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(columns), lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({c: (f"{r[c]:.10f}" if isinstance(r[c], float) else r[c])
                        for c in columns})


def write_tex(path: Path, summary: List[Dict[str, Any]]) -> None:
    pct = lambda x: f"{100 * x:.1f}"  # noqa: E731
    lines = ["% Requires \\usepackage{booktabs}", "",
             "\\begin{table}[t]", "\\centering", "\\small",
             "\\begin{tabular}{llrlrl}", "\\toprule",
             "Concept & Method & $R_{\\mathrm{abs}}$ & 95\\% CI "
             "& $P_{\\mathrm{closer}}$ & 95\\% CI \\\\", "\\midrule"]
    for ci, concept in enumerate(CONCEPTS):
        if ci:
            lines.append("\\midrule")
        for ri, method in enumerate(METHODS):
            r = next(s for s in summary
                     if s["topic"] == concept["name"] and s["method"] == method)
            lines.append(
                f"{concept['name'] if ri == 0 else ''} & {method} "
                f"& {r['R_abs']:.3f} & [{r['R_abs_ci_low']:.3f}, {r['R_abs_ci_high']:.3f}] "
                f"& {pct(r['P_closer'])}\\% "
                f"& [{pct(r['P_closer_ci_low'])}, {pct(r['P_closer_ci_high'])}]\\% \\\\")
    lines += ["\\bottomrule", "\\end{tabular}",
              "\\caption{Question-level distance to the concept-excluded twin on the 50 "
              "held-out target questions. $R_{\\mathrm{abs}}$ is the mean absolute paired "
              "answer-NLL difference to the twin, divided by the same distance for the "
              "unerased full model; below 1 the erasure model is closer to the twin than "
              "doing nothing. $P_{\\mathrm{closer}}$ is the fraction of individual "
              "questions on which it is closer. Intervals are percentile 95\\% CIs from a "
              "paired bootstrap over questions (10{,}000 resamples, seed 0).}",
              "\\label{tab:question-level-similarity}", "\\end{table}", ""]
    path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-root", required=True,
                    help="dir holding rome/results.json, baseball/results.json, ai/results.json")
    ap.add_argument("--scored-root", help="score_model.py output root; "
                                          "default <results-root>/../scored")
    ap.add_argument("--selected-checkpoints",
                    default=str(HERE.parent / "acc_selection/results_sciq/selected_checkpoints.json"))
    ap.add_argument("--extra-methods", metavar="JSON",
                    help="add conditions beyond the paper's three, as "
                         "{topic name: {method: expected checkpoint label}}. The "
                         "checkpoint identity check applies to these exactly as it "
                         "does to the defaults -- the label is still asserted against "
                         "what results.json names. Used for the RMU+EMBER / SNMF+EMBER "
                         "ensembles, whose winners are frozen in a separate file.")
    ap.add_argument("--out-dir", default=str(HERE / "question_level_similarity"))
    ap.add_argument("--out-csv", help="default <out-dir>/summary.csv")
    ap.add_argument("--out-tex", help="default <out-dir>/table.tex")
    ap.add_argument("--bootstrap-repetitions", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    results_root = Path(a.results_root).resolve()
    scored_root = Path(a.scored_root).resolve() if a.scored_root else results_root.parent / "scored"
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = Path(a.out_csv) if a.out_csv else out_dir / "summary.csv"
    out_tex = Path(a.out_tex) if a.out_tex else out_dir / "table.tex"

    global METHODS
    if a.extra_methods:
        extra = json.loads(Path(a.extra_methods).read_text())
        added: List[str] = []
        for concept in CONCEPTS:
            for method, label in extra.get(concept["name"], {}).items():
                concept["methods"][method] = label
                if method not in added:
                    added.append(method)
        METHODS = METHODS + tuple(m for m in added if m not in METHODS)
        print(f"extra conditions: {', '.join(added)}\n")

    v = Validation()
    check_selected_checkpoints(Path(a.selected_checkpoints), v)

    per_question: List[Dict[str, Any]] = []
    summary: List[Dict[str, Any]] = []
    inputs: List[str] = [a.selected_checkpoints]
    for concept in CONCEPTS:
        ids, nll = load_concept(results_root, concept, v)
        check_nll_definition(scored_root, concept, ids, nll, v)
        pq, sm = analyse(concept, ids, nll, a.bootstrap_repetitions, a.seed)
        per_question += pq
        summary += sm
        inputs.append(str(results_root / concept["slug"] / "results.json"))

    v.record("output", "450 per-question rows", len(per_question) == 450, f"{len(per_question)}")
    v.record("output", "9 summary rows", len(summary) == 9, f"{len(summary)}")

    print("Validation\n" + v.report())
    if v.failed:
        print(f"\n{len(v.failed)} check(s) FAILED; nothing written.", file=sys.stderr)
        raise SystemExit(1)
    print(f"\nAll {len(v.checks)} checks passed.\n")

    write_csv(out_dir / "per_question.csv", PER_QUESTION_COLUMNS, per_question)
    write_csv(out_csv, SUMMARY_COLUMNS, summary)
    write_tex(out_tex, summary)

    commit, dirty = git_commit(HERE)
    manifest = {
        "analysis": "question-level distance to the concept-excluded twin, "
                    "mean absolute paired answer-NLL difference",
        "git_commit": commit, "git_tree_dirty_at_run_time": dirty,
        "git_commit_note": "HEAD when the analysis ran, i.e. the commit before the "
                           "one that carries these artifacts",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "script": str(Path(__file__).resolve()),
        "command": " ".join([Path(sys.executable).name] + sys.argv),
        "model_inference_rerun": False,
        "model_inference_note": "no checkpoint was loaded; per-question NLLs were read from "
                                "the existing report.py output, which score_model.py produced "
                                "on 2026-09-20/21",
        "concepts": [c["name"] for c in CONCEPTS],
        "split": SPLIT,
        "split_note": "held-out test half only; selection, neighbour, unrelated and sciq "
                      "items are dropped at load",
        "questions_per_concept": {c["name"]: 50 for c in CONCEPTS},
        "checkpoints": {c["name"]: {"full": c["full"], "twin": c["twin"], **c["methods"]}
                        for c in CONCEPTS},
        "metrics": {
            "NLL": "teacher-forced mean over the correct continuation's answer tokens, nats",
            "D_abs(M,T)": "(1/N) sum_i |NLL_M(i) - NLL_T(i)|",
            "D_abs(F,T)": "(1/N) sum_i |NLL_F(i) - NLL_T(i)|",
            "R_abs(M,T)": "D_abs(M,T) / D_abs(F,T); <1 closer to the twin than the full model",
            "P_closer(M,T)": "(1/N) sum_i 1[|NLL_M(i)-NLL_T(i)| < |NLL_F(i)-NLL_T(i)|]",
            "P_tied(M,T)": "(1/N) sum_i 1[|NLL_M(i)-NLL_T(i)| == |NLL_F(i)-NLL_T(i)|]; "
                           "ties are never counted as closer",
        },
        "bootstrap": {"repetitions": a.bootstrap_repetitions, "seed": a.seed,
                      "unit": "question", "paired": True,
                      "scheme": "resample question indices with replacement; the same indices "
                                "index the method and full-model distances in every draw",
                      "interval": "percentile, 2.5th and 97.5th"},
        "input_paths": {"results_root": str(results_root), "scored_root": str(scored_root),
                        "selected_checkpoints": str(Path(a.selected_checkpoints).resolve()),
                        "results_json": [str(results_root / c["slug"] / "results.json")
                                         for c in CONCEPTS]},
        "input_sha256": {p: sha256(Path(p)) for p in inputs},
        "outputs": [str(p.resolve()) for p in
                    (out_dir / "per_question.csv", out_csv, out_tex,
                     out_dir / "run_manifest.json")],
        "validation": {"n_checks": len(v.checks), "n_failed": 0, "checks": v.checks},
    }
    (out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")

    hdr = f"{'Concept':24s} {'Method':6s} {'R_abs':>7s} {'95% CI':>18s} {'P_closer':>9s} {'95% CI':>16s}"
    print(hdr + "\n" + "-" * len(hdr))
    for s in summary:
        print(f"{s['topic']:24s} {s['method']:6s} {s['R_abs']:7.3f} "
              f"[{s['R_abs_ci_low']:7.3f},{s['R_abs_ci_high']:7.3f}] "
              f"{100 * s['P_closer']:8.1f}% "
              f"[{100 * s['P_closer_ci_low']:5.1f},{100 * s['P_closer_ci_high']:5.1f}]%")
    print("\nwrote", out_dir)


if __name__ == "__main__":
    main()
