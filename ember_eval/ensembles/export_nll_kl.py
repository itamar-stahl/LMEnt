#!/usr/bin/env python
"""Held-out NLL and full-vocabulary KL for the two ensemble conditions.

Per question, for each concept x method x group:

    NLL_M(i)  teacher-forced mean over the tokens of the fixed correct
              continuation only -- score_model.py's own value, re-derived here
              from token_nlls and checked against it
    KL        mean over answer positions of KL(Twin || Model) over the full
              vocabulary. The twin is ALWAYS the left/reference distribution.

The unrelated group is SciQ, not the older other-concept pool. Those live in
different sets files -- `sets/<topic>.json` carries target/neighbour plus the
old pool, `sets/sciq.json` carries SciQ -- so unrelated rows are read from the
sciq scoring and target/neighbour from the topic scoring. The 50 SciQ ids are
identical to the ones the accuracy manifest uses, which this checks.

Writes nll_kl_per_question.csv, nll_kl_summary.csv, target_ratios.csv,
nll_selectivity.csv and a run manifest, and fails loudly rather than emitting a
partial export.

    python ember_eval/ensembles/export_nll_kl.py --ens <run root> --out <dir>
"""
from __future__ import annotations

import argparse, csv, hashlib, json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "nll_kl"))
from compare import load_scored, kl_item   # noqa: E402  the published KL path

CONCEPTS = [
    ("rome", "Ancient Rome", "lment-1b-norome-2e-b131k"),
    ("baseball", "Baseball", "lment-1b-nobaseball-2e-b131k"),
    ("ai", "Artificial intelligence", "lment-1b-noai-2e-b131k"),
]
FULL = "lment-1b-control-2e-b131k"
# Frozen by the request; NOT reselected here.
METHODS = {
    "RMU+EMBER": {"rome": "rmuember_rome_L5mid_a100",
                  "baseball": "rmuember_baseball_L5mid_a100",
                  "ai": "rmuember_ai_L5mid_a10"},
    "SNMF+EMBER": {"rome": "snmfv2_rome_ratio_in",
                   "baseball": "snmfv2_baseball_ratio_both",
                   "ai": "snmfv2_ai_ratio_in"},
}
GROUPS = ("target_test", "neighbour_test", "unrelated_test")
BOOT_N, BOOT_SEED = 10000, 0

COLS_PQ = ("concept", "method", "checkpoint", "group", "question_id",
           "nll_method", "nll_full", "nll_twin",
           "delta_nll_method_full", "delta_nll_method_twin", "delta_nll_full_twin",
           "kl_twin_to_method", "kl_twin_to_full")


class V:
    def __init__(self): self.checks: List[Dict[str, Any]] = []
    def rec(self, name, ok, detail):
        self.checks.append({"check": name, "result": "PASS" if ok else "FAIL",
                            "detail": detail})
        return ok
    @property
    def failed(self): return [c for c in self.checks if c["result"] == "FAIL"]


def find_scored(roots: List[Path], label: str, setname: str) -> Path:
    for r in roots:
        d = r / label / setname
        if (d / "records.json").exists():
            return d
    raise SystemExit(f"no scoring for {label}/{setname} under {[str(r) for r in roots]}")


def series(sc: Dict[str, Any], group: str) -> Dict[str, Dict[str, Any]]:
    return {r["id"]: r for r in sc["records"].values()
            if r["set"] == group and r["phase"] == "test"}


def kl_series(twin_sc, model_sc, ids) -> Dict[str, float]:
    """mean over answer positions of KL(twin || model), full vocabulary."""
    if twin_sc["dists"] is None or model_sc["dists"] is None:
        raise SystemExit("missing saved distributions; KL cannot be computed")
    out = {}
    for i in ids:
        s0, e0 = twin_sc["index"][i]
        s1, e1 = model_sc["index"][i]
        out[i] = kl_item(twin_sc["dists"][s0:e0], model_sc["dists"][s1:e1])
    return out


def boot_ci(groups: Dict[str, np.ndarray], seed: int, reps: int) -> Tuple[float, float]:
    """S_NLL = target - 0.5*neighbour - 0.5*unrelated, resampling each group
    independently with replacement."""
    rng = np.random.default_rng(seed)
    t, n, u = (groups[g] for g in GROUPS)
    vals = np.empty(reps)
    for b in range(reps):
        vals[b] = (t[rng.integers(0, t.size, t.size)].mean()
                   - 0.5 * n[rng.integers(0, n.size, n.size)].mean()
                   - 0.5 * u[rng.integers(0, u.size, u.size)].mean())
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def sd(x: np.ndarray) -> float:
    return float(x.std(ddof=1)) if x.size > 1 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ens", required=True)
    ap.add_argument("--nk", default="/home/morg/NLP_2526b/galbarak2/runs/nll_kl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--acc-manifest",
                    default=str(HERE.parent / "acc_selection/questions_with_options_TEST.csv"))
    ap.add_argument("--sciq-manifest", default=str(HERE.parent / "acc_selection/sciq_unrelated.csv"))
    a = ap.parse_args()
    ENS, NK, OUT = Path(a.ens), Path(a.nk), Path(a.out)
    OUT.mkdir(parents=True, exist_ok=True)
    roots = [ENS / "scored_nllkl", ENS / "merged_nllkl", NK / "scored"]
    v = V()

    # the SciQ ids the accuracy path uses, for the cross-path identity check
    acc_sciq = {r["question_id"] for r in csv.DictReader(open(a.sciq_manifest, encoding="utf-8"))
                if r["question_group"] == "unrelated_test"}

    rows: List[Dict[str, Any]] = []
    for slug, cname, twin_lbl in CONCEPTS:
        full_topic = load_scored(find_scored(roots, FULL, slug))
        full_sciq = load_scored(find_scored(roots, FULL, "sciq"))
        twin_topic = load_scored(find_scored(roots, twin_lbl, slug))
        twin_sciq = load_scored(find_scored(roots, twin_lbl, "sciq"))
        for method, per in METHODS.items():
            lbl = per[slug]
            m_topic = load_scored(find_scored(roots, lbl, slug))
            m_sciq = load_scored(find_scored(roots, lbl, "sciq"))
            for group in GROUPS:
                src = "sciq" if group == "unrelated_test" else slug
                mm, ff, tt = ((m_sciq, full_sciq, twin_sciq) if src == "sciq"
                              else (m_topic, full_topic, twin_topic))
                M, F, T = series(mm, group), series(ff, group), series(tt, group)
                ids = sorted(T)
                v.rec(f"{cname}/{method}/{group}: 50 questions", len(ids) == 50, f"n={len(ids)}")
                v.rec(f"{cname}/{method}/{group}: ids match across method/full/twin",
                      sorted(M) == ids and sorted(F) == ids,
                      f"method {len(M)}, full {len(F)}, twin {len(T)}")
                if group == "unrelated_test":
                    v.rec(f"{cname}/{method}: unrelated group is the SciQ set",
                          set(ids) == acc_sciq,
                          f"{len(set(ids) & acc_sciq)}/50 shared with the SciQ manifest")
                klm = kl_series(tt, mm, ids)
                klf = kl_series(tt, ff, ids)
                for i in ids:
                    nm, nf, nt = M[i]["nll"], F[i]["nll"], T[i]["nll"]
                    # NLL must be the mean over answer tokens, not a sum
                    if abs(nm - float(np.mean(M[i]["token_nlls"]))) > 1e-9:
                        raise SystemExit(f"{lbl}/{i}: nll != mean(token_nlls)")
                    rows.append({
                        "concept": cname, "method": method, "checkpoint": lbl,
                        "group": group, "question_id": i,
                        "nll_method": nm, "nll_full": nf, "nll_twin": nt,
                        "delta_nll_method_full": nm - nf,
                        "delta_nll_method_twin": nm - nt,
                        "delta_nll_full_twin": nf - nt,
                        "kl_twin_to_method": klm[i], "kl_twin_to_full": klf[i]})

    v.rec("900 per-question rows", len(rows) == 900, f"{len(rows)}")
    key = [(r["concept"], r["method"], r["group"], r["question_id"]) for r in rows]
    v.rec("no duplicate (concept, method, group, question_id)",
          len(set(key)) == len(key), f"{len(key)} rows, {len(set(key))} distinct")
    nonfinite = [r for r in rows for k in COLS_PQ[5:]
                 if isinstance(r[k], float) and not np.isfinite(r[k])]
    v.rec("all NLL/KL values finite", not nonfinite, f"{len(nonfinite)} non-finite")
    v.rec("only test rows", all(r["group"].endswith("_test") for r in rows), "every group ends _test")

    def w(name, cols, data):
        with open(OUT / name, "w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(cols), lineterminator="\n")
            wr.writeheader()
            for d in data:
                wr.writerow({c: (f"{d[c]:.10f}" if isinstance(d[c], float) else d[c]) for c in cols})

    # ---- summary -----------------------------------------------------------
    STATS = ("delta_nll_method_full", "delta_nll_method_twin", "delta_nll_full_twin",
             "kl_twin_to_method", "kl_twin_to_full")
    summary, ratios, selec = [], [], []
    for slug, cname, _ in CONCEPTS:
        for method, per in METHODS.items():
            lbl = per[slug]
            bygroup = {g: [r for r in rows if r["concept"] == cname and r["method"] == method
                           and r["group"] == g] for g in GROUPS}
            for g in GROUPS:
                sub = bygroup[g]
                rec = {"concept": cname, "method": method, "checkpoint": lbl,
                       "group": g, "n": len(sub)}
                for s in STATS:
                    x = np.array([r[s] for r in sub])
                    rec[f"mean_{s}"], rec[f"sd_{s}"] = float(x.mean()), sd(x)
                summary.append(rec)
            tgt = bygroup["target_test"]
            num = abs(float(np.mean([r["delta_nll_method_twin"] for r in tgt])))
            den = abs(float(np.mean([r["delta_nll_full_twin"] for r in tgt])))
            knum = float(np.mean([r["kl_twin_to_method"] for r in tgt]))
            kden = float(np.mean([r["kl_twin_to_full"] for r in tgt]))
            for nm, d in (("R_NLL", den), ("R_KL", kden)):
                if not np.isfinite(d) or d == 0:
                    raise SystemExit(f"{cname}/{method}: {nm} denominator invalid ({d})")
            ratios.append({"concept": cname, "method": method, "checkpoint": lbl,
                           "n": len(tgt), "R_NLL": num / den, "R_KL": knum / kden})
            gm = {g: np.array([r["delta_nll_method_full"] for r in bygroup[g]]) for g in GROUPS}
            s_nll = float(gm["target_test"].mean() - 0.5 * gm["neighbour_test"].mean()
                          - 0.5 * gm["unrelated_test"].mean())
            lo, hi = boot_ci(gm, BOOT_SEED, BOOT_N)
            selec.append({"concept": cname, "method": method, "checkpoint": lbl,
                          "target_mean": float(gm["target_test"].mean()),
                          "neighbour_mean": float(gm["neighbour_test"].mean()),
                          "sciq_mean": float(gm["unrelated_test"].mean()),
                          "S_NLL": s_nll, "ci95_low": lo, "ci95_high": hi})

    v.rec("18 summary rows (3 concepts x 2 methods x 3 groups)", len(summary) == 18, f"{len(summary)}")
    v.rec("6 ratio rows", len(ratios) == 6, f"{len(ratios)}")
    v.rec("6 selectivity rows", len(selec) == 6, f"{len(selec)}")

    print("Validation")
    for c in v.checks:
        print(f"  [{c['result']}] {c['check']}: {c['detail']}")
    if v.failed:
        print(f"\n{len(v.failed)} check(s) FAILED; nothing written.", file=sys.stderr)
        raise SystemExit(1)

    w("nll_kl_per_question.csv", COLS_PQ, rows)
    w("nll_kl_summary.csv", ["concept", "method", "checkpoint", "group", "n"]
      + [f"{p}_{s}" for s in STATS for p in ("mean", "sd")], summary)
    w("target_ratios.csv", ["concept", "method", "checkpoint", "n", "R_NLL", "R_KL"], ratios)
    w("nll_selectivity.csv", ["concept", "method", "checkpoint", "target_mean",
                              "neighbour_mean", "sciq_mean", "S_NLL", "ci95_low", "ci95_high"], selec)

    commit = subprocess.run(["git", "-C", str(HERE), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip() or "unknown"
    man = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": commit,
        "command": " ".join([Path(sys.executable).name] + sys.argv),
        "checkpoints": {"full": FULL,
                        "twins": {c[1]: c[2] for c in CONCEPTS},
                        "methods": {m: {c[1]: per[c[0]] for c in CONCEPTS}
                                    for m, per in METHODS.items()}},
        "groups": {g: 50 for g in GROUPS},
        "unrelated_group": "SciQ (sets/sciq.json); the older other-concept pool is NOT used",
        "metrics": {
            "NLL": "teacher-forced mean over the correct continuation's answer tokens",
            "KL": "mean over answer positions of KL(Twin || Model), full vocabulary",
            "KL_orientation": "twin is always the left/reference distribution",
            "R_NLL": "|mean(delta_nll_method_twin)| / |mean(delta_nll_full_twin)| on target_test",
            "R_KL": "mean(kl_twin_to_method) / mean(kl_twin_to_full) on target_test",
            "S_NLL": "mean(d_method_full|target) - 0.5*mean(|neighbour) - 0.5*mean(|unrelated)",
        },
        "bootstrap": {"repetitions": BOOT_N, "seed": BOOT_SEED,
                      "scheme": "resample each group's 50 questions independently with replacement",
                      "interval": "percentile 2.5 / 97.5"},
        "inputs": {"scored_roots": [str(r) for r in roots],
                   "sciq_manifest": a.sciq_manifest, "acc_manifest": a.acc_manifest},
        "outputs": sorted(str(OUT / f) for f in
                          ("nll_kl_per_question.csv", "nll_kl_summary.csv",
                           "target_ratios.csv", "nll_selectivity.csv")),
        "validation": {"n_checks": len(v.checks), "n_failed": 0, "checks": v.checks},
    }
    (OUT / "nll_kl_run_manifest.json").write_text(json.dumps(man, indent=1) + "\n")
    print(f"\nAll {len(v.checks)} checks passed. Wrote 4 CSVs + manifest to {OUT}")


if __name__ == "__main__":
    main()
