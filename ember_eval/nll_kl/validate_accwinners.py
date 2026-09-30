#!/usr/bin/env python
"""Execute every validation the current-results export claims, and say so.

Twelve checks, each run against the artifacts rather than asserted in prose,
because this repository has already shipped one table built from a superseded
checkpoint selection and one metric that changed mid-run. Writes a JSON
summary for the run manifest and exits non-zero if anything fails.

    python ember_eval/nll_kl/validate_accwinners.py --run <nll_kl run root> \
        --out ember_eval/nll_kl/results_accwinners/validation.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

TOPICS = [("rome", "Ancient Rome"), ("baseball", "Baseball"), ("ai", "Artificial intelligence")]
METHODS = ("EMBER", "RMU", "SNMF")
ENSEMBLES = ("RMU+EMBER", "SNMF+EMBER")
GROUPS_TEST = ("target_test", "neighbour_test", "unrelated_test")
GROUPS_ALL = GROUPS_TEST + ("target_selection", "neighbour_selection", "unrelated_selection")

# the superseded NLL-rule winners; nothing here may come from these
SUPERSEDED = {
    "rome": {"EMBER": "ember_rome_d5000", "RMU": "rmu_rome_L6hi_a10", "SNMF": "snmf_rome_ratio_in"},
    "baseball": {"EMBER": "ember_baseball_d500", "RMU": "rmu_baseball_L5mid_a100",
                 "SNMF": "snmf_baseball_ratio_out"},
    "ai": {"EMBER": "ember_ai_d500", "RMU": "rmu_ai_L5mid_a100", "SNMF": "snmf_ai_ratio_in"},
}


class Checks:
    def __init__(self) -> None:
        self.out: List[Dict[str, Any]] = []

    def add(self, num: int, name: str, ok: bool, detail: str) -> None:
        self.out.append({"check": num, "name": name,
                         "result": "PASS" if ok else "FAIL", "detail": detail})
        print(f"  [{'PASS' if ok else 'FAIL'}] {num:2d}. {name}: {detail}")

    @property
    def failed(self) -> bool:
        return any(c["result"] == "FAIL" for c in self.out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="/home/morg/NLP_2526b/galbarak2/runs/nll_kl")
    ap.add_argument("--repo", default=str(REPO))
    ap.add_argument("--out")
    a = ap.parse_args()
    R, W = Path(a.run), Path(a.repo)
    C = Checks()

    res = {t: json.loads((W / f"ember_eval/nll_kl/results_accwinners/{t}/results.json").read_text())
           for t, _ in TOPICS}
    sel = json.loads((W / "ember_eval/acc_selection/results_sciq/selected_checkpoints.json").read_text())
    want = {(w["topic"], w["method"]): w for w in sel["winners"]}

    # 1 --------------------------------------------------------------------
    bad = []
    for t, _ in TOPICS:
        for m in METHODS:
            lab = res[t]["erased"].get(m)
            w = want.get((t, m))
            if w is None or lab != w["model_label"]:
                bad.append(f"{t}/{m}: {lab} != {w and w['model_label']}")
                continue
            meta = json.loads((R / "scored" / lab / t / "records.json").read_text())["meta"]
            if Path(meta["model"]).resolve() != Path(w["checkpoint_path"]).resolve():
                bad.append(f"{t}/{m}: scored {meta['model']} != selected {w['checkpoint_path']}")
    C.add(1, "every condition uses the latest accuracy-selected checkpoint", not bad,
          "9/9 labels and checkpoint paths match results_sciq/selected_checkpoints.json"
          if not bad else "; ".join(bad))

    # 2 --------------------------------------------------------------------
    changed, kept = [], []
    for t, _ in TOPICS:
        for m in METHODS:
            old, new = SUPERSEDED[t][m], res[t]["erased"][m]
            (kept if old == new else changed).append(f"{t}/{m}: {old} -> {new}")
    ok2 = all(res[t]["erased"][m] == want[(t, m)]["model_label"] for t, _ in TOPICS for m in METHODS)
    C.add(2, "no result comes from an older checkpoint-selection run", ok2,
          f"{len(changed)} of 9 cells differ from the superseded NLL-rule winners; "
          f"the {len(kept)} unchanged cells ({', '.join(x.split(':')[0] for x in kept)}) "
          "are cells where both rules chose the same checkpoint")

    # 3 --------------------------------------------------------------------
    bad = []
    for t, _ in TOPICS:
        for row in res[t]["rows"] + res[t]["kl_rows"]:
            per_group = defaultdict(list)
            for i in row["items"]:
                per_group[i["set"]].append(i["id"])
            for g, ids in per_group.items():
                if len(ids) != 50 or len(set(ids)) != 50:
                    bad.append(f"{t}/{row['label']}/{g}: {len(ids)} rows, {len(set(ids))} unique")
            if set(per_group) != set(GROUPS_ALL):
                bad.append(f"{t}/{row['label']}: groups {sorted(per_group)}")
    pq = W / "ember_eval/acc_selection/final/current_test_per_question.csv"
    acc_rows = list(csv.DictReader(open(pq, encoding="utf-8"))) if pq.exists() else []
    cells = defaultdict(list)
    for r in acc_rows:
        cells[(r["concept"], r["method"], r["question_group"])].append(r["question_id"])
    for k, ids in cells.items():
        if len(ids) != 50 or len(set(ids)) != 50:
            bad.append(f"accuracy {k}: {len(ids)} rows, {len(set(ids))} unique")
    C.add(3, "every evaluated group contains exactly 50 unique questions", not bad,
          f"{sum(len(r['items']) for t, _ in TOPICS for r in res[t]['rows'] + res[t]['kl_rows'])} "
          f"NLL/KL items in 6 groups x 50 per row, and {len(cells)} accuracy cells x 50"
          if not bad else "; ".join(bad[:6]))

    # 4 --------------------------------------------------------------------
    bad, n_pairs = [], 0
    for t, _ in TOPICS:
        labels = [res[t]["full"], res[t]["twin"]] + [res[t]["erased"][m] for m in METHODS]
        recs = {}
        for lab in labels:
            rr = json.loads((R / "scored" / lab / t / "records.json").read_text())["records"]
            recs[lab] = {r["id"]: tuple(r["completion_token_ids"]) for r in rr}
        ref = recs[labels[0]]
        for lab in labels[1:]:
            if set(recs[lab]) != set(ref):
                bad.append(f"{t}/{lab}: question ids differ from {labels[0]}")
                continue
            diff = [q for q in ref if recs[lab][q] != ref[q]]
            n_pairs += 1
            if diff:
                bad.append(f"{t}/{lab}: {len(diff)} completions differ from {labels[0]}")
    # accuracy side: same ids per (concept, group) across methods
    for (c, g) in sorted({(k[0], k[2]) for k in cells}):
        sets = {k[1]: set(v) for k, v in cells.items() if k[0] == c and k[2] == g}
        ref_m, ref_ids = sorted(sets.items())[0]
        for m, s in sets.items():
            if s != ref_ids:
                bad.append(f"accuracy {c}/{g}: {m} ids differ from {ref_m}")
    C.add(4, "all compared models use identical question IDs and answer continuations", not bad,
          f"{n_pairs} model pairs checked token-for-token across 3 topics; "
          "accuracy ids identical across methods within every (concept, group)"
          if not bad else "; ".join(bad[:6]))

    # 5 --------------------------------------------------------------------
    worst, n = 0.0, 0
    for t, _ in TOPICS:
        for lab in [res[t]["full"], res[t]["twin"]] + [res[t]["erased"][m] for m in METHODS]:
            for r in json.loads((R / "scored" / lab / t / "records.json").read_text())["records"]:
                d = abs(float(np.mean(r["token_nlls"])) - r["nll"])
                worst = max(worst, d)
                n += 1
                if len(r["token_nlls"]) != r["n_tokens"]:
                    worst = float("inf")
    C.add(5, "NLL equals the mean over correct-answer token NLLs", worst < 1e-9,
          f"recomputed from token_nlls for all {n} records; max |mean(token_nlls) - nll| = {worst:.3g}")

    # 6 + 8 ----------------------------------------------------------------
    def load_dists(lab: str, t: str):
        d = R / "scored" / lab / t
        return np.load(d / "dists.npy", mmap_mode="r"), json.loads((d / "dists_index.json").read_text())

    worst_kl, asym, neg, nkl = 0.0, [], 0, 0
    for t, _ in TOPICS:
        twin_d, twin_i = load_dists(res[t]["twin"], t)
        for row in res[t]["kl_rows"]:
            lab = res[t]["full"] if row["label"] == "Full" else res[t]["erased"].get(row["label"])
            if lab is None:
                continue
            ev_d, ev_i = load_dists(lab, t)
            items = [i for i in row["items"] if "kl" in i]
            for i in items:
                if i["kl"] < -1e-9:
                    neg += 1
            nkl += len(items)
            for i in items[:5]:                      # spot-recompute from the raw distributions
                s0, e0 = twin_i[i["id"]]
                s1, e1 = ev_i[i["id"]]
                p = np.exp(np.asarray(twin_d[s0:e0], dtype=np.float64))
                fwd = float((p * (np.asarray(twin_d[s0:e0], dtype=np.float64)
                                  - np.asarray(ev_d[s1:e1], dtype=np.float64))).sum(axis=1).mean())
                worst_kl = max(worst_kl, abs(fwd - i["kl"]))
                q = np.exp(np.asarray(ev_d[s1:e1], dtype=np.float64))
                rev = float((q * (np.asarray(ev_d[s1:e1], dtype=np.float64)
                                  - np.asarray(twin_d[s0:e0], dtype=np.float64))).sum(axis=1).mean())
                asym.append(abs(fwd - rev))
    C.add(6, "KL is full-vocabulary KL(twin || model), twin on the left", worst_kl < 1e-6,
          f"recomputed sum_v p_twin(v)(log p_twin - log p_model) over the full vocabulary for "
          f"{len(asym)} spot-checked items; max |recomputed - stored| = {worst_kl:.3g}. "
          f"The reversed direction differs by up to {max(asym):.3g} nats, so the direction is "
          f"not a no-op. kl_convention field: {res['rome']['kl_convention']!r}")

    # 7 --------------------------------------------------------------------
    bad = []
    for t, _ in TOPICS:
        for row in res[t]["rows"] + res[t]["kl_rows"]:
            for i in row["items"]:
                for k in ("nll_eval", "nll_ref", "delta", "kl"):
                    if k in i and not np.isfinite(i[k]):
                        bad.append(f"{t}/{row['label']}/{i['id']}/{k}")
    n_scores = 0
    for r in acc_rows:
        for k in ("correct_option_score", "option_0_score", "option_1_score",
                  "option_2_score", "option_3_score"):
            n_scores += 1
            if not np.isfinite(float(r[k])):
                bad.append(f"accuracy {r['concept']}/{r['method']}/{r['question_id']}/{k}")
    rk = W / "ember_eval/nll_kl/rkl_bootstrap.csv"
    rk_rows = list(csv.DictReader(open(rk, encoding="utf-8"))) if rk.exists() else []
    for r in rk_rows:
        for k in ("R_KL", "ci_lo", "ci_hi", "mean_kl_edited", "mean_kl_full"):
            if not np.isfinite(float(r[k])):
                bad.append(f"rkl {r['concept']}/{r['method']}/{k}")
    C.add(7, "all NLL, KL, option-score and ratio values are finite", not bad,
          f"{nkl} KL values, {sum(len(r['items']) for t, _ in TOPICS for r in res[t]['rows'])} NLL "
          f"items, {n_scores} option scores and {len(rk_rows) * 5} ratio/interval values"
          if not bad else "; ".join(bad[:6]))

    C.add(8, "KL values are nonnegative up to floating-point error", neg == 0,
          f"{nkl} KL values, none below -1e-9; minimum is "
          f"{min(i['kl'] for t, _ in TOPICS for r in res[t]['kl_rows'] for i in r['items'] if 'kl' in i):.3g}")

    # 9 --------------------------------------------------------------------
    bad = []
    for t, name in TOPICS:
        md = (W / f"ember_eval/nll_kl/tables_accwinners/{t}/table.md").read_text()
        for row in res[t]["kl_rows"]:
            for g, pretty in zip(GROUPS_TEST, ("Target", "Neighbour", "Unrelated")):
                mean = row["per_set"][g]["kl"]["mean"]
                if f"**{mean:.4f}**" not in md:
                    bad.append(f"{t}/{row['label']}/{g}: {mean:.4f} not in the committed table")
        for row in res[t]["rows"]:
            for g in GROUPS_TEST:
                mean = row["per_set"][g]["nll_diff"]["mean"]
                if f"**{mean:+.3f}**" not in md:
                    bad.append(f"{t}/{row['label']}/{g}: {mean:+.3f} not in the committed table")
    C.add(9, "computed point estimates reproduce the current paper tables", not bad,
          "every KL and NLL-difference mean in the three results.json appears at the "
          "displayed precision in tables_accwinners/<topic>/table.md"
          if not bad else "; ".join(bad[:6]))

    # 10 -------------------------------------------------------------------
    tr = {(r["concept"], r["method"]): r for r in
          csv.DictReader(open(W / "ember_eval/ensembles/results/target_ratios.csv", encoding="utf-8"))}
    bad, worst_r = [], 0.0
    for r in rk_rows:
        k = (r["concept"], r["method"])
        if r["method"] not in ENSEMBLES:
            continue
        if k not in tr:
            bad.append(f"{k}: absent from target_ratios.csv")
            continue
        d = abs(float(r["R_KL"]) - float(tr[k]["R_KL"]))
        worst_r = max(worst_r, d)
        if d > 1e-6:
            bad.append(f"{k}: {float(r['R_KL']):.6f} vs {float(tr[k]['R_KL']):.6f}")
        if r["checkpoint"] != tr[k]["checkpoint"]:
            bad.append(f"{k}: checkpoint {r['checkpoint']} vs {tr[k]['checkpoint']}")
    C.add(10, "the six ensemble ratios agree with ensembles/results/target_ratios.csv", not bad,
          f"6/6 R_KL agree; max |difference| = {worst_r:.3g}" if not bad else "; ".join(bad))

    # 11 -------------------------------------------------------------------
    agg = {(r["concept"], r["method"], r["question_group"]): float(r["accuracy"])
           for r in csv.DictReader(
               open(W / "ember_eval/acc_selection/final/current_test_accuracy.csv", encoding="utf-8"))}
    recomputed = defaultdict(lambda: [0, 0])
    for r in acc_rows:
        k = (r["concept"], r["method"], r["question_group"])
        recomputed[k][0] += int(r["is_correct"])
        recomputed[k][1] += 1
    bad = []
    for k, (c_, n_) in recomputed.items():
        if abs(c_ / n_ - agg.get(k, -1)) > 1e-9:
            bad.append(f"{k}: aggregate {agg.get(k)} != recomputed {c_ / n_}")

    slug = {name: t for t, name in TOPICS}
    n_h = 0
    for r in csv.DictReader(open(W / "ember_eval/acc_selection/final/H_test.csv", encoding="utf-8")):
        concept = [n for t, n in TOPICS if t == r["topic"]][0]
        g_un = "sciq_unrelated_test" if r["unrelated_source"] == "sciq" else "unrelated_test"
        for col, g in (("acc_target", "target_test"), ("acc_neighbour", "neighbour_test"),
                       ("acc_unrelated", g_un)):
            k = (concept, r["method"], g)
            n_h += 1
            if abs(agg.get(k, -1) - float(r[col])) > 1e-9:
                bad.append(f"H_test {r['topic']}/{r['method']}/{col}: "
                           f"{r[col]} != {agg.get(k)} from the per-question export")
        for col, g, m in (("full_target", "target_test", "Full"),
                          ("full_neighbour", "neighbour_test", "Full"),
                          ("full_unrelated", g_un, "Full")):
            n_h += 1
            if abs(agg.get((concept, m, g), -1) - float(r[col])) > 1e-9:
                bad.append(f"H_test {r['topic']}/{col}: {r[col]} != {agg.get((concept, m, g))}")
    n_p = 0
    for r in csv.DictReader(
            open(W / "ember_eval/ensembles/results/panel_A_accuracy_sciq.csv", encoding="utf-8")):
        for col, g in (("acc_target", "target_test"), ("acc_neighbour", "neighbour_test"),
                       ("acc_unrelated", "sciq_unrelated_test")):
            n_p += 1
            k = (r["concept"], r["method"], g)
            if abs(agg.get(k, -1) - float(r[col])) > 1e-9:
                bad.append(f"panel_A_sciq {r['concept']}/{r['method']}/{col}: "
                           f"{r[col]} != {agg.get(k)}")
    n_roll = 0
    label_of = {(r["concept"], r["method"]): r["checkpoint"] for r in
                csv.DictReader(open(W / "ember_eval/acc_selection/final/current_test_accuracy.csv",
                                    encoding="utf-8"))}
    for r in csv.DictReader(
            open(W / "ember_eval/acc_selection/test_phase/test_accuracy_rollup.csv", encoding="utf-8")):
        if r["topic"] == "(any)":
            g, concepts = "sciq_unrelated_test", [n for _, n in TOPICS]
        else:
            g, concepts = r["group"], [n for t, n in TOPICS if t == r["topic"]]
        for concept in concepts:
            k = [(c, m, gg) for (c, m, gg) in agg
                 if c == concept and gg == g and label_of.get((c, m)) == r["model"]]
            if not k:
                continue
            n_roll += 1
            if abs(agg[k[0]] - float(r["accuracy"])) > 1e-9:
                bad.append(f"rollup {r['model']}/{r['topic']}/{r['group']}: "
                           f"{r['accuracy']} != {agg[k[0]]}")
    C.add(11, "accuracy aggregates agree with the current accuracy and H_test inputs", not bad,
          f"{len(recomputed)} aggregates recomputed from the {len(acc_rows)} per-question rows; "
          f"{n_h} values cross-checked against final/H_test.csv, {n_p} against "
          f"ensembles panel_A_accuracy_sciq.csv and {n_roll} against test_accuracy_rollup.csv"
          if not bad else "; ".join(bad[:8]))

    # 12 -------------------------------------------------------------------
    want_rows = {(n, m) for _, n in TOPICS for m in METHODS + ENSEMBLES}
    got = {(r["concept"], r["method"]) for r in rk_rows}
    C.add(12, "the final R_KL output contains all 15 expected method-concept rows",
          len(rk_rows) == 15 and got == want_rows,
          f"{len(rk_rows)} rows, 3 concepts x 5 conditions, none missing"
          if got == want_rows else f"missing {sorted(want_rows - got)}, extra {sorted(got - want_rows)}")

    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(C.out, indent=1) + "\n")
        print("\nwrote", a.out)
    return 1 if C.failed else 0


if __name__ == "__main__":
    sys.exit(main())
