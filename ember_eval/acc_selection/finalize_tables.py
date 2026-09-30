#!/usr/bin/env python
"""Final aggregation: H_test, and sciq NLL/KL, for the frozen winners.

Nothing here loads a model. H_test reuses the same formula the selection used,
applied to the HELD-OUT half:

    Acc~_g(M)       = clip01( (Acc_g(M) - 0.25) / (Acc_g(F) - 0.25) )
    efficacy(M)     = 1 - Acc~_target(M)
    preservation(M) = HM( Acc~_neighbour(M), Acc~_unrelated(M) )
    H_test(M)       = HM( efficacy(M), preservation(M) )

It is computed under BOTH unrelated sources, because they are not
interchangeable: the original pool leaves the full model at 0.28-0.46 with CIs
spanning chance on two concepts, so its denominator is unsound even where the
winners happened to agree.

H_test is NOT the selection score and must not be read as one. These checkpoints
were chosen for low target accuracy on the SELECTION half, so part of that drop
was fitted noise and does not survive; the two halves differ by up to 14 points
on the full model alone.

sciq NLL/KL is read from the nll_kl scored tree, mean +- sample SD over the 50
held-out sciq questions, with KL keeping the twin as first argument throughout.
"""
from __future__ import annotations

import argparse, csv, json, sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "Ember-on-LMEnt"))
sys.path.insert(0, str(HERE.parents[0] / "nll_kl"))
from ember.evals.harmonic import harmonic_mean  # noqa: E402

CHANCE = 0.25
FULL = "lment-1b-control-2e-b131k"
FULL_ACC = "FULL_lment-1b-control-2e-b131k"
TWIN = {"rome": "lment-1b-norome-2e-b131k", "baseball": "lment-1b-nobaseball-2e-b131k",
        "ai": "lment-1b-noai-2e-b131k"}


def clip01(x): return min(1.0, max(0.0, x))


def kl_pair(ref, ev, ids):
    out = {}
    for i in ids:
        if i not in ref["index"] or i not in ev["index"]:
            continue
        s0, e0 = ref["index"][i]; s1, e1 = ev["index"][i]
        r = np.asarray(ref["dists"][s0:e0], dtype=np.float64)
        e = np.asarray(ev["dists"][s1:e1], dtype=np.float64)
        out[i] = float((np.exp(r) * (r - e)).sum(axis=1).mean())
    return out


def load_nk(scored: Path, label: str, topic: str):
    d = scored / label / topic
    if not (d / "records.json").exists():
        return None
    rec = json.loads((d / "records.json").read_text())
    o = {"nll": {r["id"]: r["nll"] for r in rec["records"] if r["phase"] == "test"},
         "dists": None, "index": None}
    if (d / "dists.npy").exists():
        o["dists"] = np.load(d / "dists.npy", mmap_mode="r")
        o["index"] = json.loads((d / "dists_index.json").read_text())
    return o


def ms(v):
    a = np.asarray(v, dtype=np.float64)
    return (float(a.mean()), float(a.std(ddof=1)) if a.size > 1 else 0.0, int(a.size))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollup", required=True)
    ap.add_argument("--rankings", required=True)
    ap.add_argument("--nk-scored", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    acc = {}
    for r in csv.DictReader(open(a.rollup, encoding="utf-8")):
        acc[(r["model"], r["topic"], r["group"], r["unrelated_source"])] = float(r["accuracy"])
    win = {(r["topic"], r["method"]): r["model_label"] for r in
           csv.DictReader(open(a.rankings, encoding="utf-8")) if r["selected"] == "yes"}

    # ---------- H_test -------------------------------------------------------
    rows = []
    for (t, m), lbl in sorted(win.items()):
        for src in ("sciq", "orig_pool"):
            tgt = acc.get((lbl, t, "target_test", ""))
            nbr = acc.get((lbl, t, "neighbour_test", ""))
            unr = acc.get((lbl, "(any)", "unrelated_test", "sciq")) if src == "sciq" \
                else acc.get((lbl, t, "unrelated_test", "orig_pool"))
            ftg = acc.get((FULL_ACC, t, "target_test", ""))
            fnb = acc.get((FULL_ACC, t, "neighbour_test", ""))
            fun = acc.get((FULL_ACC, "(any)", "unrelated_test", "sciq")) if src == "sciq" \
                else acc.get((FULL_ACC, t, "unrelated_test", "orig_pool"))
            if None in (tgt, nbr, unr, ftg, fnb, fun):
                continue
            den = {"target": ftg - CHANCE, "neighbour": fnb - CHANCE, "unrelated": fun - CHANCE}
            if min(den.values()) <= 0:
                rows.append({"topic": t, "method": m, "model": lbl, "unrelated_source": src,
                             "H_test": "undefined", "note": "a full-model denominator is <= 0"})
                continue
            at = clip01((tgt - CHANCE) / den["target"])
            an = clip01((nbr - CHANCE) / den["neighbour"])
            au = clip01((unr - CHANCE) / den["unrelated"])
            eff = 1.0 - at
            pres = harmonic_mean([an, au])
            rows.append({"topic": t, "method": m, "model": lbl, "unrelated_source": src,
                         "acc_target": tgt, "acc_neighbour": nbr, "acc_unrelated": unr,
                         "full_target": ftg, "full_neighbour": fnb, "full_unrelated": fun,
                         "min_denominator": round(min(den.values()), 4),
                         "norm_target": round(at, 6), "norm_neighbour": round(an, 6),
                         "norm_unrelated": round(au, 6),
                         "phi_efficacy": round(eff, 6), "phi_preservation": round(pres, 6),
                         "H_test": round(harmonic_mean([eff, pres]), 6), "note": ""})
    with (out / "H_test.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader()
        for r in rows: w.writerow(r)
    print(f"H_test.csv: {len(rows)} rows")

    # ---------- sciq NLL / KL ------------------------------------------------
    S = Path(a.nk_scored)
    full = load_nk(S, FULL, "sciq")
    if full is None:
        print("sciq NLL/KL: full model not scored yet -- skipping"); return
    twins = {t: load_nk(S, n, "sciq") for t, n in TWIN.items()}
    nrows = []
    for (t, m), lbl in sorted(win.items()):
        mod, tw = load_nk(S, lbl, "sciq"), twins[t]
        if mod is None or tw is None:
            print(f"  sciq NLL/KL missing for {t}/{m} ({lbl})"); continue
        ids = sorted(set(mod["nll"]) & set(full["nll"]) & set(tw["nll"]))
        dF = [mod["nll"][i] - full["nll"][i] for i in ids]
        dT = [mod["nll"][i] - tw["nll"][i] for i in ids]
        kl = kl_pair(tw, mod, ids) if (tw["dists"] is not None and mod["dists"] is not None) else {}
        mF, sF, n = ms(dF); mT, sT, _ = ms(dT)
        r = {"topic": t, "method": m, "model": lbl, "n": n,
             "sciq_dNLL_vs_full_mean": round(mF, 6), "sciq_dNLL_vs_full_sd": round(sF, 6),
             "sciq_dNLL_vs_twin_mean": round(mT, 6), "sciq_dNLL_vs_twin_sd": round(sT, 6)}
        if kl:
            mk, sk, _ = ms(list(kl.values()))
            r["sciq_KL_twin_to_model_mean"] = round(mk, 6)
            r["sciq_KL_twin_to_model_sd"] = round(sk, 6)
        nrows.append(r)
    # the twin-vs-full reference row, per concept
    for t, tw in twins.items():
        if tw is None: continue
        ids = sorted(set(tw["nll"]) & set(full["nll"]))
        mT, sT, n = ms([tw["nll"][i] - full["nll"][i] for i in ids])
        row = {"topic": t, "method": "TWIN(ref)", "model": TWIN[t], "n": n,
               "sciq_dNLL_vs_full_mean": round(mT, 6), "sciq_dNLL_vs_full_sd": round(sT, 6),
               "sciq_dNLL_vs_twin_mean": 0.0, "sciq_dNLL_vs_twin_sd": 0.0}
        if tw["dists"] is not None and full["dists"] is not None:
            mk, sk, _ = ms(list(kl_pair(tw, full, ids).values()))
            row["sciq_KL_twin_to_model_mean"] = round(mk, 6)   # KL(twin || full)
            row["sciq_KL_twin_to_model_sd"] = round(sk, 6)
            row["model"] = TWIN[t] + "  [KL row is KL(twin||FULL)]"
        nrows.append(row)
    if nrows:
        cols = sorted({k for r in nrows for k in r}, key=lambda c: (c != "topic", c))
        with (out / "sciq_nll_kl.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
            for r in nrows: w.writerow(r)
        print(f"sciq_nll_kl.csv: {len(nrows)} rows")


if __name__ == "__main__":
    main()
