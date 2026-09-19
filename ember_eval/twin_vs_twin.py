#!/usr/bin/env python
"""Use the OTHER untaught twin as the reference, instead of the shared control.

BASEBALL_RESULTS.md's correction section shows that `ablated - control` on
questions is contaminated: a twin that never ablated baseball reproduces 76% of
Baseball's apparent QA "ablation effect". There is one shared control behind
every arm, so any run-level idiosyncrasy of that single run enters every
contrast identically.

Two ablated twins exist, so the offset can be cancelled by differencing them
instead: for Ancient Rome use `norome2e - nobaseball2e`, for Baseball the
reverse. Both sides are then ablated twins and whatever they share drops out.

The point is not to restate the effects. It is to TEST the explanation. The
null spread for `twin - control` decomposes into ~0.044 sampling and ~0.046
systematic (EVALUATION-style variance decomposition). If the systematic half is
control-side, this estimator's null should lose it. If it does not shrink, the
offset is per-twin, the explanation is wrong, and this estimator is no better.

Nothing here needs a GPU or the dataset: it reads stored completion records.
"""
from __future__ import annotations
import argparse, glob, json, math, statistics as st
from pathlib import Path

CMPL = "/home/dcor/galbarak2/LMEnt-ember/ember_eval/results/completion"
SPLITS = ["QA_train", "QA_test", "SimdomQA_train", "SimdomQA_test"]
ARMS = {"rome": "norome2e", "baseball": "nobaseball2e"}


def pmi_per_char(recs):
    return [r["pmi"][r["correct_index"]] / r["option_chars"][r["correct_index"]]
            for r in recs]


def load(model, concept, split):
    g = sorted(glob.glob(f"{CMPL}/cmpl_{model}_{concept}_{split}_*.json"),
               key=lambda p: int(p.rsplit("_", 1)[1].split(".")[0]))
    if not g:
        return None
    return pmi_per_char(json.load(open(g[-1]))["records"])


def paired(a, b):
    """mean, dz, t, n for a - b, paired element-wise."""
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    m = sum(d) / n
    sd = st.stdev(d)
    return {"mean": m, "dz": m / sd if sd else float("nan"),
            "t": m / (sd / math.sqrt(n)) if sd else float("nan"),
            "se": sd / math.sqrt(n), "n": n}


def decompose(means, ses, label):
    between = st.stdev(means)
    within = sum(ses) / len(ses)
    extra = between ** 2 - within ** 2
    systematic = math.sqrt(extra) if extra > 0 else 0.0
    print(f"  {label:22s} between {between:.4f}   sampling {within:.4f}   "
          f"systematic {systematic:.4f}   (n_concepts={len(means)})")
    return {"between_sd": between, "sampling_se": within,
            "systematic_sd": systematic, "n_concepts": len(means)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/home/dcor/galbarak2/lment-rome-check/results/twin_vs_twin.json")
    a = ap.parse_args()

    all_conc = sorted({Path(p).name.split("_")[2]
                       for p in glob.glob(f"{CMPL}/cmpl_control2e_*_QA_train_*.json")})
    null_conc = [c for c in all_conc if c not in ARMS]
    print(f"concepts with data: {all_conc}")
    print(f"null concepts (neither twin ablated): {null_conc}\n")

    out = {"null_concepts": null_conc, "effects": {}, "null": {}, "decomposition": {}}

    print("=== EFFECTS: own twin minus the OTHER twin, vs the published own-minus-control ===")
    for conc, own in ARMS.items():
        other = ARMS["baseball" if conc == "rome" else "rome"]
        for s in SPLITS:
            o, x, c = load(own, conc, s), load(other, conc, s), load("control2e", conc, s)
            if None in (o, x, c):
                continue
            tt, tc = paired(o, x), paired(o, c)
            out["effects"][f"{conc}/{s}"] = {"twin_minus_twin": tt, "twin_minus_control": tc}
            print(f"  {conc:9s} {s:15s} twin-twin {tt['mean']:+.4f} (dz {tt['dz']:+.2f}, t {tt['t']:+.2f})"
                  f"   | twin-control {tc['mean']:+.4f} (dz {tc['dz']:+.2f})")

    print("\n=== THE TEST: does the null lose its systematic component? ===")
    for s in SPLITS:
        tt_m, tt_se, tc_m, tc_se = [], [], [], []
        per = {}
        for c in null_conc:
            o, x, ctl = load(ARMS["baseball"], c, s), load(ARMS["rome"], c, s), load("control2e", c, s)
            if None in (o, x, ctl):
                continue
            a1, a2 = paired(o, x), paired(o, ctl)
            tt_m.append(a1["mean"]); tt_se.append(a1["se"])
            tc_m.append(a2["mean"]); tc_se.append(a2["se"])
            per[c] = {"twin_minus_twin": a1["mean"], "twin_minus_control": a2["mean"]}
        if len(tt_m) < 3:
            continue
        print(f"\n{s}:")
        d_tc = decompose(tc_m, tc_se, "twin - control")
        d_tt = decompose(tt_m, tt_se, "twin - twin")
        out["null"][s] = {"per_concept": per,
                          "twin_minus_control": {"mean": sum(tc_m)/len(tc_m), **d_tc},
                          "twin_minus_twin": {"mean": sum(tt_m)/len(tt_m), **d_tt}}
        # z for the two arms on each estimator
        for conc in ARMS:
            k = f"{conc}/{s}"
            if k not in out["effects"]:
                continue
            e = out["effects"][k]
            z_tt = (e["twin_minus_twin"]["mean"] - sum(tt_m)/len(tt_m)) / d_tt["between_sd"]
            z_tc = (e["twin_minus_control"]["mean"] - sum(tc_m)/len(tc_m)) / d_tc["between_sd"]
            e["z_twin_minus_twin"] = z_tt
            e["z_twin_minus_control"] = z_tc
            print(f"     {conc:9s} z: twin-twin {z_tt:+.2f}   twin-control {z_tc:+.2f}")

    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
