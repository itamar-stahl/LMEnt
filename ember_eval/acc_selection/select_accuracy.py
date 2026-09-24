#!/usr/bin/env python
"""Adapted EMBER-style accuracy selection: normalise, score, rank, freeze.

Selection uses ONLY the full model, the candidates, and the selection questions.
The concept-excluded twin is never loaded, referenced, or read here -- that is the
whole point: the research question is whether a rule blind to the twin still lands
near it. Nothing in this file opens a _test set either.

    Acc~_g(M) = clip_[0,1]( (Acc_g(M) - 0.25) / (Acc_g(F) - 0.25) )
    efficacy(M)     = 1 - Acc~_target(M)
    preservation(M) = HM( Acc~_neighbour(M), Acc~_unrelated(M) )
    H(M)            = HM( efficacy(M), preservation(M) )

HM is the repository's own ember.evals.harmonic.harmonic_mean, which returns 0.0
if either component is non-positive or non-finite.

If Acc_g(F) <= 0.25 for any group, the ratio is undefined: this refuses to compute
it, marks the topic, and stops selecting for it rather than silently clamping.

    python select.py --scored <dir> --out <dir>
"""
from __future__ import annotations

import argparse, csv, json, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "Ember-on-LMEnt"))
from ember.evals.harmonic import harmonic_mean  # noqa: E402

TOPICS = ("rome", "baseball", "ai")
METHODS = ("EMBER", "RMU", "SNMF", "RMU+EMBER", "SNMF+EMBER", "SNMF+EMBER-V1")
GROUPS = ("target_selection", "neighbour_selection", "unrelated_selection")
CHANCE = 0.25


def parse_config(label: str) -> dict:
    """Typed hyperparameters, and the aggressiveness key used for tie-breaks."""
    p = label.split("_")
    fam = p[0].lower()
    if fam == "ember":                      # ember_<topic>_d<delta>
        delta = float(p[2][1:])
        return {"method": "EMBER", "hp": f"delta={p[2][1:]}", "delta": delta,
                "tiebreak": (delta,), "tiebreak_desc": f"smaller delta ({delta:g})"}
    if fam == "rmu":                        # rmu_<topic>_L<layer><band>_a<alpha>
        m = re.match(r"L(\d+)(hi|mid)", p[2])
        layer, band = (int(m.group(1)), m.group(2)) if m else (0, "")
        alpha = float(p[3][1:]) if len(p) > 3 else 0.0
        return {"method": "RMU", "hp": f"layer={layer},band={band},alpha={alpha:g}",
                "layer": layer, "band": band, "alpha": alpha,
                "tiebreak": (layer, 0 if band == "mid" else 1, alpha),
                "tiebreak_desc": f"smaller (layer,band,alpha) ({layer},{band},{alpha:g})"}
    if fam == "snmf":                       # snmf_<topic>_<select>_<side>
        sel, side = p[2], (p[3] if len(p) > 3 else "")
        return {"method": "SNMF", "hp": f"select={sel},side={side}",
                "select": sel, "side": side,
                "tiebreak": (0 if sel == "ratio" else 1, {"in": 0, "out": 1, "both": 2}.get(side, 9)),
                "tiebreak_desc": f"repo defines no aggressiveness order for SNMF; "
                                 f"declared order (ratio<judge, in<out<both)"}
    # --- ensembles, added 2026-09-24 -------------------------------------
    # MLP method applied on top of the concept's selected EMBER checkpoint.
    # The normalisation, H, ranking and tie-break KEYS are deliberately
    # identical to the standalone families above: only the method name and the
    # label prefix differ, so an ensemble cell is ranked by exactly the rule its
    # standalone counterpart was. Re-running the original inventory through this
    # file reproduces the frozen 2026-09-23 winners unchanged.
    if fam == "rmuember":                   # rmuember_<topic>_L<layer><band>_a<alpha>
        m = re.match(r"L(\d+)(hi|mid)", p[2])
        layer, band = (int(m.group(1)), m.group(2)) if m else (0, "")
        alpha = float(p[3][1:]) if len(p) > 3 else 0.0
        return {"method": "RMU+EMBER", "hp": f"layer={layer},band={band},alpha={alpha:g}",
                "layer": layer, "band": band, "alpha": alpha,
                "tiebreak": (layer, 0 if band == "mid" else 1, alpha),
                "tiebreak_desc": f"smaller (layer,band,alpha) ({layer},{band},{alpha:g})"}
    if fam in ("snmfv1", "snmfv2"):         # snmfv<n>_<topic>_<select>_<side>
        sel, side = p[2], (p[3] if len(p) > 3 else "")
        method = "SNMF+EMBER" if fam == "snmfv2" else "SNMF+EMBER-V1"
        return {"method": method, "hp": f"select={sel},side={side}",
                "select": sel, "side": side,
                "tiebreak": (0 if sel == "ratio" else 1, {"in": 0, "out": 1, "both": 2}.get(side, 9)),
                "tiebreak_desc": f"repo defines no aggressiveness order for SNMF; "
                                 f"declared order (ratio<judge, in<out<both)"}
    raise SystemExit(f"unparseable candidate label: {label}")


def group_acc(scored: Path, label: str, topic: str) -> dict:
    f = scored / label / topic / "per_question.csv"
    if not f.exists():
        return {}
    rows = list(csv.DictReader(open(f, encoding="utf-8")))
    out = {}
    for g in GROUPS:
        sub = [r for r in rows if r["question_group"] == g]
        if not sub:
            continue
        margins = [float(r["gold_margin"]) for r in sub]
        n = len(sub); ncorr = sum(int(r["is_correct"]) for r in sub)
        mean = sum(margins) / n
        sd = (sum((m - mean) ** 2 for m in margins) / (n - 1)) ** 0.5 if n > 1 else 0.0
        out[g] = {"n": n, "n_correct": ncorr, "accuracy": ncorr / n,
                  "mean_margin": mean, "sd_margin": sd,
                  "se_margin": sd / (n ** 0.5) if n else 0.0,
                  "n_ties": sum(int(r["n_ties_at_top"]) for r in sub),
                  "n_nonfinite": sum(int(r["n_nonfinite"]) for r in sub)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True)
    ap.add_argument("--inventory", required=True, help="JSON list of [topic, method, label, path]")
    ap.add_argument("--full-label", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    scored, out = Path(a.scored), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    inv = json.loads(Path(a.inventory).read_text())

    # ---- full-model baseline, and the above-chance gate --------------------
    full, undefined = {}, {}
    for t in TOPICS:
        full[t] = group_acc(scored, a.full_label, t)
        if not full[t]:
            undefined[t] = "full model not scored"
            continue
        for g in GROUPS:
            if full[t][g]["accuracy"] <= CHANCE:
                undefined.setdefault(t, []) if isinstance(undefined.get(t), list) else None
                undefined[t] = (f"full-model accuracy on {g} is "
                                f"{full[t][g]['accuracy']:.4f} <= chance {CHANCE}; "
                                f"normalisation undefined")

    gacc_rows, norm_rows, rank_rows = [], [], []
    ties_log = []
    for t in TOPICS:
        if full.get(t):
            for g in GROUPS:
                s = full[t][g]
                gacc_rows.append({"model_label": a.full_label, "role": "full", "topic": t,
                                  "question_group": g, **s})
        for meth in METHODS:
            cands = [(lbl, pth) for (tt, mm, lbl, pth) in inv if tt == t and mm == meth]
            scored_c = []
            for lbl, pth in sorted(cands):
                ga = group_acc(scored, lbl, t)
                if not ga:
                    continue
                cfg = parse_config(lbl)
                for g in GROUPS:
                    gacc_rows.append({"model_label": lbl, "role": "candidate", "topic": t,
                                      "question_group": g, **ga[g]})
                if t in undefined:
                    continue
                comp = {}
                for g in GROUPS:
                    num = ga[g]["accuracy"] - CHANCE
                    den = full[t][g]["accuracy"] - CHANCE
                    unclipped = num / den
                    comp[g] = {"raw": ga[g]["accuracy"], "full_raw": full[t][g]["accuracy"],
                               "minus_chance": num, "full_minus_chance": den,
                               "unclipped": unclipped, "clipped": min(1.0, max(0.0, unclipped))}
                eff = 1.0 - comp["target_selection"]["clipped"]
                pres = harmonic_mean([comp["neighbour_selection"]["clipped"],
                                      comp["unrelated_selection"]["clipped"]])
                H = harmonic_mean([eff, pres])
                norm_rows.append({
                    "topic": t, "method": meth, "model_label": lbl, "hyperparameters": cfg["hp"],
                    "checkpoint_path": pth,
                    **{f"{k}_{g.split('_')[0]}": f"{comp[g][k]:.10f}"
                       for g in GROUPS for k in ("raw", "full_raw", "minus_chance",
                                                 "full_minus_chance", "unclipped", "clipped")},
                    "phi_efficacy": f"{eff:.10f}", "phi_preservation": f"{pres:.10f}",
                    "H_selection": f"{H:.10f}"})
                scored_c.append({"label": lbl, "path": pth, "cfg": cfg,
                                 "H": H, "pres": pres, "eff": eff})
            if t in undefined or not scored_c:
                continue
            scored_c.sort(key=lambda c: (-c["H"], -c["pres"], -c["eff"],
                                         c["cfg"]["tiebreak"], c["label"]))
            for i, c in enumerate(scored_c, 1):
                rank_rows.append({"topic": t, "method": meth, "rank": i,
                                  "model_label": c["label"], "hyperparameters": c["cfg"]["hp"],
                                  "checkpoint_path": c["path"],
                                  "H_selection": f"{c['H']:.10f}",
                                  "phi_preservation": f"{c['pres']:.10f}",
                                  "phi_efficacy": f"{c['eff']:.10f}",
                                  "tiebreak_key": str(c["cfg"]["tiebreak"]),
                                  "tiebreak_rule": c["cfg"]["tiebreak_desc"],
                                  "selected": "yes" if i == 1 else "no"})
            for i in range(len(scored_c) - 1):
                x, y = scored_c[i], scored_c[i + 1]
                if (x["H"], x["pres"], x["eff"]) == (y["H"], y["pres"], y["eff"]):
                    ties_log.append(f"{t}/{meth}: {x['label']} and {y['label']} tie on "
                                    f"(H,pres,eff)=({x['H']:.10f},{x['pres']:.10f},{x['eff']:.10f}); "
                                    f"broken by {x['cfg']['tiebreak_desc']}")

    def w(name, rows):
        if not rows: return
        with (out / name).open("w", newline="", encoding="utf-8") as fh:
            ww = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); ww.writeheader()
            for r in rows: ww.writerow(r)
    w("group_accuracies.csv", gacc_rows)
    w("normalized_scores.csv", norm_rows)
    w("rankings.csv", rank_rows)

    winners = [r for r in rank_rows if r["selected"] == "yes"]
    (out / "selected_checkpoints.json").write_text(json.dumps(
        {"n_winners": len(winners), "undefined_topics": undefined,
         "rule": "H = HM(1 - Acc~_target, HM(Acc~_neighbour, Acc~_unrelated)), "
                 "Acc~ = clip01((Acc - 0.25)/(Acc_full - 0.25)); selection sets only; "
                 "twin never loaded",
         "ties": ties_log, "winners": winners}, indent=1))
    print(f"group_accuracies {len(gacc_rows)} | normalized {len(norm_rows)} | "
          f"rankings {len(rank_rows)} | winners {len(winners)}")
    if undefined:
        print("\nUNDEFINED / STOPPED TOPICS:")
        for t, why in undefined.items(): print(f"  {t}: {why}")
    if ties_log:
        print("\nTIES:")
        for s in ties_log: print("  " + s)


if __name__ == "__main__":
    main()
